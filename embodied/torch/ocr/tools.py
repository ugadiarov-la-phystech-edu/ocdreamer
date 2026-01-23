import math

# Types
from typing import TypeVar, Optional

import torch
import torchvision.transforms
import numpy as np

from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from embodied.torch.ocr.dinov2_saur.modules.encoders import TimmExtractor, FrameEncoder
from embodied.torch.ocr.dinov2_saur.modules.groupers import SlotAttention
from embodied.torch.ocr.dinov2_saur.modules.initializers import RandomInit, FixedLearnedInit
from embodied.torch.ocr.dinov2_saur.modules.networks import build
from embodied.torch.ocr.dinov2_saur.modules.video import LatentProcessor

Tensor = TypeVar("torch.tensor")
NN = TypeVar("torch.nn")

# [B, D, H, W] -> [B, N, D]
img_to_slot = lambda x: x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


# [B, N, D] -> [B, D, H, W]
def slot_to_img(slot):
    B, N, D = slot.shape
    size = int(math.sqrt(N))
    return slot.reshape(B, size, size, D).permute(0, 3, 1, 2)


def preprocessing_obs(obs, device, type="image"):
    ret = torch.Tensor(obs.copy()).to(device).unsqueeze(0)
    if type == "image":
        return ret.permute(0, 3, 1, 2) / 255
    elif type == "state":
        return ret


# upload batch to working device
def to_device(batch, device):
    if type(batch) == type([]):
        for i in range(len(batch)):
            batch[i] = batch[i].to(device)
    elif type(batch) == type({}):
        for k in batch.keys():
            batch[k] = batch[k].to(device)
    else:
        batch = batch.to(device)
    return batch


# get_item from pytorch tensor
def get_item(x):
    if len(x.shape) == 0:
        return x.item()
    else:
        return x.detach().cpu().numpy()


# reshape image for visualization
for_viz = lambda x: np.array(
    x.clamp(0, 1).permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0, dtype=np.uint8
)


# Taken from https://github.com/singhgautam/slate/blob/master/slate.py
def visualize(images):
    B, _, H, W = images[0].shape  # first image is observation
    viz_imgs = []
    for _img in images:
        if len(_img.shape) == 4:
            viz_imgs.append(_img)
        else:
            viz_imgs += [object_image.expand_as(viz_imgs[0]) for object_image in torch.unbind(_img, dim=1)]
    viz_imgs = torch.cat(viz_imgs, dim=-1)
    # return torch.cat(torch.unbind(viz_imgs,dim=0), dim=-2).unsqueeze(0)
    return viz_imgs


# hungarian matching
def hungarian_matching(target, input, return_diff_mat=False):
    tN, tD = target.shape
    iN, iD = input.shape
    assert tN == iN and tD == iD
    diff_mat = np.zeros((tN, iN))
    for t in range(tN):
        for i in range(iN):
            diff_mat[t, i] = torch.norm(target[t] - input[i], p=1).item()
    _, col_ind = linear_sum_assignment(diff_mat)
    if return_diff_mat:
        return torch.LongTensor(col_ind).to(target.device), diff_mat[:, col_ind]
    else:
        return torch.LongTensor(col_ind).to(target.device)


# calculate ARI
def calculate_ari(true_masks, pred_masks):
    true_masks = true_masks.flatten(2)
    pred_masks = pred_masks.flatten(2)

    true_mask_ids = get_item(torch.argmax(true_masks, dim=1))
    pred_mask_ids = get_item(torch.argmax(pred_masks, dim=1))

    aris = []
    for b in range(true_mask_ids.shape[0]):
        aris.append(adjusted_rand_score(true_mask_ids[b], pred_mask_ids[b]))

    return aris


# change img numpy array to torch Tensor
def obs_to_tensor(obs, device):
    if len(obs.shape) == 4:
        return torch.Tensor(obs.transpose(0, 3, 1, 2)).to(device) / 255.0
    else:
        return torch.Tensor(obs).to(device)


class DinoV2saur(torch.nn.Module):
    def __init__(self, config_path, checkpoint_path, device):
        super().__init__()
        self._device = device
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._config = OmegaConf.load(self._config_path).model
        self._normalization = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        initializer_kwargs = dict(self._config.initializer)
        initializer_name = initializer_kwargs.pop("name")
        self.initializer = self._instantiate(initializer_name, [RandomInit, FixedLearnedInit], initializer_kwargs)

        backbone_kwargs = dict(self._config.encoder.backbone)
        backbone_name = backbone_kwargs.pop("name")
        backbone = self._instantiate(backbone_name, [TimmExtractor], backbone_kwargs)

        output_transform_config = self._config.encoder.output_transform
        output_transform = build(output_transform_config, 'two_layer_mlp')

        encoder_config = self._config.encoder
        self.encoder = FrameEncoder(backbone=backbone, pos_embed=None, output_transform=output_transform,
                                     spatial_flatten=False, main_features_key=encoder_config.get("main_features_key", "vit_block12"))

        grouper_config = self._config.grouper
        grouper = SlotAttention(inp_dim=grouper_config.inp_dim, slot_dim=grouper_config.slot_dim,
                                     n_iters=grouper_config.n_iters, use_mlp=grouper_config.use_mlp, )
        self.processor = LatentProcessor(grouper, predictor=None)

        state_dict = torch.load(self._checkpoint_path, weights_only=False)['state_dict']
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        assert len(missing_keys) == 0, f'{missing_keys}'
        assert all(key.startswith('decoder') for key in unexpected_keys)

        self.to(self._device)
        self.requires_grad_(False)
        self.eval()

    @property
    def n_slots(self):
        return self._config.initializer.n_slots

    @property
    def dim(self):
        return self._config.initializer.dim

    @staticmethod
    def _instantiate(class_name, classes, kwargs):
        for cls in classes:
            if cls.__name__ == class_name:
                return cls(**kwargs)

        raise ValueError(f'Unknown class name: {class_name}')

    def forward(self, image, previous_slots=None):
        encoder_input = self._normalization(image)
        batch_size = image.size()[0]

        encoder_output = self.encoder(encoder_input)
        features = encoder_output["features"]

        slots_initial = previous_slots
        if slots_initial is None:
            slots_initial = self.initializer(batch_size=batch_size)

        processor_output = self.processor(slots_initial, features)
        return processor_output["state"]

    def get_slots(self, images, previous_slots, to_numpy=True):
        one_image = len(images.shape) == 3
        if one_image:
            batch_images = images[np.newaxis, ...]
        else:
            batch_images = images

        if previous_slots is not None and one_image:
            batch_prev_slots = previous_slots[np.newaxis, ...]
        else:
            batch_prev_slots = previous_slots

        batch_images = torch.as_tensor(batch_images.transpose(0, 3, 1, 2), dtype=torch.float32, device=self._device) / 255.0
        if batch_prev_slots is not None:
            batch_prev_slots = torch.as_tensor(batch_prev_slots, dtype=torch.float32, device=self._device)

        slots = self(batch_images, previous_slots=batch_prev_slots).detach()
        if one_image:
            slots = slots[0]

        if to_numpy:
            slots = slots.cpu().numpy()

        return slots


def tensor_to_one_hot(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Convert tensor to one-hot encoding by using maximum across dimension as one-hot element."""
    assert 0 <= dim
    max_idxs = torch.argmax(tensor, dim=dim, keepdim=True)
    shape = [1] * dim + [-1] + [1] * (tensor.ndim - dim - 1)
    one_hot = max_idxs == torch.arange(tensor.shape[dim], device=tensor.device).view(*shape)
    return one_hot.to(torch.long)


def adjusted_rand_index(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    """Computes adjusted Rand index (ARI), a clustering similarity score.

    This implementation ignores points with no cluster label in `true_mask` (i.e. those points for
    which `true_mask` is a zero vector). In the context of segmentation, that means this function
    can ignore points in an image corresponding to the background (i.e. not to an object).

    Implementation adapted from https://github.com/deepmind/multi_object_datasets and
    https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py

    Args:
        pred_mask: Predicted cluster assignment encoded as categorical probabilities of shape
            (batch_size, n_points, n_pred_clusters).
        true_mask: True cluster assignment encoded as one-hot of shape (batch_size, n_points,
            n_true_clusters).

    Returns:
        ARI scores of shape (batch_size,).
    """
    n_pred_clusters = pred_mask.shape[-1]
    pred_cluster_ids = torch.argmax(pred_mask, axis=-1)

    # Convert true and predicted clusters to one-hot ('oh') representations. We use float64 here on
    # purpose, otherwise mixed precision training automatically casts to FP16 in some of the
    # operations below, which can create overflows.
    true_mask_oh = true_mask.to(torch.float64)  # already one-hot
    pred_mask_oh = torch.nn.functional.one_hot(pred_cluster_ids, n_pred_clusters).to(torch.float64)

    n_ij = torch.einsum("bnc,bnk->bck", true_mask_oh, pred_mask_oh)
    a = torch.sum(n_ij, axis=-1)
    b = torch.sum(n_ij, axis=-2)
    n_fg_points = torch.sum(a, axis=1)

    rindex = torch.sum(n_ij * (n_ij - 1), axis=(1, 2))
    aindex = torch.sum(a * (a - 1), axis=1)
    bindex = torch.sum(b * (b - 1), axis=1)
    expected_rindex = aindex * bindex / torch.clamp(n_fg_points * (n_fg_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both true_mask and pred_mask assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == n_fg_points * (n_fg_points-1))
    # 2. If both true_mask and pred_mask assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    return torch.where(denominator > 0, ari, torch.ones_like(ari))


def fg_adjusted_rand_index(
    pred_mask: torch.Tensor, true_mask: torch.Tensor, bg_dim: int = 0
) -> torch.Tensor:
    """Compute adjusted random index using only foreground groups (FG-ARI).

    Args:
        pred_mask: Predicted cluster assignment encoded as categorical probabilities of shape
            (batch_size, n_points, n_pred_clusters).
        true_mask: True cluster assignment encoded as one-hot of shape (batch_size, n_points,
            n_true_clusters).
        bg_dim: Index of background class in true mask.

    Returns:
        ARI scores of shape (batch_size,).
    """
    n_true_clusters = true_mask.shape[-1]
    assert 0 <= bg_dim < n_true_clusters
    if bg_dim == 0:
        true_mask_only_fg = true_mask[..., 1:]
    elif bg_dim == n_true_clusters - 1:
        true_mask_only_fg = true_mask[..., :-1]
    else:
        true_mask_only_fg = torch.cat(
            (true_mask[..., :bg_dim], true_mask[..., bg_dim + 1 :]), dim=-1
        )

    return adjusted_rand_index(pred_mask, true_mask_only_fg)


class ARIMetric:
    """Computes ARI metric."""

    def __init__(
        self,
        foreground: bool = True,
        convert_target_one_hot: bool = False,
        ignore_overlaps: bool = False,
        background_dim: int = 0
    ):
        super().__init__()
        self.foreground = foreground
        self.background_dim = background_dim
        self.convert_target_one_hot = convert_target_one_hot
        self.ignore_overlaps = ignore_overlaps
        self.values = 0
        self.total = 0

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, ignore: Optional[torch.Tensor] = None
    ):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of classes.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
            ignore: Ignore mask of shape (B, 1, H, W) or (B, 1, K, H, W)
        """
        if prediction.ndim == 5:
            # Merge frames, height and width to single dimension.
            prediction = prediction.transpose(1, 2).flatten(-3, -1)
            target = target.transpose(1, 2).flatten(-3, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).transpose(1, 2).flatten(-3, -1)
        elif prediction.ndim == 4:
            # Merge height and width to single dimension.
            prediction = prediction.flatten(-2, -1)
            target = target.flatten(-2, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).flatten(-2, -1)
        else:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        if self.ignore_overlaps:
            overlaps = (target > 0).sum(1, keepdim=True) > 1
            if ignore is None:
                ignore = overlaps
            else:
                ignore = ignore | overlaps

        if ignore is not None:
            assert ignore.ndim == 3 and ignore.shape[1] == 1
            prediction = prediction.clone()
            prediction[ignore.expand_as(prediction)] = 0
            target = target.clone()
            target[ignore.expand_as(target)] = 0

        # Make channels / gt labels the last dimension.
        prediction = prediction.transpose(-2, -1)
        target = target.transpose(-2, -1)

        if self.convert_target_one_hot:
            target_oh = tensor_to_one_hot(target, dim=2)
            # For empty pixels (all values zero), one-hot assigns 1 to the first class, correct for
            # this (then it is technically not one-hot anymore).
            target_oh[:, :, 0][target.sum(dim=2) == 0] = 0
            target = target_oh

        # Should be either 0 (empty, padding) or 1 (single object).
        assert torch.all(target.sum(dim=-1) < 2), "Issues with target format, mask non-exclusive"

        if self.foreground:
            ari = fg_adjusted_rand_index(prediction, target, bg_dim=self.background_dim)
        else:
            ari = adjusted_rand_index(prediction, target)

        self.values += ari.sum().item()
        self.total += len(ari)

    def compute(self):
        return self.values / self.total


if __name__ == "__main__":
    config_path = '/samsung/projects/ocdreamer/embodied/torch/ocr/dinov2_saur/config/homegrid_base14_dinov2_n-slot-7.yaml'
    checkpoint_path = '/samsung/projects/videosaur/checkpoint/homegrid/n-slot-7/checkpoints/step=497500.ckpt'
    dinosaur = DinoV2saur(config_path, checkpoint_path, 'cuda')
    # state_dict = torch.load(checkpoint_path, weights_only=False)['state_dict']
    # missing_keys, unexpected_keys = dinosaur.load_state_dict(state_dict, strict=False)
    # assert len(missing_keys) == 0, f'{missing_keys}'
    # assert all(key.startswith('decoder') for key in unexpected_keys)
    # print()