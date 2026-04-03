import torch
import torch.nn.functional as F
import torchvision
from omegaconf import OmegaConf

from embodied.torch.ocr.slotcontrast import configuration, modules

from embodied.torch.ocr.tools import SlotExtractor

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


class SlotContrastExtractor(torch.nn.Module, SlotExtractor):
    def __init__(self, config_path, checkpoint_path, image_size, device, backbone_input_size=0):
        super().__init__()
        self._device = device
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._image_size = image_size
        self._backbone_input_size = backbone_input_size

        config = configuration.load_config(config_path)
        self._model_config = config.model

        self._normalization = torchvision.transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        )

        # Build model components from slotcontrast config
        self.initializer = modules.build_initializer(self._model_config.initializer)
        self.encoder = modules.build_encoder(self._model_config.encoder, "FrameEncoder")

        grouper = modules.build_grouper(self._model_config.grouper)

        input_type = self._model_config.get("input_type", "image")
        if input_type == "image":
            self.processor = modules.LatentProcessor(grouper, predictor=None)
        elif input_type == "video":
            self.encoder = modules.MapOverTime(self.encoder)
            predictor = None
            if self._model_config.predictor is not None:
                from embodied.torch.ocr.slotcontrast.modules.utils import build_module
                predictor = build_module(self._model_config.predictor)
            if self._model_config.latent_processor:
                self.processor = modules.build_video(
                    self._model_config.latent_processor,
                    "LatentProcessor",
                    corrector=grouper,
                    predictor=predictor,
                )
            else:
                self.processor = modules.LatentProcessor(grouper, predictor)
            self.processor = modules.ScanOverTime(self.processor)
        else:
            raise ValueError(f"Unknown input type {input_type}")

        self._input_type = input_type

        # Load checkpoint weights
        state_dict = torch.load(
            self._checkpoint_path, weights_only=False, map_location='cpu'
        )
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Filter to only the modules we have (initializer, encoder, processor)
        filtered_state_dict = {}
        for key, value in state_dict.items():
            for prefix in ('initializer.', 'encoder.', 'processor.'):
                if key.startswith(prefix):
                    filtered_state_dict[key] = value
                    break

        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
        assert len(missing_keys) == 0, f'Missing keys: {missing_keys}'
        assert len(unexpected_keys) == 0, f'Unexpected keys: {unexpected_keys}'

        self.to(self._device)
        self.requires_grad_(False)
        self.eval()

    @property
    def n_slots(self):
        return self._model_config.initializer.n_slots

    @property
    def dim(self):
        return self._model_config.initializer.dim

    @property
    def backbone_input_size(self):
        return self._backbone_input_size if self._backbone_input_size else None

    def forward(self, image, previous_slots=None):
        encoder_input = self._normalization(image)
        batch_size = image.size()[0]

        # For video input_type, encoder/processor are wrapped in MapOverTime/ScanOverTime
        # which expect a time dimension (B, T, ...). Add a fake T=1 dimension.
        if self._input_type == "video":
            encoder_input = encoder_input.unsqueeze(1)  # (B, C, H, W) -> (B, 1, C, H, W)

        encoder_output = self.encoder(encoder_input)
        features = encoder_output["features"]

        slots_initial = previous_slots
        if slots_initial is None:
            slots_initial = self.initializer(batch_size=batch_size)

        processor_output = self.processor(slots_initial, features)
        slots = processor_output["state"]

        # Remove the fake time dimension
        if self._input_type == "video":
            slots = slots[:, 0]  # (B, 1, n_slots, dim) -> (B, n_slots, dim)

        return slots
