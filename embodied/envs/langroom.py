"""LangRoom environment wrapper for dreamerv3.

Creates the underlying langroom.LangRoom env and optionally adds
ground-truth per-object segmentation masks to observations.
"""

import numpy as np

import elements
import embodied
from langroom.langroom import LangRoom as _LangRoom
from langroom.langroom import OBJECTS


class LangRoom(embodied.Env):

    def __init__(
        self,
        task,
        length=200,
        vocab_size=15,
        seed=None,
        seg_mode='none',
    ):
        self._env = _LangRoom(
            task=task,
            length=length,
            vocab_size=vocab_size,
            seed=seed,
        )
        self.seg_mode = seg_mode

    @property
    def obs_space(self):
        spaces = dict(self._env.obs_space)
        if self.seg_mode != 'none':
            res = self._env.resolution
            spaces['seg_mask'] = elements.Space(np.uint8, (res, res), 0, 255)
        return spaces

    @property
    def act_space(self):
        return self._env.act_space

    def step(self, action):
        obs = self._env.step(action)
        if self.seg_mode != 'none':
            seg_mask, seg_labels = self._build_seg_mask()
            obs = {**obs, 'seg_mask': seg_mask, 'seg_labels': seg_labels}
        return obs

    def _build_seg_mask(self):
        """Build segmentation mask matching the render() output exactly."""
        env = self._env
        view = 2 * env.view + 1
        grid = int(np.floor(env.resolution / view))
        res = env.resolution

        # Precomputed layout (transposed, same as in LangRoom)
        layout = env.layout
        textures = env.textures

        name_to_id = {}
        id_to_name = {0: 'background'}
        next_id = 1
        instance_mode = self.seg_mode == 'instance'

        # Walls and floors are "stuff" — always class-level.
        STUFF = {'background', 'wall', 'floor'}

        def get_id(name, instance_key=None):
            nonlocal next_id
            if not instance_mode or name in STUFF or instance_key is None:
                key = name
            else:
                key = (name, instance_key)
            if key not in name_to_id:
                name_to_id[key] = next_id
                id_to_name[next_id] = name
                next_id += 1
            return name_to_id[key]

        seg = np.zeros((view * grid, view * grid), dtype=np.uint8)

        for dx in range(-env.view, env.view + 1):
            for dy in range(-env.view, env.view + 1):
                x = env.player[0] + dx
                y = env.player[1] + dy

                i = dx + env.view
                j = dy + env.view
                ymin = i * grid
                ymax = (i + 1) * grid
                xmin = j * grid
                xmax = (j + 1) * grid
                tile = seg[ymin:ymax, xmin:xmax]

                # Out of bounds → wall
                if not (0 <= x < layout.shape[0] and 0 <= y < layout.shape[1]):
                    tile[:] = get_id('wall')
                    continue

                cell = layout[x, y]

                # Player position (drawn on top of floor)
                if (x, y) == env.player:
                    # Floor underneath
                    tile[:] = get_id('floor')
                    # Player sprite with alpha
                    tex = textures['player']
                    alpha = tex[..., -1] > 0.5  # (grid, grid)
                    tile[alpha] = get_id('player', instance_key='player')

                # Object position (digit in layout)
                elif cell in [str(n) for n in range(len(OBJECTS))]:
                    obj_name = OBJECTS[int(cell)]
                    tile[:] = get_id('floor')
                    tex = textures[obj_name]
                    alpha = tex[..., -1] > 0.5
                    inst_key = (x, y) if instance_mode else None
                    tile[alpha] = get_id(obj_name, instance_key=inst_key)

                # Wall
                elif cell == '#':
                    tile[:] = get_id('wall')

                # Floor
                elif cell == ' ':
                    tile[:] = get_id('floor')

        # Transpose to match render() which does .transpose((1, 0, 2))
        seg = seg.T

        # Pad to full resolution (same as render())
        pad = (res - seg.shape[0]) // 2
        if pad > 0:
            seg = np.pad(seg, ((pad, pad), (pad, pad)),
                         mode='constant', constant_values=0)

        return seg, id_to_name
