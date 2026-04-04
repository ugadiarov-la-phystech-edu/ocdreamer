import embodied
import numpy as np

from gym import Wrapper, spaces
from homegrid.base import (
    FloorWithObject, Inanimate, Pickable, Storage, Wall,
    CENTERED_VIEW, USE_AGENT_TEXTURE, AGENT_TEXTURE,
)
from homegrid import rendering
from homegrid.rendering import point_in_triangle, rotate_fn
from homegrid.wrappers import _COMPOSITE_GROUPS, PART_TO_WHOLE
from PIL import Image, ImageFont, ImageDraw


class HomeGrid(embodied.Env):

  def __init__(
    self,
    task,
    size=(64, 64),
    max_steps=100,
    num_trashobjs=2,
    num_trashcans=2,
    p_teleport=0.05,
    p_unsafe=0.,
    fixed_state=None,
    vis=False,
    seg_mode='none',
  ):
    from . import from_gym
    import homegrid
    import gym
    assert task in ("task", "future", "dynamics", "corrections")
    env = gym.make(f"homegrid-{task}", 
                   disable_env_checker=True,
                   max_steps=max_steps,
                   num_trashobjs=num_trashobjs,
                   num_trashcans=num_trashcans,
                   p_teleport=p_teleport,
                   p_unsafe=p_unsafe,
                   fixed_state=fixed_state)
    if seg_mode!='none':
      env = SegmentationMaskWrapper(env, seg_mode=seg_mode)
    env = homegrid.wrappers.Gym26Wrapper(env)
    self._env = env
    self.observation_space = self._env.observation_space
    self.action_space = self._env.action_space
    self.wrappers = [
      from_gym.FromGym,
      lambda e: embodied.wrappers.ResizeImage(e, (64,64)),
    ]
    self.vis = vis

  # @property
  # def obs_space(self):
  #   return self.observation_space
  
  # @property
  # def act_space(self):
  #   return self.action_space
  
  def reset(self):
    obs = self._env.reset()
    if self.vis:
      obs["log_image"] = self.render_with_text(obs["log_language_info"])
    return obs

  def step(self, action):
    result = self._env.step(action)
    obs, rew, done, info = result
    if self.vis:
      obs["log_image"] = self.render_with_text(obs["log_language_info"])
    return obs, rew, done, info

  def render(self):
    return self._env.render(mode="rgb_array")

  def render_with_text(self, text):
    img = self._env.render(mode="rgb_array")
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, (0, 0, 0))
    draw.text((0, 45), "Action: {}".format(self._env.prev_action), (0, 0, 0))
    img = np.asarray(img)
    return img

  def init_from_state(self, state):
    self._env.init_from_state(state)


class SegmentationMaskWrapper(Wrapper):
    """Adds ground-truth per-object segmentation masks to observations.

    Adds to obs:
      - 'seg_mask': (H, W) uint8 array, pixel-resolution segmentation.
         0 = background, each object gets a unique ID.
      - 'seg_labels': dict mapping int ID -> object name string.

    Compatible with old-style gym API (Gym26Wrapper returns obs from reset,
    (obs, rew, done, info) from step).

    Args:
        seg_mode: 'class' — all objects with the same name share one ID.
                  'instance' — each object occurrence gets a unique ID.
                  Floors and walls always use class-level IDs in both modes.
    """

    def __init__(self, env, tile_size=32, agent_pov=True, seg_mode="class"):
        super().__init__(env)
        assert seg_mode in ("class", "instance"), f"seg_mode must be 'class' or 'instance', got '{seg_mode}'"
        self.tile_size = tile_size
        self.agent_pov = agent_pov
        self.seg_mode = seg_mode

        if agent_pov:
            h = w = env.agent_view_size
        else:
            w, h = env.width, env.height

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces,
             "seg_mask": spaces.Box(
                 low=0, high=255,
                 shape=(h * tile_size, w * tile_size),
                 dtype="uint8",
             )}
        )

    def _get_alpha_mask(self, texture):
        """Return a boolean mask of opaque pixels after resizing texture to tile_size."""
        ts = self.tile_size
        tex = rendering.resize(texture, (ts, ts))
        if tex.shape[-1] == 4:
            return tex[:, :, 3] == 255
        # No alpha channel — fully opaque
        return np.ones((ts, ts), dtype=bool)

    def _get_obj_texture(self, obj):
        """Get the raw texture (with alpha) for a dynamic cell object."""
        if isinstance(obj, Storage):
            return obj.textures[obj.state]
        elif isinstance(obj, Pickable):
            if obj.invisible:
                return None
            return obj.texture
        elif isinstance(obj, Inanimate):
            return obj.texture
        return None

    @staticmethod
    def _build_composite_keys(grid, h, w):
        """Group adjacent tiles that are parts of the same composite object.

        Returns a dict mapping (x, y) -> instance_key, where all tiles
        belonging to the same physical object share the same key (the
        minimum (x, y) in the connected component).
        """
        # Map each cell to its composite group name (if any)
        cell_group = {}
        for j in range(h):
            for i in range(w):
                floor = grid.get_floor(i, j)
                if floor is not None and isinstance(floor, FloorWithObject) and "_" in floor.name:
                    _, obj_name = floor.name.split("_", 1)
                    if obj_name in _COMPOSITE_GROUPS:
                        cell_group[(i, j)] = _COMPOSITE_GROUPS[obj_name]

        # Flood-fill adjacent cells with the same group
        visited = set()
        composite_key = {}
        for pos in sorted(cell_group):
            if pos in visited:
                continue
            group = cell_group[pos]
            # BFS to find connected component
            component = []
            queue = [pos]
            while queue:
                p = queue.pop()
                if p in visited:
                    continue
                if cell_group.get(p) != group:
                    continue
                visited.add(p)
                component.append(p)
                x, y = p
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                        queue.append((nx, ny))
            # All cells in this component share the minimum position as key
            key = min(component)
            for p in component:
                composite_key[p] = key
        return composite_key

    def _build_seg_mask(self):
        if self.agent_pov:
            grid, vis_mask = self.gen_obs_grid()
            h, w = grid.height, grid.width
            if CENTERED_VIEW:
                agent_pos = (grid.width // 2, grid.height // 2)
            else:
                agent_pos = (grid.width // 2, grid.height - 1)
        else:
            grid = self.grid
            h, w = grid.height, grid.width
            agent_pos = tuple(self.agent_pos)

        ts = self.tile_size
        seg = np.zeros((h * ts, w * ts), dtype=np.uint8)
        name_to_id = {}
        id_to_name = {0: "background"}
        next_id = 1
        all_textures = self.unwrapped.textures
        instance_mode = self.seg_mode == "instance"

        # Floors and walls are "stuff" — always class-level IDs.
        STUFF_CLASSES = {"background", "wall", "tile", "carpet", "wood"}

        # Pre-compute composite instance keys so adjacent parts
        # (e.g. rugl+rugr) share one instance ID.
        composite_key = self._build_composite_keys(grid, h, w)

        def get_id(name, instance_key=None):
            """Get or allocate a mask ID.

            In class mode (or for stuff classes), name alone determines the ID.
            In instance mode, instance_key differentiates same-name objects.
            """
            nonlocal next_id
            name = PART_TO_WHOLE.get(name, name)
            if not instance_mode or name in STUFF_CLASSES or instance_key is None:
                key = name
            else:
                key = (name, instance_key)
            if key not in name_to_id:
                name_to_id[key] = next_id
                id_to_name[next_id] = name
                next_id += 1
            return name_to_id[key]

        for j in range(h):
            for i in range(w):
                cell = grid.get(i, j)
                floor = grid.get_floor(i, j)

                ymin = j * ts
                ymax = (j + 1) * ts
                xmin = i * ts
                xmax = (i + 1) * ts
                tile = seg[ymin:ymax, xmin:xmax]

                # Floor layer — split composite FloorWithObject into
                # base floor + static object using the alpha channel.
                if floor is not None:
                    if isinstance(floor, FloorWithObject) and "_" in floor.name:
                        base_name, obj_name = floor.name.split("_", 1)
                        tile[:] = get_id(base_name)
                        # Use composite key so adjacent parts share one ID
                        inst_key = composite_key.get((i, j), (i, j))
                        if obj_name in all_textures:
                            alpha = self._get_alpha_mask(all_textures[obj_name])
                            tile[alpha] = get_id(obj_name, instance_key=inst_key)
                        else:
                            tile[:] = get_id(floor.name, instance_key=inst_key)
                    else:
                        tile[:] = get_id(floor.name)

                # Dynamic object — use alpha for pixel-level mask
                if cell is not None and not isinstance(cell, Wall):
                    tex = self._get_obj_texture(cell)
                    obj_key = id(cell)
                    if tex is not None:
                        alpha = self._get_alpha_mask(tex)
                        tile[alpha] = get_id(cell.name, instance_key=obj_key)
                    else:
                        tile[:] = get_id(cell.name, instance_key=obj_key)

                # Wall
                if cell is not None and isinstance(cell, Wall):
                    tile[:] = get_id("wall")

                # Carried object — drawn at agent position behind the agent sprite
                if (i, j) == agent_pos:
                    carried = self.unwrapped.carrying
                    if carried is not None:
                        tex = self._get_obj_texture(carried)
                        carried_key = id(carried)
                        if tex is not None:
                            alpha = self._get_alpha_mask(tex)
                            tile[alpha] = get_id(carried.name, instance_key=carried_key)
                        else:
                            tile[:] = get_id(carried.name, instance_key=carried_key)

                # Agent — use robot texture alpha + direction arrow
                if (i, j) == agent_pos:
                    agent_id = get_id("agent", instance_key="agent")
                    if USE_AGENT_TEXTURE:
                        alpha = self._get_alpha_mask(AGENT_TEXTURE)
                        tile[alpha] = agent_id
                    else:
                        tile[:] = agent_id
                    # Include the red direction arrow in the agent mask
                    agent_dir = self.unwrapped.agent_dir
                    if CENTERED_VIEW:
                        tri_fn = point_in_triangle(
                            (0.65, 0.29), (0.87, 0.50), (0.65, 0.71))
                    else:
                        tri_fn = point_in_triangle(
                            (0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
                    tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5,
                                       theta=0.5 * 3.141592653589793 * agent_dir)
                    for py in range(ts):
                        for px in range(ts):
                            yf = (py + 0.5) / ts
                            xf = (px + 0.5) / ts
                            if tri_fn(xf, yf):
                                tile[py, px] = agent_id

        return seg, id_to_name

    def _add_seg(self, obs):
        seg_mask, seg_labels = self._build_seg_mask()
        return {**obs, "seg_mask": seg_mask, "seg_labels": seg_labels}

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            return self._add_seg(obs), info
        return self._add_seg(result)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return self._add_seg(obs), reward, terminated, truncated, info
        obs, reward, done, info = result
        return self._add_seg(obs), reward, done, info
