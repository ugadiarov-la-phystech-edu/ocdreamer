import gym

import embodied
import numpy as np

from gym import Wrapper
from PIL import Image, ImageFont, ImageDraw


class MoveFieldsFromObservationToInfoWrapper(Wrapper):
  def __init__(self, env: gym.Env):
    super().__init__(env)
    observation_space = dict(env.observation_space)
    self._fields_to_move = {'log_language_info', 'is_read_step'}
    for key in self._fields_to_move:
      del observation_space[key]

    self._observation_space = gym.spaces.Dict(observation_space)

  def _move_fields_from_observation_to_info(self, observation, info):
    for key in self._fields_to_move:
      info[key] = observation.pop(key)

  def step(self, action):
    observation, reward, terminated, truncated, info = super().step(action)
    self._move_fields_from_observation_to_info(observation, info)

    return observation, reward, terminated, truncated, info

  def reset(self):
    observation, info = super().reset()
    self._move_fields_from_observation_to_info(observation, info)

    return observation, info


class UnwrapNumpyActionWrapper(Wrapper):
  def __init__(self, env: gym.Env):
    super().__init__(env)

  def step(self, action):
    action = action.item() if isinstance(action, np.ndarray) else action
    return super().step(action)


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
    self._gym_env = UnwrapNumpyActionWrapper(MoveFieldsFromObservationToInfoWrapper(env))
    self._env = embodied.wrappers.ResizeImage(from_gym.FromGym(self._gym_env, is_022_gym_api=False), size)
    self.vis = vis

  @property
  def obs_space(self):
    obs_space = dict(self._env.obs_space)

    return obs_space

  @property
  def act_space(self):
    return self._env.act_space

  def _render_with_text(self, text):
    img = self._gym_env.render(mode="rgb_array")
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, (0, 0, 0))
    draw.text((0, 45), "Action: {}".format(self._gym_env.prev_action), (0, 0, 0))
    img = np.asarray(img)
    return img

  def step(self, action):
    obs = self._env.step(action)
    if self.vis:
      self._env._info['log_image'] = self._render_with_text(obs["log_language_info"])

    return obs

  @property
  def info(self):
    return self._env.info

  def render(self):
    return self._env.render()

  def close(self):
    self._env.close()
