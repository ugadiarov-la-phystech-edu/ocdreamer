import embodied
import numpy as np
from .remove_text import RemoveTextObs


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
    env = homegrid.wrappers.Gym26Wrapper(env)
    env = RemoveTextObs(env)
    self._env = env
    self.observation_space = self._env.observation_space
    self.action_space = self._env.action_space
    self.wrappers = [
      from_gym.FromGym,
      lambda e: embodied.wrappers.ResizeImage(e, (64,64)),
    ]

  # @property
  # def obs_space(self):
  #   return self.observation_space
  
  # @property
  # def act_space(self):
  #   return self.action_space
  
  def reset(self):
    obs, info = self._env.reset()
    return obs

  def step(self, action):
    obs, reward, terminated, truncated, info = self._env.step(action)
    return obs, reward, terminated, truncated, info

  def render(self):
    return self._env.render(mode="rgb_array")

  def init_from_state(self, state):
    self._env.init_from_state(state)
