import elements
import embodied
import numpy as np


class DummySlots(embodied.Env):

  def __init__(self, task, size=(4, 5), length=100):
    del task
    self.size = size
    self.length = length
    self.count = 0
    self.done = False

  @property
  def obs_space(self):
    return {
        'slots': elements.Space(np.float32, self.size),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'reset': elements.Space(bool),
        'act_disc': elements.Space(np.int32, (), 0, 5),
        'act_cont': elements.Space(np.float32, (6,)),
    }

  def step(self, action):
    if action.pop('reset') or self.done:
      self.count = 0
      self.done = False
      return self._obs(0, is_first=True)
    self.count += 1
    self.done = (self.count >= self.length)
    return self._obs(1, is_last=self.done, is_terminal=self.done)

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
    return dict(
        slots=np.ones(self.size, np.float32),
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )
