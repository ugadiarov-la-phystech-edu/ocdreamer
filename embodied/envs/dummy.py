import elements
import embodied
import numpy as np


class Dummy(embodied.Env):

  def __init__(self, task, size=(64, 64), length=100):
    del task
    self.size = size
    self.length = length
    self.count = 0
    self.done = False

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, self.size + (3,)),
        'vector': elements.Space(np.float32, (7,)),
        'token': elements.Space(np.int32, (), 0, 256),
        'count': elements.Space(np.float32, (), 0, self.length),
        'float2d': elements.Space(np.float32, (4, 5)),
        'int2d': elements.Space(np.int32, (2, 3), 0, 4),
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
        image=np.full(self.size + (3,), 255, np.uint8),
        vector=np.zeros(7, np.float32),
        token=np.zeros((), np.int32),
        count=np.float32(self.count),
        float2d=np.ones((4, 5), np.float32),
        int2d=np.ones((2, 3), np.int32),
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )



class CustomDummySlot(embodied.Env):

  def __init__(self, task, size=(3, 8), length=100):
    supported_tasks = {'disc', 'cont'}
    assert task in supported_tasks, f'{task} is not in {supported_tasks}'
    self.task = task
    self.size = size
    self.length = length
    self.count = 0
    self.done = False

  @property
  def obs_space(self):
    return {
        'slot': elements.Space(np.float32, self.size),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    if self.task == 'disc':
      action_space = elements.Space(np.int32, (), 0, 5)
    else:
      action_space = elements.Space(np.float32, (6,))

    return {
        'reset': elements.Space(bool),
        'action': action_space,
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
        slot=np.zeros(self.size, np.float32),
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )


class CustomDummy(embodied.Env):

  def __init__(self, task, size=(64, 64), length=100):
    supported_tasks = {'disc', 'cont'}
    assert task in supported_tasks, f'{task} is not in {supported_tasks}'
    self.task = task
    self.size = size
    self.length = length
    self.count = 0
    self.done = False

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, self.size + (3,)),
        'vector': elements.Space(np.float32, (7,)),
        'token': elements.Space(np.int32, (), 0, 256),
        'count': elements.Space(np.float32, (), 0, self.length),
        'float2d': elements.Space(np.float32, (4, 5)),
        'int2d': elements.Space(np.int32, (2, 3), 0, 4),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    if self.task == 'disc':
      action_space = elements.Space(np.int32, (), 0, 5)
    else:
      action_space = elements.Space(np.float32, (6,))

    return {
        'reset': elements.Space(bool),
        'action': action_space,
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
        image=np.full(self.size + (3,), 255, np.uint8),
        vector=np.zeros(7, np.float32),
        token=np.zeros((), np.int32),
        count=np.float32(self.count),
        float2d=np.ones((4, 5), np.float32),
        int2d=np.ones((2, 3), np.int32),
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )


class CustomDummySlot(embodied.Env):

  def __init__(self, task, size=(3, 8), length=100):
    supported_tasks = {'disc', 'cont'}
    assert task in supported_tasks, f'{task} is not in {supported_tasks}'
    self.task = task
    self.size = size
    self.length = length
    self.count = 0
    self.done = False

  @property
  def obs_space(self):
    return {
        'slot': elements.Space(np.float32, self.size),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    if self.task == 'disc':
      action_space = elements.Space(np.int32, (), 0, 5)
    else:
      action_space = elements.Space(np.float32, (6,))

    return {
        'reset': elements.Space(bool),
        'action': action_space,
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
        slot=np.zeros(self.size, np.float32),
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )

class CustomDummySlotText(embodied.Env):

  def __init__(self, task, size=(3, 8), length=100):
    supported_tasks = {'disc', 'cont'}
    assert task in supported_tasks, f'{task} is not in {supported_tasks}'
    self.task = task
    self.size = size
    self.length = length
    self.count = 0
    self.done = False

  @property
  def obs_space(self):
    return {
        'slot': elements.Space(np.float32, self.size),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'token': elements.Space(np.uint32, (),  0, 32100),
        'token_embed': elements.Space(np.float32, (512,), -np.inf, np.inf),
    }

  @property
  def act_space(self):
    if self.task == 'disc':
      action_space = elements.Space(np.int32, (), 0, 5)
    else:
      action_space = elements.Space(np.float32, (6,))

    return {
        'reset': elements.Space(bool),
        'action': action_space,
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
        slot=np.zeros(self.size, np.float32),
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        token=np.zeros((), np.uint32),
        token_embed=np.zeros((512,), np.float32), 
    )