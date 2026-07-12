import elements
import embodied
import numpy as np


class HomeGrid(embodied.Env):

  def __init__(
      self, task, size=(64, 64), lang='token', obs_read_step=True,
      max_steps=100, num_trashobjs=2, num_trashcans=2, p_teleport=0.05,
      p_unsafe=0.0):
    # Language streaming (T5 tokenization, one token per step, padding
    # between utterances, preread phase) is implemented by the homegrid
    # package itself; this adapter only converts API and curates keys.
    assert task in ('task', 'future', 'dynamics', 'corrections'), task
    assert lang in ('token', 'token_embed'), lang
    import homegrid
    self._env = homegrid.HomeGrid(
        lang_types={
            'task': ['task'],
            'future': ['task', 'future'],
            'dynamics': ['task', 'dynamics'],
            'corrections': ['task', 'corrections'],
        }[task],
        max_steps=max_steps, num_trashobjs=num_trashobjs,
        num_trashcans=num_trashcans, p_teleport=p_teleport,
        p_unsafe=p_unsafe)
    self._size = tuple(size)
    self._lang = lang
    self._obs_read_step = obs_read_step
    spaces = self._env.observation_space.spaces
    self._imgres = spaces['image'].shape[:2]
    if self._imgres != self._size:
      from PIL import Image
      self._Image = Image
    self._vocab = int(spaces['token'].high) if lang == 'token' else None
    self._embed = spaces['token_embed'].shape if lang == 'token_embed' else None
    self._done = True

  @property
  def obs_space(self):
    if self._lang == 'token':
      lang = {'token': elements.Space(np.int32, (), 0, self._vocab)}
    else:
      lang = {'token_embed': elements.Space(np.float32, self._embed)}
    read = {'is_read_step': elements.Space(bool)} if self._obs_read_step else {}
    return {
        'image': elements.Space(np.uint8, (*self._size, 3)),
        **lang,
        **read,
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs, _ = self._env.reset()
      return self._obs(obs, 0.0, is_first=True)
    obs, reward, terminated, truncated, _ = self._env.step(
        int(action['action']))
    self._done = terminated or truncated
    return self._obs(
        obs, reward, is_last=self._done, is_terminal=terminated)

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    image = obs['image']
    if self._imgres != self._size:
      image = np.asarray(self._Image.fromarray(image).resize(
          self._size, self._Image.NEAREST))
    if self._lang == 'token':
      lang = {'token': np.int32(obs['token'])}
    else:
      lang = {'token_embed': np.float32(obs['token_embed'])}
    read = {}
    if self._obs_read_step:
      read = {'is_read_step': bool(obs['is_read_step'])}
    return dict(
        image=image,
        **lang,
        **read,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )
