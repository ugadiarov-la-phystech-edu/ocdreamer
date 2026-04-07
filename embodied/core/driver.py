import elements
import numpy as np


class Driver:

  def __init__(self, batch_env, **kwargs):
    assert batch_env.n_envs >= 1
    self.kwargs = kwargs
    self.length = batch_env.n_envs
    self.batch_env = batch_env
    self.act_space = self.batch_env.act_space
    self.callbacks = []
    self.acts = None
    self.carry = None
    self.reset()

  def reset(self, init_policy=None):
    self.acts = {
        k: np.zeros((self.length,) + v.shape, v.dtype)
        for k, v in self.act_space.items()}
    self.acts['reset'] = np.ones(self.length, bool)
    self.carry = init_policy and init_policy(self.length)

  def close(self):
    self.batch_env.close()

  def on_step(self, callback):
    self.callbacks.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    acts = self.acts
    assert all(len(x) == self.length for x in acts.values())
    assert all(isinstance(v, np.ndarray) for v in acts.values())
    acts = [{k: v[i] for k, v in acts.items()} for i in range(self.length)]
    obs = self.batch_env.step(acts)
    logs = {k: v for k, v in obs.items() if k.startswith('log')}
    obs = {k: v for k, v in obs.items() if not k.startswith('log')}
    assert all(len(x) == self.length for x in obs.values()), obs
    self.carry, acts, outs = policy(self.carry, obs, **self.kwargs)
    assert all(k not in acts for k in outs), (
        list(outs.keys()), list(acts.keys()))
    if obs['is_last'].any():
      mask = ~obs['is_last']
      acts = {k: self._mask(v, mask) for k, v in acts.items()}
    self.acts = {**acts, 'reset': obs['is_last'].copy()}
    trans = {**obs, **acts, **outs, **logs}
    for i in range(self.length):
      trn = elements.tree.map(lambda x: x[i], trans)
      [fn(trn, i, **self.kwargs) for fn in self.callbacks]
    step += len(obs['is_first'])
    episode += obs['is_last'].sum()
    return step, episode

  def _mask(self, value, mask):
    while mask.ndim < value.ndim:
      mask = mask[..., None]
    return value * mask.astype(value.dtype)
