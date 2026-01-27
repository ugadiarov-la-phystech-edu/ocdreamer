import math
import re

import einops
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
i32 = jnp.int32
sg = jax.lax.stop_gradient

from embodied.jax.nets import Transformer, ObjectCentricDynamics

concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
prepend = lambda x, y: jnp.concatenate([x, y], 1)


class AbstractSSM(nj.Module):

  # base fields
  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  free_nats: float = 1.0

  def __init__(self, act_space, obs_space, **kw):
    self.act_space = act_space
    self.obs_space = obs_space
    self.kw = kw

  def _deter_out_dim(self):
    raise NotImplementedError

  @property
  def entry_space(self):
    deter = self._deter_out_dim()
    return dict(
        deter=elements.Space(np.float32, deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)))

  def _initial(self, bsize: int, context_size: int = None):
    deter = self._deter_out_dim()
    shape = [bsize,]
    if context_size is not None:
        shape.append(context_size)

    carry = nn.cast(dict(
        deter=jnp.zeros([*shape, deter], f32),
        stoch=jnp.zeros([*shape, self.stoch, self.classes], f32)))

    zeros = lambda x: jnp.zeros([*shape, *x.shape], x.dtype)
    action = jax.tree.map(zeros, self.act_space)

    return carry, action

  def initial(self, bsize: int):
    return self._initial(bsize)

  def initial_with_context(self, bsize: int):
    return self._initial(bsize, context_size=self.max_context_length)

  def _max_context_causal_mask(self, bsize: int, length: int):
    simple_mask = jnp.tril(jnp.ones((bsize, length, length), dtype=i32))
    rows = jnp.arange(length)[:, None]
    cols = jnp.arange(length)[None, :]

    # mask out dependencies longer than max_context_length
    max_context_mask = (rows - cols < self.max_context_length) & (rows >= cols)

    return simple_mask * max_context_mask.astype(simple_mask.dtype)[None]

  def _causal_mask(self, is_last):
    B, T = is_last.shape[:2]
    rows = jnp.arange(T)[:, None]
    cols = jnp.arange(T)[None, :]
    idx = jnp.arange(T)

    # region[row, i, column] = (row >= i) & (column < i)
    region = (rows[:, None, :] >= idx[None, :, None]) & (cols[None, :, :] < idx[None, :, None])

    # episode_separation_mask is used to enforce this rule:
    # the steps of subsequent episodes do not depend on the steps of the current episode and prediction from the masked
    # last step of the current episode is used for initialization the first step of the next episode, i.e.:
    # episode_end_index = jnp.where(is_last == 1)[0]
    # mask[episode_end_index:, :episode_end_index] = 0
    episode_separation_mask = jnp.any(region[None, :, :, :] & (is_last[:, None, :, None] == 1), axis=2)
    max_context_causal_mask = self._max_context_causal_mask(B, T)

    return jnp.where(episode_separation_mask, 0, max_context_causal_mask)

  @staticmethod
  def _enumerate_steps(is_last):
    B, T = is_last.shape
    absolute_step_idx = jnp.arange(T)[None, :]

    # set is_last[:, 0] = 1 as the algorithm relies on it, this modification does not change the result
    is_last = jnp.concatenate([jnp.ones((B, 1), dtype=is_last.dtype), is_last[:, 1:]], axis=1)

    # episode steps are counted from the last step of the previous episode as the last step data are masked out and used
    # for initialization of the first step
    shift = jax.lax.associative_scan(jnp.maximum, jnp.where(is_last == 1, absolute_step_idx, -1), axis=1)
    return absolute_step_idx - shift

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    return self._truncate(entries, carry=carry)

  def _truncate(self, entries, carry=None):
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def starts(self, entries, carry, actions, nlast):
    raise NotImplementedError

  def observe(self, carry, tokens, action, is_last, reset, training, single=False):
    raise NotImplementedError

  def imagine(self, carry, policy, length, training, single=False):
    raise NotImplementedError

  def loss(self, carry, tokens, acts, is_last, reset, training, text_embeds=None):
    metrics = {}
    carry, entries, feat = self.observe(carry, tokens, acts, is_last, reset, training, text_embeds=text_embeds)
    prior_logit = self._logit('imglogit', feat['deter'], self.imglayers)
    post_logit = feat['logit']
    dyn = self._dist(sg(post_logit)).kl(self._dist(prior_logit))
    rep = self._dist(post_logit).kl(self._dist(sg(prior_logit)))
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)
    losses = {'dyn': dyn, 'rep': rep}
    metrics['dyn_ent'] = self._dist(prior_logit).entropy().mean()
    metrics['rep_ent'] = self._dist(post_logit).entropy().mean()
    return carry, entries, losses, feat, metrics

  def _core(self, deter, stoch, action, is_last, training):
    raise NotImplementedError

  def _logit(self, name, feat, n_layers):
    x = feat
    for i in range(n_layers):
      x = self.sub(f'{name}_logit{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'{name}_logit{i}norm', nn.Norm, self.norm)(x))
    return self._stoch_logit(f'{name}_stochlogit', x)

  def _stoch_logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

  def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out


class RSSM(AbstractSSM):

  # base fields
  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  free_nats: float = 1.0

  # rssm fields
  blocks: int = 8

  def __init__(self, act_space, obs_space, **kw):
    super().__init__(act_space, obs_space, **kw)
    assert 'slot' not in self.obs_space, 'RSSM does not support slots'
    self.max_context_length = 1
    assert self.deter % self.blocks == 0

  def _deter_out_dim(self):
    return self.deter

  def starts(self, entries, carry, actions, nlast):
    B = len(jax.tree.leaves(carry)[0])
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  def observe(self, carry, tokens, action, is_last, reset, training, single=False):
    assert self.max_context_length == 1, f'RSSM: assume that max_context_length == 1: {(self.max_context_length, 1)}'
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      carry = jax.tree.map(lambda x: x[:, 0], carry)
      action = jax.tree.map(lambda x: x[:, 0], action)
      carry, (entry, feat) = self._observe(
          carry, tokens, action, reset, training)
      return jax.tree.map(lambda x: x[:, None], carry), entry, feat
    else:
      unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
      carry, (entries, feat) = nj.scan(
          lambda carry, inputs: self._observe(
              carry, *inputs, training),
          carry, (tokens, action, reset), unroll=unroll, axis=1)
      return carry, entries, feat

  def _observe(self, carry, tokens, action, reset, training):
    deter, stoch, action = nn.mask(
        (carry['deter'], carry['stoch'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)
    deter = self._core(deter, stoch, action, None, training)
    if isinstance(tokens, dict):
      tokens = jnp.concatenate([v for v in tokens.values()], -1)
    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    post_logit = self._logit('obslogit', x, self.obslayers)
    post_stoch = nn.cast(self._dist(post_logit).sample(seed=nj.seed()))
    carry = dict(deter=deter, stoch=post_stoch)
    feat = dict(deter=deter, stoch=post_stoch, logit=post_logit)
    entry = dict(deter=deter, stoch=post_stoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, post_stoch, post_logit))
    return carry, (entry, feat)

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      deter = self._core(carry['deter'], carry['stoch'], actemb, None, training)
      prior_logit = self._logit('imglogit', deter, self.imglayers)
      prior_stoch = nn.cast(self._dist(prior_logit).sample(seed=nj.seed()))
      carry = nn.cast(dict(deter=deter, stoch=prior_stoch))
      feat = nn.cast(dict(deter=deter, stoch=prior_stoch, logit=prior_logit))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, prior_stoch, prior_logit))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      if callable(policy):
        carry, (feat, action) = nj.scan(
            lambda c, _: self.imagine(c, policy, 1, training, single=True),
            nn.cast(carry), (), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nj.scan(
            lambda c, a: self.imagine(c, a, 1, training, single=True),
            nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
      # We can also return all carry entries but it might be expensive.
      # entries = dict(deter=feat['deter'], stoch=feat['stoch'])
      # return carry, entries, feat, action
      return carry, feat, action

  def _core(self, deter, stoch, action, is_last, training):
    stoch = stoch.reshape((stoch.shape[0], -1))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)
    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
    x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
    x2 = nn.act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))
    x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
    for i in range(self.dynlayers):
      x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deter, g, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter


class TSSM(AbstractSSM):

  # base fields
  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  free_nats: float = 1.0

  # tssm fields
  max_context_length: int = 64
  transformer_layers: int = 6
  transformer_heads: int = 8
  transformer_ffup: int = 4
  transformer_act: str = 'silu'
  transformer_norm: str = 'rms'
  transformer_glu: bool = False
  transformer_rope: bool = True
  transformer_qknorm: str = 'none'
  transformer_bias: bool = True
  transformer_outscale: float = 1.0
  transformer_concatenate_over_layers: bool = True
  transformer_normalize_out: bool = False
  transformer_dropout: float = 0.0

  def __init__(self, act_space, obs_space, **kw):
    super().__init__(act_space, obs_space, **kw)
    if type(self) == TSSM:
      assert 'slot' not in self.obs_space, 'TSSM does not support slots'

  def _deter_out_dim(self):
    if self.transformer_concatenate_over_layers:
      return self.deter * self.transformer_layers

    return self.deter

  @staticmethod
  def _zeros_like_expanded(array, n):
    B, _, *shape = array.shape
    return jnp.zeros((B, n, *shape), dtype=array.dtype)

  @staticmethod
  def _sliding_window_view_2d(array, window_size, flatten_batch_axes=True):
    n_windows = array.shape[1] - window_size + 1
    idx = jnp.arange(n_windows)[:, None] + jnp.arange(window_size)
    result = array[:, idx]
    if flatten_batch_axes:
      B, T, *shape = result.shape
      return result.reshape(B * T, *shape)

    return result

  def starts(self, entries, carry, actions, nlast, text_embeds=None):
    assert nlast > 0, (nlast, 0)
    pad_length = self.max_context_length - 1
    pad = jax.tree.map(lambda x: self._zeros_like_expanded(x, pad_length), entries)
    entries = concat([pad, jax.tree.map(lambda x: x[:, -nlast:], entries)], 1)
    state_starts = jax.tree.map(lambda x: self._sliding_window_view_2d(x, self.max_context_length), entries)

    # do not use last step masking during imagination
    is_last = jnp.zeros(actions['action'].shape[:2], dtype=i32)
    pad_is_last = jnp.ones((is_last.shape[0], pad_length), dtype=is_last.dtype)
    is_last = jax.tree.map(lambda x: self._sliding_window_view_2d(x, self.max_context_length), prepend(pad_is_last, is_last))

    pad_action = jax.tree.map(lambda x: self._zeros_like_expanded(x, pad_length + 1), actions)
    actions = concat([pad_action, jax.tree.map(lambda x: x[:, x.shape[1] - nlast + 1:x.shape[1]], actions)], 1)
    action_starts = jax.tree.map(lambda x: self._sliding_window_view_2d(x, self.max_context_length), actions)

    # Handle text embeddings: apply same windowing as states
    text_starts = None
    if text_embeds is not None:
      text_pad = jax.tree.map(lambda x: self._zeros_like_expanded(x, pad_length), text_embeds)
      text_embeds = concat([text_pad, jax.tree.map(lambda x: x[:, -nlast:], text_embeds)], 1)
      text_starts = jax.tree.map(lambda x: self._sliding_window_view_2d(x, self.max_context_length), text_embeds)
    imagination_carry = (state_starts, action_starts['action'], is_last, text_starts)

    return imagination_carry

  def observe(self, carry, tokens, action, is_last, reset, training, single=False, text_embeds=None):
    if isinstance(tokens, dict):
      tokens = jnp.concatenate([v for v in tokens.values()], -1)
    carry, tokens, action, is_last = nn.cast((carry, tokens, action, is_last['is_last']))
    action = nn.DictConcat(self.act_space, len(action['action'].shape) - 1)(action)
    post_logit = self._logit('obslogit', tokens, self.obslayers)
    post_stoch = nn.cast(self._dist(post_logit).sample(seed=nj.seed()))
    mask_last_steps = lambda x: x * (1 - jnp.expand_dims(is_last, range(len(is_last.shape), len(x.shape))))
    action = jax.tree.map(mask_last_steps, action)

    if single:
      assert not training
      # Currently this mode is used only in agent.policy()
      # Assume that carry and action contain the context data for transformer inference
      carry = jax.tree.map(mask_last_steps, carry)
      deter = self._core(None, carry['stoch'], action, is_last, training, text_embeds=text_embeds)
      carry['stoch'] = prepend(carry['stoch'][:, 1:], post_stoch[:, None])
      carry['deter'] = deter
      entries = {'deter': deter[:, -1], 'stoch': post_stoch}
      feat = dict(entries)
      feat['logit'] = post_logit
      return carry, entries, feat

    # This mode is used for training and reporting
    # carry contains 'deter' and 'stoch' from the previous time step
    prev_post_stoch = prepend(carry['stoch'][:, None], post_stoch[:, :-1])
    prev_post_stoch = jax.tree.map(mask_last_steps, prev_post_stoch)
    deter = self._core(None, prev_post_stoch, action, is_last, training, text_embeds=text_embeds)
    carry['stoch'] = post_stoch[:, -1]
    carry['deter'] = deter[:, -1]
    entries = {'deter': deter, 'stoch': post_stoch}
    feat = dict(entries)
    feat['logit'] = post_logit

    return carry, entries, feat

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      state_context, action_context, is_last, text_context = carry
      current_state = jax.tree.map(lambda x: x[:, -1], state_context)
      action = policy(sg(current_state)) if callable(policy) else policy
      action_context = prepend(action_context[:, 1:], action['action'][:, None])
      actemb = nn.DictConcat(self.act_space, 1)({'action': action_context})
      current_text_embed = None
      if text_context is not None:
        current_text_embed = text_context
      deter = self._core(None, state_context['stoch'], actemb, is_last, training, text_embeds=current_text_embed)
      current_prior_logit = self._logit('imglogit', deter[:, -1], self.imglayers)
      current_prior_stoch = nn.cast(self._dist(current_prior_logit).sample(seed=nj.seed()))
      state_context['deter'] = deter
      state_context['stoch'] = prepend(state_context['stoch'][:, 1:], current_prior_stoch[:, None])
      is_last = prepend(is_last[:, 1:], jnp.zeros_like(is_last[:, :1]))
      
      # Update text context
      if text_context is not None:
        text_context = current_text_embed
      
      carry = state_context, action_context, is_last, text_context
      feat = nn.cast(dict(deter=deter[:, -1], stoch=current_prior_stoch, logit=current_prior_logit))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, current_prior_stoch, current_prior_logit))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      if callable(policy):
        carry, (feat, action) = nj.scan(
            lambda c, _: self.imagine(c, policy, 1, training, single=True),
            nn.cast(carry), (), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nj.scan(
            lambda c, a: self.imagine(c, a, 1, training, single=True),
            nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
      # We can also return all carry entries but it might be expensive.
      # entries = dict(deter=feat['deter'], stoch=feat['stoch'])
      # return carry, entries, feat, action
      return carry, feat, action

  def _core(self, deter, stoch, action, is_last, training, text_embeds=None):
    # TODO: check transformer implementation: number of layer, dimensions and so on
    assert stoch.shape[:2] == is_last.shape[:2], (stoch.shape, is_last.shape)
    stoch = stoch.reshape((*stoch.shape[:2], -1))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    x = jnp.concatenate([stoch, action], -1)
    x = self.sub('dynin', nn.Linear, self.deter)(x)
    mask = self._causal_mask(is_last)
    episode_step_idx = self._enumerate_steps(is_last)

    transformer_init_kwargs = {
        'units': self.deter, 'layers': self.transformer_layers, 'heads': self.transformer_heads,
        'ffup': self.transformer_ffup, 'act': self.transformer_act, 'norm': self.transformer_norm,
        'glu': self.transformer_glu, 'rope': self.transformer_rope, 'qknorm': self.transformer_qknorm,
        'bias': self.transformer_bias, 'outscale': self.transformer_outscale,
        'concatenate_over_layers': self.transformer_concatenate_over_layers,
        'normalize_out': self.transformer_normalize_out, 'dropout': self.transformer_dropout,
    }
    deter = self.sub('transformer', Transformer, **transformer_init_kwargs)(x, mask=mask, ts=episode_step_idx,
                                                                            training=training, text_embeds=text_embeds)

    return deter


class ObjectCentricTSSM(TSSM):

  # base fields
  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  free_nats: float = 1.0

  # object-centric tssm fields
  max_context_length: int = 64
  transformer_layers: int = 6
  transformer_heads: int = 8
  transformer_ffup: int = 4
  transformer_act: str = 'silu'
  transformer_norm: str = 'rms'
  transformer_glu: bool = False
  transformer_qknorm: str = 'none'
  transformer_bias: bool = True
  transformer_outscale: float = 1.0
  transformer_concatenate_over_layers: bool = True
  transformer_normalize_out: bool = False
  transformer_dropout: float = 0.0
  transformer_position_embedding: str = 'sinusoidal' # 'sinusoidal', 'none'

 
  slot_key: str = 'slot'

  def __init__(self, act_space, obs_space, **kw):
    super().__init__(act_space, obs_space, **kw)
    assert 'slot' in self.obs_space
    self.num_slots = self.obs_space['slot'].shape[0]
    self.slotkeys = [k for k, s in obs_space.items() if k == self.slot_key]
    

  @property
  def entry_space(self):
    deter = self._deter_out_dim()
    return dict(
        deter=elements.Space(np.float32, (self.num_slots, deter)),
        stoch=elements.Space(np.float32, (self.num_slots, self.stoch, self.classes)))

  def _deter_out_dim(self):
    return self.deter

  def _initial(self, bsize: int, context_size: int = None):
    deter = self._deter_out_dim()
    shape = [bsize,]
    if context_size is not None:
        shape.append(context_size)

    carry = nn.cast(dict(
        deter=jnp.zeros([*shape, self.num_slots, deter], f32),
        stoch=jnp.zeros([*shape, self.num_slots, self.stoch, self.classes], f32)))

    zeros = lambda x: jnp.zeros([*shape, *x.shape], x.dtype)
    action = jax.tree.map(zeros, self.act_space)

    return carry, action

  def loss(self, carry, tokens, acts, is_last, reset, training, text_embeds=None):
    slot_tokens = {k: tokens[k] for k in self.slotkeys if k in tokens}
    carry, entries, losses, feat, metrics = super().loss(carry, slot_tokens, acts, is_last, reset, training, text_embeds=text_embeds)
    losses = {k: v.mean(-1) for k, v in losses.items()}

    return carry, entries, losses, feat, metrics

  def observe(self, carry, tokens, action, is_last, reset, training, single=False, text_embeds=None):
    # Extract only slot tokens
    slot_tokens = {k: tokens[k] for k in tokens if k in self.slotkeys}
    return super().observe(carry, slot_tokens, action, is_last, reset, training, single, text_embeds=text_embeds)

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 4, entries['deter'].shape
    return self._truncate(entries, carry)

  def _core(self, deter, stoch, action, is_last, training, text_embeds=None):
    assert stoch.shape[:2] == is_last.shape[:2], (stoch.shape, is_last.shape)
    stoch = stoch.reshape((*stoch.shape[:-2], -1))
    x = self.sub('dynin', nn.Linear, self.deter)(stoch)
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    action_embedding = self.sub('actin', nn.Linear, self.deter)(action)

    # process an action as a slot
    x = jnp.concatenate([x, jnp.expand_dims(action_embedding, -2)], -2)
    mask = self._causal_mask(is_last)
    episode_step_idx = self._enumerate_steps(is_last)

    init_kw = {
        'units': self.deter, 'layers': self.transformer_layers, 'heads': self.transformer_heads,
        'ffup': self.transformer_ffup, 'act': self.transformer_act, 'norm': self.transformer_norm,
        'glu': self.transformer_glu, 'qknorm': self.transformer_qknorm,
        'bias': self.transformer_bias, 'outscale': self.transformer_outscale,
        'normalize_out': self.transformer_normalize_out, 'dropout': self.transformer_dropout,
        'position_embedding': self.transformer_position_embedding,
    }
    assert text_embeds is not None, "ObjectCentricTSSM requires text embeddings"
    deter = self.sub('object_centric_dynamics', ObjectCentricDynamics, **init_kw)(x, mask=mask, ts=episode_step_idx,
                                                                            training=training, text_embeds=text_embeds)
    # cut off action-slot
    deter = deter[..., :-1, :]

    return deter


class Encoder(nj.Module):

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  outer: bool = False
  strided: bool = False
  slot_key: str = 'slot'

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.vec_keys = kw.pop('vec_keys', '.*')
    self.img_keys = kw.pop('img_keys', '.*')

    self.slotkeys = [k for k, s in obs_space.items() if k == self.slot_key]
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2 and re.match(self.vec_keys, k) and k != self.slot_key]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3 and re.match(self.img_keys, k) and k != self.slot_key]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

    if len(self.slotkeys) > 0:
      assert len(self.slotkeys) == 1, f'{self.slotkeys}'
      assert len(self.imgkeys) == 0, f'{self.imgkeys}: slot observation cannot be mixed with images'

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, obs, reset, training, single=False, only_text=False):
    bdims = 1 if single else 2
    outs = {}
    bshape = reset.shape

    if self.slotkeys and not only_text:
      x = obs[self.slot_key]
      x = x.reshape((-1, *x.shape[len(bshape):]))
      outs[self.slot_key] = x

    if self.veckeys:
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      vecs = {k: obs[k] for k in self.veckeys}
      squish = nn.symlog if self.symlog else lambda x: x
      x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
      x = x.reshape((-1, *x.shape[bdims:]))
      #x = nn.COMPUTE_DTYPE(x) # ensure compute dtype
      # for i in range(self.layers):
      #   x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
      #   x = nn.act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
      assert len(self.veckeys)==1, "Expected only token or token_embed as vector input"
      for k in self.veckeys:  
        outs[k] = x

    if self.imgkeys and not only_text:
      K = self.kernel
      imgs = [obs[k] for k in sorted(self.imgkeys)]
      assert all(x.dtype == jnp.uint8 for x in imgs)
      x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
      x = x.reshape((-1, *x.shape[bdims:]))
      for i, depth in enumerate(self.depths):
        if self.outer and i == 0:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
        elif self.strided:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
        else:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
          B, H, W, C = x.shape
          x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
        x = nn.act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
      assert 3 <= x.shape[-3] <= 16, x.shape
      assert 3 <= x.shape[-2] <= 16, x.shape
      x = x.reshape((x.shape[0], -1))
      assert len(self.imgkeys)==1, "Expected only one image input"
      for k in self.imgkeys:  
        outs[k] = x

    # x = jnp.concatenate(outs, -1)
    tokens = {}
    for k, v in outs.items():
        tokens[k] = v.reshape((*bshape, *v.shape[1:]))
    entries = {}
    return carry, entries, tokens


class Decoder(nj.Module):

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  outscale: float = 1.0
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  bspace: int = 8
  outer: bool = False
  strided: bool = False
  # vec_keys: str = '.*'
  # img_keys: str = '.*'
  slot_key: str = 'slot'

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.vec_keys =  kw.pop('vec_keys', '.*')
    self.img_keys =  kw.pop('img_keys', '.*')
    self.slotkeys = [k for k, s in obs_space.items() if k == self.slot_key]
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2 and re.match(self.vec_keys, k) and k != self.slot_key]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3 and re.match(self.img_keys, k) and k != self.slot_key]   
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
    if self.imgkeys:
      self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
    self.kw = kw
    self.vec_dist = kw.pop('vec_dist', None)
  
    if len(self.slotkeys) > 0:
      assert len(self.slotkeys) == 1, f'{self.slotkeys}'
      assert len(self.imgkeys) == 0, f'{self.imgkeys}: slot observation cannot be mixed with images'

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, feat, reset, training, single=False):
    K = self.kernel
    recons = {}
    bshape = reset.shape
    if self.slotkeys:
      num_slots = feat['deter'].shape[-2]
      bshape = (*bshape, num_slots)
    else:
      assert feat['deter'].shape[-1] % self.bspace == 0

    inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]
    inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
    inp = jnp.concatenate(inp, -1)

    if self.slotkeys:
      spaces = {k: self.obs_space[k].shape[-1:] for k in self.slotkeys}
      outputs = {k: 'symlog_mse' if self.symlog else 'mse' for k, v in spaces.items()}
      kw = dict(**self.kw, act=self.act, norm=self.norm)
      x = self.sub('mlp_slots', nn.MLP, self.layers, self.units, **kw)(inp)
      x = x.reshape((*bshape, *x.shape[1:]))
      kw = dict(**self.kw, outscale=self.outscale)
      outs = self.sub('slot', embodied.jax.DictHead, spaces, outputs, **kw)(x)
      outs = {k: embodied.jax.outs.Agg(v, 1, jnp.sum) for k, v in outs.items()}
      recons.update(outs)
    bshape = bshape[:-1] if self.slotkeys else bshape
    inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]
    inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
    inp = jnp.concatenate(inp, -1)
    if self.veckeys:
      spaces = {k: self.obs_space[k] for k in self.veckeys}
      outputs = {k: self.vec_dist for k in spaces.keys()}
      kw = dict(**self.kw, act=self.act, norm=self.norm)
      x = self.sub('mlp_vec', nn.MLP, self.layers, self.units, **kw)(inp)
      x = x.reshape((*bshape, *x.shape[1:]))
      kw = dict(**self.kw, outscale=self.outscale)
      outs = self.sub('vec', embodied.jax.DictHead, spaces, outputs, **kw)(x)
      recons.update(outs)

    if self.imgkeys:
      factor = 2 ** (len(self.depths) - int(bool(self.outer)))
      minres = [int(x // factor) for x in self.imgres]
      assert 3 <= minres[0] <= 16, minres
      assert 3 <= minres[1] <= 16, minres
      shape = (*minres, self.depths[-1])
      if self.bspace:
        u, g = math.prod(shape), self.bspace
        x0, x1 = nn.cast((feat['deter'], feat['stoch']))
        x1 = x1.reshape((*x1.shape[:-2], -1))
        x0 = x0.reshape((-1, x0.shape[-1]))
        x1 = x1.reshape((-1, x1.shape[-1]))
        x0 = self.sub('sp0', nn.BlockLinear, u, g, **self.kw)(x0)
        x0 = einops.rearrange(
            x0, '... (g h w c) -> ... h w (g c)',
            h=minres[0], w=minres[1], g=g)
        x1 = self.sub('sp1', nn.Linear, 2 * self.units, **self.kw)(x1)
        x1 = nn.act(self.act)(self.sub('sp1norm', nn.Norm, self.norm)(x1))
        x1 = self.sub('sp2', nn.Linear, shape, **self.kw)(x1)
        x = nn.act(self.act)(self.sub('spnorm', nn.Norm, self.norm)(x0 + x1))
      else:
        x = self.sub('space', nn.Linear, shape, **kw)(inp)
        x = nn.act(self.act)(self.sub('spacenorm', nn.Norm, self.norm)(x))
      for i, depth in reversed(list(enumerate(self.depths[:-1]))):
        if self.strided:
          kw = dict(**self.kw, transp=True)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, 2, **kw)(x)
        else:
          x = x.repeat(2, -2).repeat(2, -3)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'conv{i}norm', nn.Norm, self.norm)(x))
      if self.outer:
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      elif self.strided:
        kw = dict(**self.kw, outscale=self.outscale, transp=True)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, 2, **kw)(x)
      else:
        x = x.repeat(2, -2).repeat(2, -3)
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      x = jax.nn.sigmoid(x)
      x = x.reshape((*bshape, *x.shape[1:]))
      split = np.cumsum(
          [self.obs_space[k].shape[-1] for k in self.imgkeys][:-1])
      for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
        out = embodied.jax.outs.MSE(out)
        out = embodied.jax.outs.Agg(out, 3, jnp.sum)
        recons[k] = out

    entries = {}
    return carry, entries, recons
