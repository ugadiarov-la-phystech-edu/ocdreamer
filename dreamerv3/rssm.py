import math

import einops
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
sg = jax.lax.stop_gradient

from embodied.jax.nets import Transformer as TransformerOrig


class RSSM(nj.Module):

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
  blocks: int = 8
  free_nats: float = 1.0
  use_transformer: bool = True
  max_context_length: int = 63

  def __init__(self, act_space, **kw):
    self.act_space = act_space
    self.kw = kw
    if self.use_transformer:
      self._core = self._core_transformer
      self.observe = self.observe_transdreamer
      self.n_transformer_layers = 6
    else:
      assert self.deter % self.blocks == 0

  @property
  def entry_space(self):
    deter = self.deter * self.n_transformer_layers
    return dict(
        deter=elements.Space(np.float32, deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)))

  def initial(self, bsize):
    deter = self.deter * self.n_transformer_layers
    carry = nn.cast(dict(
        deter=jnp.zeros([bsize, deter], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32)))
    return carry

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def starts(self, entries, carry, nlast):
    B = len(jax.tree.leaves(carry)[0])
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  def observe_transdreamer(self, carry, tokens, action, reset, training, single=False):
    # TODO: handle reset
    carry, tokens, action = nn.cast((carry, tokens, action))
    # Check if the input is a single observation or history
    # single observation shape: (n_envs, token_dim)
    # history shape: (n_envs, context_length, token_dim)
    assert len(tokens.shape) in (2, 3), f'shape={tokens.shape}'
    single = len(tokens.shape) == 2
    if single:
      tokens = tokens[:, None]

    action = nn.DictConcat(self.act_space, len(action['action'].shape) - 1)(action)
    post = self._prior('posterior_transdreamer', tokens)
    post_stoch = nn.cast(self._dist(post).sample(seed=nj.seed()))
    prev_states = jnp.concatenate([carry['stoch'][:, None], post_stoch[:, :-1]], axis=1)
    deter = self._core(None, prev_states, action[:, -prev_states.shape[1]:])
    # prior = self._prior('prior_transdreamer', deter)
    # dyn = self._dist(sg(post)).kl(self._dist(prior))
    # rep = self._dist(post).kl(self._dist(sg(prior)))
    entries = {'deter': deter, 'stoch': post_stoch}
    feat = dict(entries)
    feat['logit'] = post
    carry = {'deter': entries['deter'][:, -1], 'stoch': entries['stoch'][:, -1]}
    # if single:
    #   entries = {k: v[:, 0] for k, v in entries.items()}
    #   feat = {k: v[:, 0] for k, v in feat.items()}

    return carry, entries, feat

  def observe(self, carry, tokens, action, reset, training, single=False):
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      carry, (entry, feat) = self._observe(
          carry, tokens, action, reset, training)
      return carry, entry, feat
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
    deter = self._core(deter, stoch, action)
    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
    carry = dict(deter=deter, stoch=stoch)
    feat = dict(deter=deter, stoch=stoch, logit=logit)
    entry = dict(deter=deter, stoch=stoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def imagine_transformer(self, carry, policy, length, training, single=False):
    if single:
      state_context, action_context = carry
      current_state = {'deter': state_context['deter'][:, -1], 'stoch': state_context['stoch'][:, -1]}
      action = policy(sg(current_state)) if callable(policy) else policy
      action_context = jnp.concatenate([action_context, action['action'][:, None]], axis=1)[:, -state_context['stoch'].shape[1]:]
      actemb = nn.DictConcat(self.act_space, 1)({'action': action_context})
      deter = self._core(state_context['deter'], state_context['stoch'], actemb)[:, -1:]
      logit = self._prior('prior', deter)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      state_context = {
          'deter': jnp.concatenate([state_context['deter'], deter], axis=1)[:, -self.max_context_length:],
          'stoch': jnp.concatenate([state_context['stoch'], stoch], axis=1)[:, -self.max_context_length:],
      }
      carry = state_context, action_context
      feat = nn.cast(dict(deter=deter[:, 0], stoch=stoch[:, 0], logit=logit[:, 0]))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      if callable(policy):
        carry, (feat, action) = nj.scan(
            lambda c, _: self.imagine_transformer(c, policy, 1, training, single=True),
            nn.cast(carry), (), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nj.scan(
            lambda c, a: self.imagine_transformer(c, a, 1, training, single=True),
            nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
      # We can also return all carry entries but it might be expensive.
      # entries = dict(deter=feat['deter'], stoch=feat['stoch'])
      # return carry, entries, feat, action
      return carry, feat, action

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      deter = self._core(carry['deter'], carry['stoch'], actemb)
      logit = self._prior('prior', deter)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      carry = nn.cast(dict(deter=deter, stoch=stoch))
      feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
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

  def loss(self, carry, tokens, acts, reset, training):
    metrics = {}
    carry, entries, feat = self.observe(carry, tokens, acts, reset, training)
    prior = self._prior('prior', feat['deter'])
    post = feat['logit']
    dyn = self._dist(sg(post)).kl(self._dist(prior))
    rep = self._dist(post).kl(self._dist(sg(prior)))
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)
    losses = {'dyn': dyn, 'rep': rep}
    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()
    return carry, entries, losses, feat, metrics

  def _core(self, deter, stoch, action):
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

  def _core_transformer(self, deter, stoch, action):
    # TODO: check transformer implementation
    # TODO: should operate on context
    # single = len(stoch.shape) == 3
    # if single:
    #   stoch = stoch[:, None]
    #   action = action[:, None]

    stoch = stoch.reshape((stoch.shape[0], stoch.shape[1], -1))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    x = jnp.concatenate([stoch, action], -1)
    x = self.sub('img_in', nn.Linear, self.deter)(x)
    # deter = self.sub('transformer', Transformer, d_model_inner=64,
    #     num_heads=8, feed_forward_dim=self.deter, num_layers=6,
    #         kw={'act':'none'})(x, deter)
    # x = jnp.concatenate([deter, x], axis=-1)
    deter = self.sub('transformer', TransformerOrig, layers=6, units=self.deter)(x)
    # if single:
    #   deter = deter[:, 0]

    # if deter.shape[-2] == 1:
    #   print()

    return deter

  def _prior(self, name, feat):
    x = feat
    for i in range(self.imglayers):
      x = self.sub(f'{name}_prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'{name}_prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit(f'{name}_priorlogit', x)

  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

  def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out


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

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, obs, reset, training, single=False):
    bdims = 1 if single else 2
    outs = []
    bshape = reset.shape

    if self.veckeys:
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      vecs = {k: obs[k] for k in self.veckeys}
      squish = nn.symlog if self.symlog else lambda x: x
      x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
      x = x.reshape((-1, *x.shape[bdims:]))
      for i in range(self.layers):
        x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
      outs.append(x)

    if self.imgkeys:
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
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    tokens = x.reshape((*bshape, *x.shape[1:]))
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

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
    self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, feat, reset, training, single=False):
    assert feat['deter'].shape[-1] % self.bspace == 0
    K = self.kernel
    recons = {}
    bshape = reset.shape
    inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]
    inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
    inp = jnp.concatenate(inp, -1)

    if self.veckeys:
      spaces = {k: self.obs_space[k] for k in self.veckeys}
      o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')
      outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}
      kw = dict(**self.kw, act=self.act, norm=self.norm)
      x = self.sub('mlp', nn.MLP, self.layers, self.units, **kw)(inp)
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


class MultiHeadSelfAttention(nj.Module):
  def __init__(self, d_model_inner, num_heads, kw):
    self.d_model_inner = d_model_inner
    self.num_heads = num_heads
    self._kw = kw
    self.head_dim = self.d_model_inner // self.num_heads
    self.hidden = self.d_model_inner * self.num_heads

  def __call__(self, x):
    batch_size, embed_dim = x.shape

    # Create queries, keys, values
    query = self.sub('linear1_M', nn.Linear, self.hidden)(x)
    query = query.reshape(batch_size, -1, self.num_heads, self.head_dim)
    key = self.sub('linear2_M', nn.Linear, self.hidden)(x)
    key = key.reshape(batch_size, -1, self.num_heads, self.head_dim)
    value = self.sub('linear3_M', nn.Linear, self.hidden)(x)
    value = value.reshape(batch_size, -1, self.num_heads, self.head_dim)

    # Calculate attention scores
    scores = jnp.einsum('bqhd,bkhd->bhqk', query, key) / jnp.sqrt(self.head_dim)
    weights = jax.nn.softmax(scores, axis=-1)

    # Apply attention to value
    attention_output = jnp.einsum('bhqk,bkhd->bqhd', weights, value).reshape(batch_size, -1, self.hidden)

    # Final dense layer
    output = self.sub('linear4_M', nn.Linear, embed_dim)(attention_output)
    output = output.reshape(1, batch_size, -1).squeeze(0)
    return output


class TransformerLayer(nj.Module):

  def __init__(self, d_model_inner, num_heads, feed_forward_dim, kw):
    self.d_model_inner = d_model_inner
    self.num_heads = num_heads
    self.feed_forward_dim = feed_forward_dim
    self._kw = kw

  def __call__(self, x):
    batch_size, embed_dim = x.shape
    # Multi-head self-attention
    attn_output = self.sub('MultiHeadSelfAttention', MultiHeadSelfAttention,
            self.d_model_inner, self.num_heads, self._kw)(x)

    attn_output = self.sub('layernorm1_TL', nn.Norm, 'layer')(x + attn_output)

    # Feed-forward
    ff_output = self.sub('linear1_TL', nn.Linear, self.feed_forward_dim)(attn_output)
    ff_output = jax.nn.relu(ff_output)
    ff_output = self.sub('linear2_TL', nn.Linear, embed_dim)(ff_output)
    ff_output = self.sub('layernorm2_TL', nn.Norm, 'layer')(attn_output + ff_output)
    ff_output = ff_output.reshape(1, batch_size, -1).squeeze(0)
    return ff_output

# class PositionalEmbedding(nj.Module):
#   def __init__(self, d_model=600):
#     self.d_model = d_model

#   def __call__(self, positions):
#       # Initialize the frequencies
#       inv_freq = 1 / (10000 ** (jnp.arange(0.0, self.dim, 2.0) / self.dim))

#       # Calculate the positional embeddings
#       sinusoid_inp = jnp.einsum("i,j->ij", positions.astype(jnp.float32), inv_freq)
#       pos_emb = jnp.concatenate([jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)], axis=-1)
#       return pos_emb[:, None, :]

class Transformer(nj.Module):
  def __init__(self, d_model_inner=64, num_heads=8, feed_forward_dim=512, num_layers=6, kw=None):
    self.d_model_inner = d_model_inner
    self.num_heads = num_heads
    self.feed_forward_dim = feed_forward_dim
    self.num_layers = num_layers
    self._kw = kw

  def __call__(self, x, deter):
    x = jnp.concatenate([deter, x], axis=-1)

    # Transformer layers
    for _ in range(self.num_layers):
      x = self.sub('TransformerLayer', TransformerLayer, self.d_model_inner,
            self.num_heads, self.feed_forward_dim, self._kw)(x)

    # Add an additional Dense layer to project down to the desired size
    x = self.sub('linear2_T', nn.Linear, deter.shape[1])(x)

    return x
