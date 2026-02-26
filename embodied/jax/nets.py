import functools
import math
from typing import Callable

import einops
import jax
import jax.ad_checkpoint as adc
import jax.numpy as jnp
import numpy as np
from dreamerv3 import ninjax_old as nj

COMPUTE_DTYPE = jnp.bfloat16
LAYER_CALLBACK = lambda tensor, name: tensor

f32 = jnp.float32


def cast(xs, force=False):
  if force:
    should = lambda x: True
  else:
    should = lambda x: jnp.issubdtype(x.dtype, jnp.floating)
  return jax.tree.map(lambda x: COMPUTE_DTYPE(x) if should(x) else x, xs)


def act(name):
  if name == 'none':
    return lambda x: x
  elif name == 'mish':
    return lambda x: x * jnp.tanh(jax.nn.softplus(x))
  elif name == 'relu2':
    return lambda x: jnp.square(jax.nn.relu(x))
  elif name == 'swiglu':
    def fn(x):
      x, y = jnp.split(x, 2, -1)
      return jax.nn.silu(x) * y
    return fn
  else:
    return getattr(jax.nn, name)


def init(name):
  if callable(name):
    return name
  elif name.endswith(('_in', '_out', '_avg')):
    dist, fan = name.rsplit('_', 1)
  else:
    dist, fan = name, 'in'
  return Initializer(dist, fan, 1.0)


def dropout(x, prob, training):
  if not prob or not training:
    return x
  keep = jax.random.bernoulli(nj.rng(), 1.0 - prob, x.shape)
  return x * keep / (1.0 - prob)


def symlog(x):
  return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
  return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def where(condition, xs, ys):
  assert condition.dtype == bool, condition.dtype
  def fn(x, y):
    assert x.shape == y.shape, (x.shape, y.shape)
    expanded = jnp.expand_dims(condition, list(range(condition.ndim, x.ndim)))
    return jnp.where(expanded, x, y)
  return jax.tree.map(fn, xs, ys)


def mask(xs, mask):
  return where(mask, xs, jax.tree.map(jnp.zeros_like, xs))


def available(*trees, bdims=None):
  def fn(*xs):
    masks = []
    for x in xs:
      if jnp.issubdtype(x.dtype, jnp.floating):
        mask = (x != -jnp.inf)
      elif jnp.issubdtype(x.dtype, jnp.signedinteger):
        mask = (x != -1)
      elif (
          jnp.issubdtype(x.dtype, jnp.unsignedinteger) or
          jnp.issubdtype(x.dtype, bool)):
        shape = x.shape if bdims is None else x.shape[:bdims]
        mask = jnp.full(shape, True, bool)
      else:
        raise NotImplementedError(x.dtype)
      if bdims is not None:
        mask = mask.all(tuple(range(bdims, mask.ndim)))
      masks.append(mask)
    return jnp.stack(masks, 0).all(0)
  return jax.tree.map(fn, *trees)


@functools.partial(jax.custom_vjp, nondiff_argnums=[1, 2])
def ensure_dtypes(x, fwd=None, bwd=None):
  fwd = fwd or COMPUTE_DTYPE
  bwd = bwd or COMPUTE_DTYPE
  assert x.dtype == fwd, (x.dtype, fwd)
  return x
def ensure_dtypes_fwd(x, fwd=None, bwd=None):
  fwd = fwd or COMPUTE_DTYPE
  bwd = bwd or COMPUTE_DTYPE
  return ensure_dtypes(x, fwd, bwd), ()
def ensure_dtypes_bwd(fwd, bwd, cache, dx):
  fwd = fwd or COMPUTE_DTYPE
  bwd = bwd or COMPUTE_DTYPE
  assert dx.dtype == bwd, (dx.dtype, bwd)
  return (dx,)
ensure_dtypes.defvjp(ensure_dtypes_fwd, ensure_dtypes_bwd)


def rms(xs):
  xs = jax.tree.leaves(xs)
  count = sum(x.size for x in xs)
  sumsq = jnp.stack([f32(jnp.square(x).sum()) for x in xs]).sum()
  return jnp.sqrt(sumsq / f32(count))


def rope(x, ts=None, inverse=False, maxlen=4096):
  B, T, _, D = x.shape
  if ts is None:
    ts = jnp.ones(B, jnp.int32)[:, None] * jnp.arange(T)[None, :]  # [B, T]
  assert ts.shape == (B, T), (ts.shape, (B, T))
  if inverse:
    ts = -ts
  freq_exponents = (2.0 / D) * jnp.arange(D // 2)  # [D/2]
  timescale = maxlen ** freq_exponents
  radians = ts[:, :, None] / timescale[None, None, :]  # [B, T, D/2]
  radians = radians[..., None, :].astype(x.dtype)  # [B, T, 1, D/2]
  sin, cos = jnp.sin(radians), jnp.cos(radians)
  x1, x2 = jnp.split(x, 2, axis=-1)  # [B, T, H, D/2]
  res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
  return res


class Initializer:

  def __init__(self, dist='trunc_normal', fan='in', scale=1.0):
    self.dist = dist
    self.fan = fan
    self.scale = scale

  def __call__(self, shape, dtype=jnp.float32, fshape=None):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    assert all(isinstance(x, int) for x in shape), (
        shape, [type(x) for x in shape])
    assert all(x > 0 for x in shape), shape
    fanin, fanout = self.compute_fans(shape if fshape is None else fshape)
    fan = {
        'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout, 'none': 1,
    }[self.fan]
    if self.dist == 'zeros':
      x = jnp.zeros(shape, dtype)
    elif self.dist == 'uniform':
      limit = np.sqrt(1 / fan)
      x = jax.random.uniform(nj.rng(), shape, dtype, -limit, limit)
    elif self.dist == 'normal':
      x = jax.random.normal(nj.rng(), shape)
      x *= np.sqrt(1 / fan)
    elif self.dist == 'trunc_normal':
      x = jax.random.truncated_normal(nj.rng(), -2, 2, shape)
      x *= 1.1368 * np.sqrt(1 / fan)
    elif self.dist == 'normed':
      x = jax.random.uniform(nj.rng(), shape, dtype, -1, 1)
      x *= (1 / jnp.linalg.norm(x.reshape((-1, shape[-1])), 2, 0))
    else:
      raise NotImplementedError(self.dist)
    x *= self.scale
    x = x.astype(dtype)
    return x

  def __repr__(self):
    return f'Initializer({self.dist}, {self.fan}, {self.scale})'

  def __eq__(self, other):
    if not isinstance(other, Initializer):
      return NotImplemented
    attributes = ('dist', 'fan', 'scale')
    return all(getattr(self, k) == getattr(other, k) for k in attributes)

  @staticmethod
  def compute_fans(shape):
    if len(shape) == 0:
      return (1, 1)
    elif len(shape) == 1:
      return (1, shape[0])
    elif len(shape) == 2:
      return shape
    else:
      space = math.prod(shape[:-2])
      return (shape[-2] * space, shape[-1] * space)


class Embed(nj.Module):

  einit: str | Callable = Initializer('trunc_normal', 'out')
  combine: bool = False

  def __init__(self, classes, units, shape=()):
    self.classes = classes
    self.units = units
    self.shape = shape

  def __call__(self, x):
    batch_shape = x.shape[:x.ndim - len(self.shape)]
    event_shape = x.shape[x.ndim - len(self.shape):]
    assert event_shape == self.shape, (self.shape, x.shape)
    N = math.prod(self.shape)
    K = self.classes
    D = self.units
    shape = (*self.shape, self.classes, self.units)
    table = self.get('table', init(self.einit), shape)
    table = table.reshape(N, K, D)
    table = table.astype(COMPUTE_DTYPE)
    index = x.reshape(-1, N)
    embed = table[jnp.arange(N), index]
    if self.combine:
      embed = embed.sum(-2).reshape(*batch_shape, self.units)
    else:
      embed = embed.reshape(*batch_shape, *self.shape, self.units)
    return embed


class Linear(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, units):
    self.units = (units,) if isinstance(units, int) else tuple(units)

  def __call__(self, x):
    ensure_dtypes(x)
    size = math.prod(self.units)
    shape = (x.shape[-1], size)
    x = x @ self.get('kernel', self._scaled_winit, shape).astype(x.dtype)
    if self.bias:
      x += self.get('bias', init(self.binit), size).astype(x.dtype)
    x = x.reshape((*x.shape[:-1], *self.units))
    return x

  def _scaled_winit(self, *args, **kwargs):
    return init(self.winit)(*args, **kwargs) * self.outscale


class BlockLinear(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, units, blocks):
    assert isinstance(units, int), (units, type(units))
    assert blocks <= units and units % blocks == 0, (blocks, units)
    self.units = units
    self.blocks = blocks

  def __call__(self, x):
    ensure_dtypes(x)
    assert x.shape[-1] % self.blocks == 0, (x.shape, self.blocks)
    insize = x.shape[-1]
    shape = (self.blocks, insize // self.blocks, self.units // self.blocks)
    kernel = self.get('kernel', self._scaled_winit, shape).astype(x.dtype)
    x = x.reshape((*x.shape[:-1], self.blocks, insize // self.blocks))
    x = jnp.einsum('...ki,kio->...ko', x, kernel)
    x = x.reshape((*x.shape[:-2], self.units))
    if self.bias:
      x += self.get('bias', init(self.binit), self.units).astype(x.dtype)
    return x

  def _scaled_winit(self, *args, **kwargs):
    return init(self.winit)(*args, **kwargs) * self.outscale


class Conv2D(nj.Module):

  transp: bool = False
  groups: int = 1
  pad: str = 'same'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, depth, kernel, stride=1):
    self.depth = depth
    self.kernel = (kernel,) * 2 if isinstance(kernel, int) else kernel
    self.stride = stride

  def __call__(self, x):
    ensure_dtypes(x)
    shape = (*self.kernel, x.shape[-1] // self.groups, self.depth)
    kernel = self.get('kernel', self._scaled_winit, shape).astype(x.dtype)
    if self.transp:
      assert self.pad == 'same', self.pad
      # Manual implementation of fractionally strided convolution because the
      # cuDNN implementation used by XLA has bugs and performance issues.
      x = x.repeat(self.stride, -2).repeat(self.stride, -3)
      maskh = ((jnp.arange(x.shape[-3]) - 1) % self.stride == 0)[:, None]
      maskw = ((jnp.arange(x.shape[-2]) - 1) % self.stride == 0)[None, :]
      x *= (maskh * maskw)[:, :, None]
      stride = (1, 1)
    else:
      stride = (self.stride, self.stride)
    x = jax.lax.conv_general_dilated(
        x, kernel, stride, self.pad.upper(),
        feature_group_count=self.groups,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    if self.bias:
      x += self.get('bias', init(self.binit), self.depth).astype(x.dtype)
    return x

  def _scaled_winit(self, *args, **kwargs):
    return init(self.winit)(*args, **kwargs) * self.outscale


class Conv3D(nj.Module):

  transp: bool = False
  groups: int = 1
  pad: str = 'same'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')

  def __init__(self, depth, kernel, stride=1):
    self.depth = depth
    self.kernel = (kernel,) * 3 if isinstance(kernel, int) else kernel
    self.stride = (stride,) * 3 if isinstance(stride, int) else stride

  def __call__(self, x):
    ensure_dtypes(x)
    if self.transp:
      assert self.groups == 1, self.groups
      shape = (*self.kernel, x.shape[-1], self.depth)
      kernel = self.get('kernel', init(self.winit), shape).astype(x.dtype)
      x = jax.lax.conv_transpose(
          x, kernel, self.stride, self.pad.upper(),
          dimension_numbers=('NTHWC', 'THWIO', 'NTHWC'))
    else:
      shape = (*self.kernel, x.shape[-1] // self.groups, self.depth)
      kernel = self.get('kernel', init(self.winit), shape).astype(x.dtype)
      x = jax.lax.conv_general_dilated(
          x, kernel, self.stride, self.pad.upper(),
          feature_group_count=self.groups,
          dimension_numbers=('NTHWC', 'THWIO', 'NTHWC'))
    if self.bias:
      x += self.get('bias', init(self.binit), self.depth).astype(x.dtype)
    return x


class Norm(nj.Module):

  axis: tuple = (-1,)
  eps: float = 1e-4
  scale: bool = True
  shift: bool = True

  def __init__(self, impl):
    if '1em' in impl:
      impl, exp = impl.split('1em')
      self._fields['eps'] = 10 ** -int(exp)
    self.impl = impl

  def __call__(self, x):
    ensure_dtypes(x)
    dtype = x.dtype
    x = f32(x)
    axis = [a % x.ndim for a in self.axis]
    shape = [x.shape[i] if i in axis else 1 for i in range(min(axis), x.ndim)]
    if self.impl == 'none':
      pass
    elif self.impl == 'rms':
      mean2 = jnp.square(x).mean(axis, keepdims=True)
      mean2 = adc.checkpoint_name(mean2, 'small')
      scale = self._scale(shape, x.dtype)
      x = x * (jax.lax.rsqrt(mean2 + self.eps) * scale)
    elif self.impl == 'layer':
      mean = x.mean(axis, keepdims=True)
      mean2 = jnp.square(x).mean(axis, keepdims=True)
      mean2 = adc.checkpoint_name(mean2, 'small')
      var = jnp.maximum(0, mean2 - jnp.square(mean))
      var = adc.checkpoint_name(var, 'small')
      scale = self._scale(shape, x.dtype)
      shift = self._shift(shape, x.dtype)
      x = (x - mean) * (jax.lax.rsqrt(var + self.eps) * scale) + shift
    else:
      raise NotImplementedError(self.impl)
    x = x.astype(dtype)
    return x

  def _scale(self, shape, dtype):
    if not self.scale:
      return jnp.ones(shape, dtype)
    return self.get('scale', jnp.ones, shape, f32).astype(dtype)

  def _shift(self, shape, dtype):
    if not self.shift:
      return jnp.zeros(shape, dtype)
    return self.get('shift', jnp.zeros, shape, f32).astype(dtype)


class Attention(nj.Module):

  heads: int = 8
  kv_heads: int = 0
  dropout: float = 0.0
  rope: bool = True
  qknorm: str = 'none'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __call__(self, x, mask=None, ts=None, training=True):
    kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    B, T, D = x.shape
    kv_heads = self.kv_heads or self.heads
    assert self.heads % kv_heads == 0
    head_ratio = self.heads // kv_heads
    if head_ratio == 1:
      qkv = self.get('qkv', Linear, 3 * D, **kw)(x)
      q, k, v = jnp.split(qkv, 3, -1)
    else:
      q = self.get('q', Linear, D, **kw)(x)
      k = self.get('k', Linear, D // head_ratio, **kw)(x)
      v = self.get('v', Linear, D // head_ratio, **kw)(x)
    q = einops.rearrange(q, 'b t (h d) -> b t h d', h=self.heads)
    k = einops.rearrange(k, 'b t (h d) -> b t h d', h=kv_heads)
    v = einops.rearrange(v, 'b t (h d) -> b t h d', h=kv_heads)

    if self.qknorm != 'none':
      q = self.get('normq', Norm, self.qknorm)(q)
      k = self.get('normk', Norm, self.qknorm)(k)

    if self.rope:
      q = rope(q, ts)
      k = rope(k, ts)

    q = einops.rearrange(q, 'b t (h g) d -> b t h g d', h=kv_heads)
    logits = einops.einsum(q, k, 'b tq h g d, b tk h d -> b h g tq tk')
    logits = logits * (1.0 / np.sqrt(k.shape[-1]))
    logits = f32(logits)
    if mask is not None:
      Tq, Tk = q.shape[1], k.shape[1]
      assert mask.shape == (B, Tq, Tk), (mask.shape, (B, Tq, Tk))
      mask = einops.rearrange(mask, 'b tq tk -> b 1 1 tq tk')
      logits = jnp.where(mask, logits, -1e30)
    weights = jax.nn.softmax(logits)
    weights = weights.astype(x.dtype)
    weights = dropout(weights, self.dropout, training)
    x = einops.einsum(weights, v, 'b h g tq tk, b tk h d -> b tq h g d')
    x = einops.rearrange(x, 'b t h g d -> b t (h g d)')
    x = self.get('proj', Linear, D, **kw, outscale=self.outscale)(x)
    return x


class CrossAttention(Attention):

  heads: int = 8
  kv_heads: int = 0
  dropout: float = 0.0
  rope: bool = True
  qknorm: str = 'none'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def __call__(self, slots, feats, mask=None, ts=None, training=True):
    kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    B, T_q, D_q = slots.shape
    B_f, T_k, D_f = feats.shape
    assert T_q == T_k, f"Sequence length mismatch: {T_q} vs {T_k}"
    assert B == B_f, f"Batch size mismatch: {B} vs {B_f}"
    kv_heads = self.kv_heads or self.heads
    assert self.heads % kv_heads == 0
    head_ratio = self.heads // kv_heads
    q = self.get('to_q', Linear, D_q, **kw)(slots)
    k = self.get('to_k', Linear, D_q // head_ratio, **kw)(feats)
    v = self.get('to_v', Linear, D_q // head_ratio, **kw)(feats)
    
    q = einops.rearrange(q, 'b t (h d) -> b t h d', h=self.heads)
    k = einops.rearrange(k, 'b t (h d) -> b t h d', h=kv_heads)
    v = einops.rearrange(v, 'b t (h d) -> b t h d', h=kv_heads)

    if self.qknorm != 'none':
      q = self.get('normq', Norm, self.qknorm)(q)
      k = self.get('normk', Norm, self.qknorm)(k)

    if self.rope:
      q = rope(q, ts)

    q = einops.rearrange(q, 'b t (h g) d -> b t h g d', h=kv_heads)
    logits = einops.einsum(q, k, 'b tq h g d, b tk h d -> b h g tq tk')
    logits = logits * (1.0 / np.sqrt(k.shape[-1]))
    logits = f32(logits)
    if mask is not None:
      assert mask.shape == (B, T_q, T_k), (mask.shape, (B, T_q, T_k))
      mask = einops.rearrange(mask, 'b tq tk -> b 1 1 tq tk')
      logits = jnp.where(mask, logits, -1e30)
    weights = jax.nn.softmax(logits)
    weights = weights.astype(slots.dtype)
    weights = dropout(weights, self.dropout, training)
    x = einops.einsum(weights, v, 'b h g tq tk, b tk h d -> b tq h g d')
    x = einops.rearrange(x, 'b t h g d -> b t (h g d)')
    x = self.get('proj', Linear, D_q, **kw, outscale=self.outscale)(x)
    return x


class DictConcat:

  def __init__(self, spaces, fdims, squish=lambda x: x):
    assert 1 <= fdims, fdims
    self.keys = sorted(spaces.keys())
    self.spaces = spaces
    self.fdims = fdims
    self.squish = squish

  def __call__(self, xs):
    assert all(k in xs for k in self.spaces), (self.spaces, xs.keys())
    bdims = xs[self.keys[0]].ndim - len(self.spaces[self.keys[0]].shape)
    ys = []
    for key in self.keys:
      space = self.spaces[key]
      x = xs[key]
      m = available(x, bdims=bdims)
      x = mask(x, m)
      assert x.shape[bdims:] == space.shape, (key, bdims, space.shape, x.shape)
      if space.dtype == jnp.uint8 and len(space.shape) in (2, 3):
        raise NotImplementedError('Images are not supported.')
      elif space.discrete:
        classes = np.asarray(space.classes).flatten()
        assert (classes == classes[0]).all(), classes
        classes = classes[0].item()
        x = x.astype(jnp.int32)
        x = jax.nn.one_hot(x, classes, dtype=COMPUTE_DTYPE)
      else:
        x = self.squish(x)
        x = x.astype(COMPUTE_DTYPE)
      x = mask(x, m)
      x = x.reshape((*x.shape[:bdims + self.fdims - 1], -1))
      ys.append(x)
    return jnp.concatenate(ys, -1)


class DictEmbed(nj.Module):

  squish: Callable = lambda x: x
  padone: bool = True
  bias: bool = True
  einit: str | Callable = Initializer('trunc_normal', 'out')
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  impl: str = 'onehot'

  def __init__(self, spaces, units):
    self.keys = sorted(spaces.keys())
    self.spaces = spaces
    self.units = units
    self.ekw = dict(einit=self.einit)
    self.lkw = dict(bias=self.bias, winit=self.winit, binit=self.binit)

  def __call__(self, xs, bshape):
    assert isinstance(bshape, tuple), bshape
    assert all(k in xs for k in self.spaces), (self.spaces, xs.keys())
    ys = []
    init = self.get('init', self.einit, (self.units,))
    init = jnp.broadcast_to(init, (*bshape, self.units))
    init = COMPUTE_DTYPE(init)
    ys.append(init)
    for key in self.keys:
      try:
        space = self.spaces[key]
        x = xs[key]
        assert x.dtype == space.dtype, (key, space.dtype, x.dtype, x.shape)
        m = available(x, bdims=len(bshape))
        x = mask(x, m)
        if space.discrete:
          if space.dtype == jnp.uint8 and len(space.shape) in (2, 3):
            raise NotImplementedError('Images are not supported.')
          classes = int(np.asarray(space.classes).max())
          assert classes <= 256, (key, space, classes)
          if self.impl == 'lookup':
            x = self.get(
                key, Embed, classes, self.units, space.shape,
                combine=True, **self.ekw)(x)
            # x = x.reshape((*x.shape[:len(bshape)], -1))
          elif self.impl == 'onehot':
            x = jax.nn.one_hot(x, classes, dtype=COMPUTE_DTYPE)
            x = x.reshape((*x.shape[:len(bshape)], -1))
            x = self.get(key, Linear, self.units, **self.lkw)(x)
          else:
            raise NotImplementedError(self.impl)
        else:
          x = self.squish(x)
          x = x.astype(COMPUTE_DTYPE)
          x = x.reshape((*x.shape[:len(bshape)], -1))
          x = self.get(key, Linear, self.units, **self.lkw)(x)
        x = mask(x, m)
        ys.append(x)
      except Exception:
        print(f"Error encoding key '{key}' with space {space}.")
        raise
    x = sum(ys)
    return x


class MLP(nj.Module):

  act: str = 'silu'
  norm: str = 'rms'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')

  def __init__(self, layers=5, units=1024):
    self.layers = layers
    self.units = units
    self.kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)

  def __call__(self, x):
    shape = x.shape[:-1]
    x = x.astype(COMPUTE_DTYPE)
    x = x.reshape([-1, x.shape[-1]])
    for i in range(self.layers):
      x = self.get(f'linear{i}', Linear, self.units, **self.kw)(x)
      x = self.get(f'norm{i}', Norm, self.norm)(x)
      x = act(self.act)(x)
    x = x.reshape((*shape, x.shape[-1]))
    return x


class Transformer(nj.Module):

  units: int = 1024
  layers: int = 6
  heads: int = 8
  ffup: int = 4
  act: str = 'silu'
  norm: str = 'rms'
  glu: bool = False
  rope: bool = True
  qknorm: str = 'none'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0
  concatenate_over_layers: bool = True
  normalize_out: bool = False
  dropout: float = 0.0
  aggregation: bool = False
  use_cross_attention: bool = False

  def __call__(self, x, mask=None, ts=None, text_embeds=None, training=True):
    init_kw = {k: getattr(self, k) for k in ('units', 'heads', 'ffup', 'act', 'norm', 'glu', 'rope', 'qknorm', 'bias', 'winit', 'binit', 'outscale', 'dropout')}
    init_kw['use_cross_attention'] = self.use_cross_attention
    B, T, D = x.shape
    assert D == self.units, (D, self.units)
    out = []
    if self.aggregation:
      # use a learnable token to aggregate information over input
      aggregation_token_kw = {'winit': self.winit, 'outscale': self.outscale, 'shape': (1, x.shape[-1]), 'dtype': x.dtype,}
      aggregation_token = self.get('aggregation_token', Learnable, **aggregation_token_kw)()
      aggregation_token = jnp.repeat(aggregation_token[None], B, axis=0)
      x = jnp.concatenate([x, aggregation_token], axis=1)

    for i in range(self.layers):
      with nj.scope(f'layer{i}'):
        x = self.get('transformer_layer', TransformerLayer, **init_kw)(x, mask, ts, text_embeds, training)
        out.append(x)

    if self.concatenate_over_layers:
      x = jnp.stack(out, axis=2)
      x = x.reshape((x.shape[0], x.shape[1], -1))

    if self.normalize_out:
      x = self.get('outnorm', Norm, self.norm)(x)

    if self.aggregation:
      return x[:, -1]

    return x


class GRU(nj.Module):

  units: int = 1024
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  norm: str = 'rms'
  update_bias: float = -1.0

  def initial(self, batch_size):
    return jnp.zeros((batch_size, self.units), COMPUTE_DTYPE)

  def __call__(self, carry, inputs, resets, single=False):
    assert carry.dtype == COMPUTE_DTYPE, carry.dtype
    assert inputs.dtype == COMPUTE_DTYPE, inputs.dtype
    assert resets.dtype == bool, resets.dtype
    if single:
      return self.step(carry, inputs, resets)
    carry, outputs = nj.scan(
        lambda carry, args: self.step(carry, *args),
        carry, (inputs, resets), axis=1)
    return carry, outputs

  def step(self, carry, inp, reset):
    # NOTE: When passing previous actions as input, ensure to zero out past
    # actions on is_first and clip actions to bounds if needed.
    kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    carry = mask(carry, ~reset)
    x = jnp.concatenate([carry, inp], -1)
    x = self.get('norm', Norm, self.norm)(x)
    x = self.get('linear', Linear, 3 * self.units, **kw)(x)
    res, cand, update = jnp.split(x, 3, -1)
    cand = jnp.tanh(jax.nn.sigmoid(res) * cand)
    update = jax.nn.sigmoid(update + self.update_bias)
    carry = output = update * cand + (1 - update) * carry
    return carry, output


class TransformerLayer(nj.Module):

  units: int = 1024
  heads: int = 8
  ffup: int = 4
  act: str = 'silu'
  norm: str = 'rms'
  glu: bool = False
  rope: bool = True
  qknorm: str = 'none'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0
  dropout: float = 0.0
  use_cross_attention: bool = False


  def __call__(self, x, mask=None, ts=None, text_embeds=None, training=True):
    kw = {k: getattr(self, k) for k in ('bias', 'winit', 'binit')}
    ak = {k: getattr(self, k) for k in ('heads', 'rope', 'qknorm', 'outscale', 'dropout')}
    D = x.shape[-1]
    if text_embeds is not None:
      text_embeds = text_embeds.astype(COMPUTE_DTYPE) #change from float32 to bfloat16
    assert D == self.units, (D, self.units)
    skip = x
    x = self.get('norm1', Norm, self.norm)(x)
    x = self.get('mha', Attention, **kw, **ak)(x, mask, ts, training)
    x += skip
    if self.use_cross_attention:
      skip = x
      x = self.get('norm_cross', Norm, self.norm)(x)
      cross_attn = self.get('cross_attn', CrossAttention, **kw, **ak)
      x = cross_attn(x, text_embeds, mask=mask, ts=ts, training=training)
      x += skip
    skip = x
    x = self.get('norm2', Norm, self.norm)(x)
    if self.glu:
      U = max(D, int((D * self.ffup * 2 / 3) // 32 * 32))
      ff1 = self.get('ff1', Linear, U, **kw)
      ff2 = self.get('ff2', Linear, U, **kw)
      ff3 = self.get('ff3', Linear, D, **kw, outscale=self.outscale)
      x = ff3(act(self.act)(ff1(x)) * ff2(x))
    else:
      ff1 = self.get('ff1', Linear, D * self.ffup, **kw)
      ff2 = self.get('ff2', Linear, D, **kw, outscale=self.outscale)
      x = ff2(act(self.act)(ff1(x)))

    x += skip

    return x


class ObjectCentricDynamicsLayer(nj.Module):

  units: int = 1024
  heads: int = 8
  ffup: int = 4
  act: str = 'silu'
  norm: str = 'rms'
  glu: bool = False
  qknorm: str = 'none'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0
  dropout: float = 0.0
  use_cross_attention: bool = True

  def __call__(self, x, mask=None, ts=None, text_embeds=None, training=True):
    kw = {k: getattr(self, k) for k in ('units', 'heads', 'ffup', 'act', 'norm', 'glu', 'qknorm', 'bias', 'winit', 'binit', 'outscale', 'dropout')}
    kw['rope'] = False
    kw['use_cross_attention'] = self.use_cross_attention
    B, T, num_slots, slot_dim = x.shape
    x = x.reshape(B * T, num_slots, slot_dim)
    text_embeds = text_embeds.reshape(B * T, *text_embeds.shape[2:])
    text_embeds = text_embeds[:, None, :]
    text_embeds = jnp.repeat(text_embeds, num_slots, axis=1)
    x = self.get('object_encoder_block', TransformerLayer, **kw)(x, mask=None, ts=None, training=training, text_embeds=text_embeds)
    x = x.reshape(B, T, num_slots, slot_dim)
    text_embeds = text_embeds.reshape(B, T, num_slots, -1)

    x = jnp.swapaxes(x, 1, 2).reshape(B * num_slots, T, slot_dim)
    text_embeds = jnp.swapaxes(text_embeds, 1, 2).reshape(B * num_slots, T, -1)
    mask = jnp.repeat(mask, num_slots, axis=0) if mask is not None else None
    x = self.get('time_encoder_block', TransformerLayer, **kw)(x, mask, ts=None, text_embeds=text_embeds, training=training)
    x = jnp.swapaxes(x.reshape(B, num_slots, T, slot_dim), 1, 2)

    return x


class ObjectCentricDynamics(nj.Module):

  units: int = 1024
  layers: int = 6
  heads: int = 8
  ffup: int = 4
  act: str = 'silu'
  norm: str = 'rms'
  glu: bool = False
  qknorm: str = 'none'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0
  dropout: float = 0.0
  residual: bool = False
  normalize_out: bool = False
  position_embedding: str = 'sinusoidal' # 'none', 'sinusoidal'

  def __init__(self):
    super().__init__()
    self._inv_freq = None

  def _sinusoidal_position_embedding(self, ts, num_slots, dim):
    assert ts is not None
    if self._inv_freq is None:
      inv_freq = 1 / (10000 ** (jnp.arange(0.0, dim, 2.0) / dim))
      inv_freq = inv_freq[None]
      if not isinstance(inv_freq, jax.core.Tracer):
        # inv_freq is an actual array
        self._inv_freq = inv_freq
    else:
      inv_freq = self._inv_freq

    x = einops.einsum(ts, inv_freq, 'i j, i k -> i j k')
    position_embedding = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    position_embedding = jnp.expand_dims(position_embedding, axis=-2).repeat(num_slots, axis=-2)

    return position_embedding

  def __call__(self, x, mask=None, ts=None, text_embeds=None, training=True):
    num_slots = x.shape[-2]
    input_x = x
    if self.position_embedding == 'sinusoidal':
      x = x + dropout(self._sinusoidal_position_embedding(ts, num_slots, x.shape[-1]).astype(x.dtype), self.dropout, training)

    kw = {k: getattr(self, k) for k in ('units', 'heads', 'ffup', 'act', 'norm', 'glu', 'qknorm', 'bias', 'winit', 'binit', 'outscale', 'dropout')}
    for i in range(self.layers):
      with nj.scope(f'layer{i}'):
        x = self.get('object_centric_dynamics_layer', ObjectCentricDynamicsLayer, **kw)(x, mask, ts, text_embeds, training)

    if self.residual:
      x = x + input_x

    if self.normalize_out:
      x = self.get('outnorm', Norm, self.norm)(x)

    return x


class Learnable(nj.Module):

  winit: str | Callable = Initializer('trunc_normal')
  outscale: float = 1.0

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype

  def __call__(self):
    return self.get('learnable', self._scaled_winit, self.shape).astype(self.dtype)

  def _scaled_winit(self, *args, **kwargs):
    return init(self.winit)(*args, **kwargs) * self.outscale
