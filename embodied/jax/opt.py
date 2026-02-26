import math
import re 

import jax
import jax.numpy as jnp
import ninjax as nj
from dreamerv3.ninjax_local import Module
import optax

from . import internal
from . import nets

f32 = jnp.float32
i32 = jnp.int32
sg = jax.lax.stop_gradient
tree_map = jax.tree_util.tree_map
tree_leaves = jax.tree_util.tree_leaves
COMPUTE_DTYPE = jnp.float32


# class Optimizer(nj.Module):

#   summary_depth: int = 2

#   def __init__(self, modules, opt):
#     modules = modules if isinstance(modules, (list, tuple)) else (modules,)
#     self.modules = modules
#     self.opt = opt
#     self.step = nj.Variable(jnp.array, 0, i32, name='step')
#     self.scaling = (nets.COMPUTE_DTYPE == jnp.float16)
#     if self.scaling:
#       self.opt = optax.apply_if_finite(self.opt, max_consecutive_errors=1000)
#       self.grad_scale = nj.Variable(jnp.array, 1e4, f32, name='grad_scale')
#       self.good_steps = nj.Variable(jnp.array, 0, i32, name='good_steps')

#   def __call__(self, lossfn, *args, has_aux=False, **kwargs):
#     metrics = {}

#     def lossfn2(*args, **kwargs):
#       outs = lossfn(*args, **kwargs)
#       loss, aux = outs if has_aux else (outs, None)
#       assert loss.dtype == f32, (self.name, loss.dtype)
#       assert loss.shape == (), (self.name, loss.shape)
#       if self.scaling:
#         loss *= sg(self.grad_scale.read())
#       return loss, aux

#     loss, params, grads, aux = nj.grad(
#         lossfn2, self.modules, has_aux=True)(*args, **kwargs)
#     if self.scaling:
#       loss *= 1 / self.grad_scale.read()

#     counts = {k: math.prod(v.shape) for k, v in params.items()}
#     if nj.creating():
#       print(self._summarize_params(counts, self.summary_depth))

#     axes = internal.get_data_axes()
#     if axes:
#       grads = jax.tree.map(lambda x: jax.lax.pmean(x, axes), grads)

#     if self.scaling:
#       invscale = 1 / self.grad_scale.read()
#       grads = jax.tree.map(lambda x: x * invscale, grads)

#     state = self.sub('state', nj.Tree, self.opt.init, params)
#     updates, new_state = self.opt.update(grads, state.read(), params)
#     nj.context().update(optax.apply_updates(params, updates))
#     state.write(new_state)
#     grad_norm = optax.global_norm(grads)
#     if self.scaling:
#       self._update_scale(grads, jnp.isfinite(grad_norm))
#       grad_norm = jnp.where(jnp.isfinite(grad_norm), grad_norm, jnp.nan)
#       self.step.write(self.step.read() + i32(jnp.isfinite(grad_norm)))
#       metrics['grad_scale'] = self.grad_scale.read()
#       metrics['grad_overflow'] = f32(~jnp.isfinite(grad_norm))
#     else:
#       self.step.write(self.step.read() + 1)
#     metrics['loss'] = loss.mean()
#     metrics['updates'] = self.step.read()
#     metrics['grad_norm'] = grad_norm
#     metrics['grad_rms'] = nets.rms(grads)
#     metrics['update_rms'] = nets.rms(updates)
#     metrics['param_rms'] = nets.rms([x.values for x in self.modules])
#     metrics['param_count'] = jnp.array(list(counts.values()), f32).sum()
#     metrics = {f'{self.name}/{k}': v for k, v in metrics.items()}
#     return (metrics, aux) if has_aux else metrics

#   def _update_scale(self, grads, finite):
#     keep = (finite & (self.good_steps.read() < 1000))
#     incr = (finite & (self.good_steps.read() >= 1000))
#     decr = ~finite
#     self.good_steps.write(i32(keep) * (self.good_steps.read() + 1))
#     self.grad_scale.write(jnp.clip(
#         f32(keep) * self.grad_scale.read() +
#         f32(incr) * self.grad_scale.read() * 2 +
#         f32(decr) * self.grad_scale.read() / 2, 1e-4, 1e5))
#     return finite

#   def _summarize_params(self, counts, depth):
#     lines = []
#     pfxs = []
#     for key in counts:
#       parts = key.split('/')
#       pfxs += ['/'.join(parts[: i + 1]) for i in range(min(len(parts), depth))]
#     subcounts = {
#         prefix: sum(v for k, v in counts.items() if k.startswith(prefix))
#         for prefix in set(pfxs)}
#     lines = [f'Optimizer {self.name} has {sum(counts.values()):,} params:']
#     for prefix, count in sorted(subcounts.items(), key=lambda x: -x[1]):
#       lines.append(f'{count:>14,} {prefix}')
#     return '\n'.join(lines)

class Optimizer(Module):

  PARAM_COUNTS = {}

  def __init__(
      self, lr, opt='adam', eps=1e-5, clip=100.0, warmup=0, wd=0.0,
      wd_pattern=r'/(w|kernel)$', lateclip=0.0, frozen_keys=r'^$'):
    assert wd_pattern[0] not in ('0', '1')
    # assert self.path not in self.PARAM_COUNTS
    self.PARAM_COUNTS[self.path] = None
    wd_pattern = re.compile(wd_pattern)
    frozen_keys = re.compile(frozen_keys)
    chain = []
    if clip:
      chain.append(optax.clip_by_global_norm(clip))
    if opt == 'adam':
      chain.append(optax.scale_by_adam(eps=eps))
    elif opt == 'lion':
      chain.append(optax.scale_by_lion())
    else:
      raise NotImplementedError(opt)
    if lateclip:
      chain.append(late_grad_clip(lateclip))
    if wd:
      chain.append(optax.additive_weight_decay(wd, lambda params: (
          tree_map(lambda k: bool(wd_pattern.search(k)), tree_keys(params)))))
    if warmup:
      schedule = optax.linear_schedule(0.0, -lr, warmup)
      chain.append(optax.inject_hyperparams(optax.scale)(schedule))
    else:
      chain.append(optax.scale(-lr))
    def partition_params_fn(params): 
      def label_key(k):
        label = "frozen" if bool(frozen_keys.search(k)) else "trainable"
        if label == "frozen":
          print(f" {k}")
        return label
      print("Frozen keys: ")
      return tree_map(label_key, tree_keys(params))

    self.opt = optax.multi_transform(
      {"trainable": optax.chain(*chain), "frozen": optax.set_to_zero()},
      partition_params_fn
    )
#    self.opt = optax.chain(*chain)
    self.step = nj.Variable(jnp.array, 0, jnp.int32, name='step')
    self.scaling = (COMPUTE_DTYPE == jnp.float16)
    if self.scaling:
      self.opt = optax.apply_if_finite(self.opt, max_consecutive_errors=1000)
      self.grad_scale = nj.Variable(
          jnp.array, 1e4, jnp.float32, name='grad_scale')
      self.good_steps = nj.Variable(
          jnp.array, 0, jnp.int32, name='good_steps')

  def __call__(self, modules, lossfn, *args, has_aux=False, **kwargs):
    def wrapped(*args, **kwargs):
      outs = lossfn(*args, **kwargs)
      loss, aux = outs if has_aux else (outs, None)
      assert loss.dtype == jnp.float32, (self.name, loss.dtype)
      assert loss.shape == (), (self.name, loss.shape)
      if self.scaling:
        loss *= sg(self.grad_scale.read())
      return loss, aux
    metrics = {}
    loss, params, grads, aux = nj.grad(
        wrapped, modules, has_aux=True)(*args, **kwargs)
    if not self.PARAM_COUNTS[self.path]:
      count = sum([math.prod(x.shape) for x in tree_leaves(params)])
      print(f'Optimizer {self.name} has {count:,} variables.')
      self.PARAM_COUNTS[self.path] = count
    if parallel():
      grads = tree_map(lambda x: jax.lax.pmean(x, 'i'), grads)
    if self.scaling:
      grads = tree_map(lambda x: x / self.grad_scale.read(), grads)
      finite = self._update_scale(grads)
      metrics[f'{self.name}_grad_scale'] = self.grad_scale.read()
      metrics[f'{self.name}_grad_overflow'] = (~finite).astype(jnp.float32)
    optstate = self.get('state', self.opt.init, params)
    updates, optstate = self.opt.update(grads, optstate, params)
    self.put('state', optstate)
    nj.context().update(optax.apply_updates(params, updates))
    norm = optax.global_norm(grads)
    if self.scaling:
      norm = jnp.where(jnp.isfinite(norm), norm, jnp.nan)
    self.step.write(self.step.read() + jnp.isfinite(norm).astype(jnp.int32))
    metrics['loss'] = loss.mean()
    metrics['grad_norm'] = norm
    metrics['grad_steps'] = self.step.read()
    metrics = {f'{self.name}_{k}': v for k, v in metrics.items()}
    return (metrics, aux) if has_aux else metrics

  def _update_scale(self, grads):
    finite = jnp.array([
        jnp.isfinite(x).all() for x in jax.tree_util.tree_leaves(grads)]).all()
    keep = (finite & (self.good_steps.read() < 1000))
    incr = (finite & (self.good_steps.read() >= 1000))
    decr = ~finite
    self.good_steps.write(
        keep.astype(jnp.int32) * (self.good_steps.read() + 1))
    self.grad_scale.write(jnp.clip(
        keep.astype(jnp.float32) * self.grad_scale.read() +
        incr.astype(jnp.float32) * self.grad_scale.read() * 2 +
        decr.astype(jnp.float32) * self.grad_scale.read() / 2,
        1e-4, 1e4))
    return finite


def late_grad_clip(value=1.0):
  def init_fn(params):
    return ()
  def update_fn(updates, state, params):
    updates = tree_map(lambda x: jnp.clip(x, -value, value), updates)
    return updates, ()
  return optax.GradientTransformation(init_fn, update_fn)


def tree_keys(params, prefix=''):
  if hasattr(params, 'items'):
    return type(params)({
        k: tree_keys(v, prefix + '/' + k.lstrip('/'))
        for k, v in params.items()})
  elif isinstance(params, (tuple, list)):
    return [tree_keys(x, prefix) for x in params]
  elif isinstance(params, jnp.ndarray):
    return prefix
  else:
    raise TypeError(type(params))

def parallel():
  try:
    jax.lax.axis_index('i')
    return True
  except NameError:
    return False

def clip_by_agc(clip=0.3, pmin=1e-3):

  def init_fn(params):
    return ()

  def update_fn(updates, state, params=None):
    def fn(param, update):
      unorm = jnp.linalg.norm(update.flatten(), 2)
      pnorm = jnp.linalg.norm(param.flatten(), 2)
      upper = clip * jnp.maximum(pmin, pnorm)
      return update * (1 / jnp.maximum(1.0, unorm / upper))
    updates = jax.tree.map(fn, params, updates) if clip else updates
    return updates, ()

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_rms(beta=0.999, eps=1e-8):

  def init_fn(params):
    nu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, nu)

  def update_fn(updates, state, params=None):
    step, nu = state
    step = optax.safe_int32_increment(step)
    nu = jax.tree.map(
        lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
    nu_hat = optax.bias_correction(nu, beta, step)
    updates = jax.tree.map(
        lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
    return updates, (step, nu)

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_momentum(beta=0.9, nesterov=False):

  def init_fn(params):
    mu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, mu)

  def update_fn(updates, state, params=None):
    step, mu = state
    step = optax.safe_int32_increment(step)
    mu = optax.update_moment(updates, mu, beta, 1)
    if nesterov:
      mu_nesterov = optax.update_moment(updates, mu, beta, 1)
      mu_hat = optax.bias_correction(mu_nesterov, beta, step)
    else:
      mu_hat = optax.bias_correction(mu, beta, step)
    return mu_hat, (step, mu)

  return optax.GradientTransformation(init_fn, update_fn)
