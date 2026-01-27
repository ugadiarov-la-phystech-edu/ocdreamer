import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import ssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
prepend = lambda x, y: jnp.concatenate([x, y], 1)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = {
        'simple': ssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    self.dyn = {
        'rssm': ssm.RSSM,
        'tssm': ssm.TSSM,
        'octssm': ssm.ObjectCentricTSSM,
    }[config.dyn.typ](act_space, obs_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': ssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = self._build_head(scalar, config.rewhead, name='rew')
    self.con = self._build_head(binary, config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = self._build_head(act_space, config.policy, name='pol', output=outs)

    self.val = self._build_head(scalar, config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        self._build_head(scalar, config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    opt_modules = [self.dyn, self.dec, self.rew, self.con, self.pol, self.val]
    self.modules = opt_modules + [self.enc]
    if config.dyn.typ != 'octssm':
        # an encoder returns slots as is
        opt_modules.append(self.enc)

    self.opt = embodied.jax.Optimizer(
        opt_modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    if self.config.loss_scales["lm"] ==  0:
      scales.pop('lm')

    # do not use losses which have coef=0 and are not reconstructed by a decoder
    scales = {name: coef for name, coef in scales.items() if coef > 0 or name in [*self.dec.veckeys, *self.dec.imgkeys]}
    scales.update({k: rec for k in self.dec.slotkeys if k not in scales})
    scales.update({k: rec for k in self.dec.veckeys if k not in scales})
    scales.update({k: rec for k in self.dec.imgkeys if k not in scales})
    if not self.config.repval_loss:
      del scales['repval']

    self.scales = scales

  def _build_head(self, space, head_config, name, output=None):
    head_config = dict(head_config)
    if output is not None:
      head_config[head_config['typ']]['output'] = output

    if head_config['typ'] == 'mlp':
      return embodied.jax.MLPHead(space, **head_config['mlp'], name=name)
    elif head_config['typ'] == 'transformer':
      cfg = dict(head_config['transformer'])
      cfg['units'] = self.dyn.deter + self.dyn.stoch * self.dyn.classes
      return embodied.jax.AggregationTransformerHead(space, **cfg, name=name)
    else:
      raise NotImplementedError(f'Head type not implemented: {head_config["typ"]}')

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|pol)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    carry, action = self.dyn.initial_with_context(batch_size)
    text_context = None
    return (
        self.enc.initial(batch_size),
        carry,
        self.dec.initial(batch_size),
        action,
        {'is_last': jnp.ones(action['action'].shape[:2], dtype=i32)},
        text_context,
    )

  def init_train(self, batch_size):
    carry, action = self.dyn.initial(batch_size)
    return (
        self.enc.initial(batch_size),
        carry,
        self.dec.initial(batch_size),
        action,
        {'is_last': jnp.ones(action['action'].shape[:1], dtype=i32)})

  def init_report(self, batch_size):
    return self.init_train(batch_size)

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, dec_carry, prevact, is_last, text_context) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_kwargs = dict(**kw)
    if self.config.dyn.typ == 'octssm':
      text_key = self.enc.veckeys[0]
      current_text = tokens[text_key]
      if text_context is None:
        B = current_text.shape[0]
        T = prevact['action'].shape[1]
        text_context = jnp.zeros((B, T, *current_text.shape[1:]), current_text.dtype)
      text_context = prepend(text_context[:, 1:], current_text[:, None])
      dyn_kwargs['text_embeds'] = text_context
    dyn_carry, dyn_entry, feat = self.dyn.observe(dyn_carry, tokens, prevact, is_last, reset, **dyn_kwargs)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    policy = self.pol(self.feat2tensor(feat), bdims=1, training=False)
    act = sample(policy)
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))

    prevact['action'] = prepend(prevact['action'][:, 1:], act['action'][:, None])
    is_last['is_last'] = prepend(is_last['is_last'][:, 1:], obs['is_last'][:, None].astype(is_last['is_last'].dtype))
    carry = (enc_carry, dyn_carry, dec_carry, prevact, is_last, text_context)

    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, is_last, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, is_last, training=True, has_aux=True)
    metrics.update(mets)
    self.slowval.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space},
             {'is_last': data['is_last'][:, -1].astype(is_last['is_last'].dtype)})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, is_last, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training)
    text_embeds = None
    if self.config.dyn.typ == 'octssm':
      text_embeds = tokens[self.enc.veckeys[0]]
      assert text_embeds is not None, 'ObjectCentricTSSM requires text embeddings'
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, is_last, reset, training, text_embeds=text_embeds)
    losses.update(los)
    metrics.update(mets)

    dec_carry, dec_entries, recons = self.dec(dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2, training=training).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2, training=training).loss(con)

    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      #assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      target = jax.nn.one_hot(target, self.obs_space[key].high) if key == 'token' else target
      losses[key] = recon.loss(sg(target))
    if 'lm' in self.scales.keys():
      print("Adding LM loss")
      next_ac = prevact[:, :-1].reshape((-1, 1, *prevact.shape[2:]))
      context = {'feat': repfeat[:, :-1].reshape((-1, *repfeat.shape[2:]))}
      one_step = self.dec(self.dyn.imagine(next_ac, context, False))
      truth = obs['token'][:, 1:].reshape((-1, 1, *obs['token'].shape[2:]))
      nll = -(one_step["token"].log_prob(truth)).mean(-1)
      losses['lm'] = (nll * self.scales["lm"]).mean()

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, prevact, K, text_embeds=text_embeds)
    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1, training=training))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training=False)
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    los, imgloss_out, mets = imag_loss(
        imgact,
        self.rew(inp, 2, training=False).pred(),
        self.con(inp, 2, training=False).prob(1),
        self.pol(inp, 2, training=training),
        self.val(inp, 2, training=training),
        self.slowval(inp, 2, training=False),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2, training=training),
          self.slowval(inp, 2, training=False),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    #data = self.preprocess(data)
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, is_last, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    reset= obs['is_first']
    RB = min(6, B)
    metrics = {}
    enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training=False)

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, is_last, training=False)
    mets.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    text_embeds_report = None
    if self.config.dyn.typ == 'octssm':
      text_embeds_report = firsthalf(tokens[self.enc.veckeys[0]])
      assert text_embeds_report is not None, "ObjectCentricTSSM requires text embeddings in report()"
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact), firsthalf(is_last),
        firsthalf(obs['is_first']), training=False,text_embeds=text_embeds_report)
    imagination_states = jax.tree.map(lambda x: x[:, None], dyn_carry)
    imagination_actions = jax.tree.map(lambda x: x[:RB, :1], prevact)
    
    imagination_carry = self.dyn.starts(imagination_states, dyn_carry, imagination_actions, nlast=1, text_embeds=text_embeds_report)
    _, imgfeat, _ = self.dyn.imagine(
        imagination_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space},
             {'is_last': data['is_last'][:, -1].astype(is_last['is_last'].dtype)})
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact, is_last) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prevact = {k: prepend(prevact[k][:, None], data[k][:, :-1]) for k in self.act_space}
    is_last = {'is_last': prepend(is_last['is_last'][:, None], data['is_last'][:, :-1])}
    if not self.config.replay_context:
      return carry, obs, prevact, is_last, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_is_last = {'is_last': data['is_last'][:, K - 1: -1]}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, is_last, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(is_last), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_is_last, rep_stepid))
    return carry, obs, prevact, is_last, stepid

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)
  
  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log') or key in ('key',):
        continue
      if key in ('is_first', 'is_last', 'is_terminal'):
        obs[key] = value.astype(jnp.bool_)
        continue
      elif key == "token":
        value = jax.nn.one_hot(value, self.obs_space[key].high)
        value = value.astype(nn.COMPUTE_DTYPE)
      elif len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jax.tree_map(lambda x: x.astype(nn.COMPUTE_DTYPE), value) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    return obs


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
