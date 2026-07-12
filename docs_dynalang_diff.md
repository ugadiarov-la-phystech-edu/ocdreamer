# DreamerV3 `main` vs Dynalang `main`: comparison

Result of a side-by-side read of both codebases (July 2026):

- DreamerV3: `/samsung/projects/ocdreamer`, branch `main` (2025 rewrite of danijar/dreamerv3)
- Dynalang: `/samsung/projects/dynalang`, branch `main` (fork of DreamerV3-2023)

Reference run being matched (Dynalang xlarge on HomeGrid):

```sh
python dynalang/train.py --configs xlarge --run.script train --task homegrid_future \
  --envs.amount 66 --encoder.mlp_keys token$ --decoder.mlp_keys token$ \
  --decoder.vector_dist onehot --batch_size 16 --batch_length 256 \
  --run.log_every 300 --run.train_ratio 32
```

The DreamerV3 counterpart is the `homegrid_xl` config block in `dreamerv3/configs.yaml`:

```sh
python dreamerv3/main.py --configs homegrid_xl --logdir ~/logdir/{timestamp} --seed N
```

## 1. Irreducible differences

Differences that survive after configuring DreamerV3 as closely as possible; none of
these can be closed by config alone.

### 1.1 Optimizer structure (most significant)

- **Dynalang**: three separate **true Adam** optimizers
  (`optax.scale_by_adam`, jaxutils.py:392) —
  world model (lr 1e-4, eps 1e-8, global-norm clip 1000),
  actor (lr 3e-5, eps 1e-5, clip 100),
  critic (lr 3e-5, eps 1e-5, clip 100). No warmup.
- **DreamerV3**: one optimizer over all parameters (lr 4e-5), and it is **not
  literal Adam**: the chain (agent.py:358-360, opt.py:126-164) RMS-normalizes the
  gradient first and applies momentum to the *normalized* update —
  `EMA_0.9(g/√v̂)` rather than Adam's `EMA_0.9(g)/√v̂` (the LaProp ordering).
  Same moments (0.9/0.999, bias-corrected), but eps 1e-20, and clipped with
  **AGC 0.3** (per-parameter, relative to parameter norm) instead of global-norm.
- Main has no per-module learning rates and no global-norm clipping option.

### 1.2 Learned initial RSSM state

- **Dynalang** (`initial: learned`): initial `deter = tanh(learned parameter)`,
  initial `stoch` sampled from the prior given that deter. This is also the reset
  target for `is_first` steps *inside* training sequences.
- **DreamerV3**: resets to zeros in both places; no `initial` option exists.

### 1.3 GRU input wiring

Gate math is identical in both (see §3), but the path into the gates differs:

- **Dynalang**: one Linear embeds `[stoch, action]`; result is concatenated with the
  **raw** deter; a single bias-free Linear with LayerNorm on the 3×deter
  pre-activations produces the gates.
- **DreamerV3**: deter, stoch, action are embedded **separately**, each with its own
  norm + activation; the gate projection (`BlockLinear`) has no norm on its output.
- `dynlayers: 0` removes main's extra hidden layer, but the embedding structure
  still differs.

### 1.4 Decoder image output

- **DreamerV3**: always applies **sigmoid** to the image mean.
- **Dynalang**: outputs the raw value **+ 0.5** (no squashing).
- Additionally main's decoder space-projection Linear is followed by norm +
  activation; Dynalang's input Linear is plain.

### 1.5 Linear bias placement

- **Dynalang**: drops the bias on any Linear followed by a norm (the norm's shift
  takes its place).
- **DreamerV3**: keeps bias everywhere, in addition to the norm shift.
- Pure parameterization difference; no flag.

### 1.6 Imagination first-step continuation

- **Dynalang**: discount of the first imagined step uses ground-truth
  `1 − is_terminal`.
- **DreamerV3**: uses the continue head's prediction for all steps.

### 1.7 Replay window alignment

With `replay_context: 0`, main forces `is_first[0] = True` on sampled windows, so
both codebases train every window from a fresh initial state (2023 semantics). But:

- **Dynalang**: uniform stride-1 windows at **any** offset into an episode.
- **DreamerV3**: chunk/stream-aligned `Consec` window sampling.
- A sampling-distribution difference with no config knob.

### 1.8 `real_env_step` accounting

Dynalang's HomeGrid loop counts "real" env steps (excluding `is_read_step` reading
steps) for logging/schedules. Main has no equivalent. Logging-only; no learning
impact.

### 1.9 LM-loss / text-pretraining code paths

Dynalang carries a token-prediction (LM) loss and pretrained-LM loading machinery
that main lacks entirely. Inactive in the reference run anyway
(`loss_scales.lm: 0.0` in Dynalang's own HomeGrid configs).

## 2. Reducible but deliberately not applied

- **Precision**: Dynalang runs float16 with dynamic loss scaling (init 1e4,
  apply_if_finite 1000); main defaults to bfloat16 but supports the identical f16
  scheme via `--jax.compute_dtype float16`. Kept at bf16 because it is strictly
  more stable.

## 3. Verified equivalent (no action needed)

Checked in code on both sides; mathematically identical:

- GRU gate equations: `reset = sigmoid`, `cand = tanh(reset · cand)`,
  `update = sigmoid(update − 1)`.
- λ-return recursion; reinforce actor loss + entropy bonus 3e-4.
- Critic: two-hot cross-entropy + slow-critic regularizer (logprob, scale 1.0),
  slow critic EMA rate 0.02 every step.
- Return normalization: 5/95 percentile EMA (rate 0.01), denominator
  `max(1, hi − lo)`.
- `symexp_twohot` bins (255, symexp-spaced) — identical bin placement.
- Weight init: main's `trunc_normal_avg` ≡ Dynalang's `'normal'`
  (truncated normal [−2, 2] × 1.1368 ≈ 1/0.8796, fan-avg).
- LayerNorm eps 1e-3 (via main's `layer1em3` norm name).
- Free nats (1.0) applied elementwise; KL stop-gradient structure
  (dyn: sg(post)‖prior, rep: post‖sg(prior)).
- Action soft-clip `action /= sg(max(1, |action|))` ≡ Dynalang `action_clip 1.0`.
- Image normalization `/255 − 0.5`.
- Categorical policy distribution ≡ Dynalang's onehot under reinforce gradients.
- Encoder/decoder shape: Dynalang xlarge uses `cnn_blocks: 0`, i.e. a plain stack of
  4× stride-2 kernel-4 convs (depths 96/192/384/768, minres 4) — matched by
  `kernel: 4, strided: True, depth: 96, mults: [1, 2, 4, 8]`.
- `train_ratio` semantics; wall-clock-based log cadence; env unseeded in both.

## 4. Reducible gaps folded into `homegrid_xl`

| Dynalang behavior | DreamerV3 setting |
| --- | --- |
| LayerNorm, eps 1e-3 | `.*\.norm: layer1em3` |
| Truncated-normal fan-avg init | `.*\.winit: trunc_normal_avg` |
| Dense (non-block) GRU | `dyn.rssm.blocks: 1` |
| Single img-input layer, no hidden gate layer | `imglayers: 1, dynlayers: 0` |
| deter 4096, 32 classes (xlarge) | `dyn.rssm: {deter: 4096, classes: 32}` |
| Plain 4×4-kernel strided CNN, depth 96 | `enc/dec.simple: {depth: 96, mults: [1,2,4,8], kernel: 4, strided: True}` |
| 5-layer MLPs and heads | `layers: 5` on enc/dec/heads/policy/value |
| KL dyn scale 0.5; no replay-value loss | `loss_scales: {dyn: 0.5, repval: 0.0}` |
| Undiscounted continues in imagination | `contdisc: False` |
| No LR warmup | `opt.warmup: 0` |
| Fresh-reset training windows | `replay_context: 0` |
| No online replay prioritization of recent | `replay.online: False` |
| `is_read_step` not fed to the model | `env.homegrid.obs_read_step: False` |
| 66 envs, batch 16×256, train_ratio 32 | `run.envs: 66, batch_size: 16, batch_length: 256` |

## 5. Verification status

- Older `homegrid_xl` (before §4 additions of norm/winit/kernel/dynlayers/warmup/
  replay settings): verified full-scale on H100 — token loss 10.05 → 3.59 over 73k
  steps, ~205 fps with 66 envs, all metrics finite.
- Updated `homegrid_xl` (current): verified end-to-end at tiny scale on CPU
  (2026-07-11) — training steps run, all losses finite, token loss starts at
  10.37 = ln(32100) as expected. Full-scale re-verification pending (previous
  remote machine expired before the updated config could be synced).
