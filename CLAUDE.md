# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orientation: this is a research fork, and `main` is not the research

`main` is (deliberately) a near-verbatim mirror of upstream `danijar/dreamerv3` — vanilla DreamerV3, JAX only. The README, `LICENSE`, `scores/`, `baselines.yaml`, and `plot.py` all come from upstream and describe *upstream*, not this fork.

The actual research — object-centric and transformer world models — lives on branches. Before doing anything, run `git branch` and confirm which branch you are on; the architecture below differs substantially between `main` and the research lineage.

```
main (vanilla dreamerv3)
└── transdreamer-dev      TSSM: transformer replaces the GRU in the world model
    ├── pos_emb → imagination
    ├── stoch-init
    ├── transdreamer → octransdreamer → dynalang    (+ homegrid/messenger, language)
    └── octransdreamer-dev        OCTSSM + PyTorch object-centric slot extractors
        └── octransdreamer-dev-shapes2d
            └── korl              <- research tip: superset of everything above
```

`korl` is the current development tip and a strict superset of the object-centric lineage (~81 files / ~10k lines over `main`). Despite the name it is not a separate idea; its unique commits are environments and infra (CausalWorld, Robosuite, ManiSkill3, periodic eval, success logging). **For any object-centric work, start from `korl`.** `dreamerv3_2023` is the old pre-rewrite codebase off a different merge-base — unrelated to current work.

Empty directories in the working tree (`embodied/torch/`, `embodied/envs/cw_envs/`, containing only `__pycache__`) are leftovers from checking out those branches; they are not part of `main`.

Untracked `z_*.py` files at the repo root are scratch JAX prototypes (attention masks, GNN pair-building, positional embeddings) — throwaway, not imported by anything.

## Commands

Training (the only real entrypoint):

```sh
python dreamerv3/main.py --logdir ~/logdir/{timestamp} --configs crafter --run.train_ratio 32
```

Every key in `dreamerv3/configs.yaml` is a CLI flag. `--configs` takes multiple named blocks applied left-to-right over `defaults`, e.g. `--configs crafter size50m`. Use `--configs <task> debug` for a fast, tiny, CPU run that exercises the whole pipeline without learning anything useful.

`--script` selects the run mode (default `train`): `train`, `train_eval`, `eval_only`, `parallel`, and the `parallel_{env,envs,replay}` workers. Re-running the same command with the same `--logdir` resumes from checkpoint.

```sh
pytest embodied/tests/test_driver.py            # run one test file
pytest embodied/tests/test_driver.py -k is_last # run one test
python -m scope.viewer --basedir ~/logdir --port 8000   # view metrics
```

There is no linter, formatter, CI, or pytest config.

**Half the test suite is dead — do not trust it as a gate.** `test_train.py` and `test_parallel.py` fail at import: they pull in `embodied/tests/utils.py`, which imports `zerofun`, the predecessor of `portal`, which is not in `requirements.txt` and not installable. `test_replay.py` calls `replay.dataset()`, which is commented out at `embodied/core/replay.py:237` (Streams replaced it). Only `test_driver.py`, `test_sampletree.py` and `test_layer_scan.py` still collect and run. Everything under `embodied/perf/` is a never-terminating benchmark with no asserts (and also imports `zerofun`) — not tests.

Despite being broken, `embodied/tests/utils.py::TestAgent` is the best executable spec of the agent contract: it asserts carry continuity by checking a synthetic `count` observation equals `carry + 1` in both `policy` and `train`. Read it, don't run it.

### Environment setup is the main source of pain

There is no single working requirements file — the pins disagree, and which one is right depends on the CUDA/driver of the target machine:

- `requirements.txt` pins `jax[cuda12]==0.4.33`, but the `Dockerfile` installs `jax[cuda]==0.5.0`.
- `requirements_v0.4.33.txt` (untracked) is the version actually used with torch-based slot extractors — it adds `comet_ml`, `omegaconf`, `moviepy`, `scikit-learn`.
- Research branches carry their own `requirements_v0.4.25.txt`.
- `install_torch.txt`, `install_ckp.txt`, `nvidia_cudnn.txt`, `z_replace` (all untracked) are working notes on getting JAX+CUDA+PyTorch to coexist — the nvidia-* pins in them matter, because installing PyTorch will otherwise pull CUDA libs that break the JAX install. Read them before touching dependencies.

## Architecture

### Config system

`dreamerv3/configs.yaml` drives everything; `defaults` is the full schema and named blocks override it. Blocks can override by **regex over the flattened key path** — e.g. `debug` sets `.*\.units: 8` and `size50m` sets `.*\.depth: 32`, which is how model size is scaled uniformly across encoder, decoder, RSSM and heads without listing each. Config is parsed in `dreamerv3/main.py:23-31` via `elements.Config` / `elements.Flags`.

`main.py` is pure wiring: it builds `make_agent` / `make_replay` / `make_env` / `make_stream` / `make_logger` closures and hands them to an `embodied.run.*` script. Adding an environment means adding a `suite: 'module:Class'` entry to the ctor dict in `make_env` (`main.py:217`) plus an `env.<suite>` config block. Task strings are `<suite>_<task>`, split on the first underscore.

### Layering

- `embodied/core/` — framework-agnostic: `Agent`/`Env`/`Stream` ABCs (`base.py`), the `Driver` (steps a batch of envs in subprocesses, dispatches `on_step` callbacks), `Replay` + `selectors` (uniform / prioritized / recency mixture), `streams`, `wrappers`, `chunk` (on-disk replay chunks).
- `embodied/jax/` — the JAX/ninjax layer: the `Agent` wrapper that JIT-compiles and shards the user agent (see below), plus `nets.py` (Linear, Conv2D, **BlockLinear**, Norm, Attention/RoPE, initializers), `heads.py` (`MLPHead`, `DictHead`), `outs.py` (output distributions: `OneHot`, `MSE`, `symexp_twohot`, `Agg`), `opt.py`, `SlowModel`, `Normalize`. Note the RSSM is *not* here — `nets.py` is generic blocks only.
- `embodied/run/` — the loops: `train`, `train_eval`, `eval_only`, `parallel`.
- `dreamerv3/` — the actual DreamerV3 model: `agent.py` (losses, actor-critic) and `rssm.py` (world model).
- `embodied/envs/` — environment adapters.

Note `dreamerv3/main.py` inserts the repo root on `sys.path`, so `embodied` resolves to the top-level `embodied/`, not `dreamerv3/embodied/` (which is a stale `__pycache__`-only shell).

### The agent contract (`embodied/core/base.py`)

An agent implements `init_policy/init_train/init_report(batch_size)`, `policy(carry, obs, mode)`, `train(carry, data)`, `report(carry, data)`, plus two properties:

- **`ext_space`** — extra keys the agent wants stored in replay alongside observations.
- **`policy_keys`** — a regex (`'^(enc|dyn|dec|pol)/'` in DreamerV3) selecting which parameters the *actor* needs. Anything the policy touches at inference must match this regex or it will not be on the actor's device.

`policy_keys` is not just a filter — it is the actor/learner weight-sync mechanism (`embodied/jax/agent.py:119-147`). Params are split in two: non-policy params are **donated** in place to the jitted train step, while policy params are returned fresh and asynchronously moved to the policy mesh, then swapped in under a lock on the next `policy()` call. Widening the regex therefore costs a device transfer every step; narrowing it too far silently starves the actor.

**`embodied.jax.Agent` is a wrapper, not a base class, despite looking like one.** `dreamerv3.agent.Agent` subclasses it, but `Agent.__new__` (`embodied/jax/agent.py:38-48`) constructs the subclass instance, stashes it as `self.model` on a *plain* `embodied.jax.Agent`, and returns that instead. So `Agent(obs_space, act_space, config)` does not return a `dreamerv3.agent.Agent` — it returns the wrapper, and your methods are called through `self.model` after being jitted, sharded and donated. This is why `dreamerv3/agent.py` methods look stateless and never mention devices.

Everything is **carry-passing, not stateful**: `carry` is a tuple `(enc_carry, dyn_carry, dec_carry, prevact)` threaded through every call, so the same code works single-step (acting) and batched-over-time (training). Each submodule provides `initial()`, `truncate()` and an `entry_space`.

**`replay_context`** is the subtle part. Rather than replaying from a zero state, recurrent state (`entries`) is written back into replay (`train()` returns `outs['replay']`) and used to warm-start the next sample of the same trajectory. `Agent._apply_replay_context` (`dreamerv3/agent.py:312`) restores it, falling back to the normal carry when `consec == 0` (first chunk). This is why a config change that alters state shapes makes an old logdir unloadable — the "Too many leaves for PyTreeDef" error.

### World model (`dreamerv3/rssm.py`)

Encoder (CNN + MLP) → tokens → `RSSM` → `feat = {deter, stoch, logit}` → Decoder + reward/cont heads. The recurrent core (`RSSM._core`, `rssm.py:135`) is a **block-diagonal GRU**: `deter` (8192 by default) is split into `blocks` (8) groups and updated with `nn.BlockLinear`, which is far cheaper than a dense recurrence at that width. `stoch` is 32 categoricals × 64 classes; `_dist` is a `OneHot` with `unimix`.

`Agent.loss` (`agent.py:156`) has three parts, all summed with `config.loss_scales`:
1. **World model** — dyn/rep KL (with `free_nats`), reconstruction per decoder key, reward, continue.
2. **Imagination** — roll the RSSM forward `imag_length` steps under the policy from `starts`, then actor-critic on the imagined trajectory (`imag_loss`, λ-returns, `retnorm`/`valnorm`/`advnorm`, slow-target critic).
3. **Replay value** (`repval_loss`) — an extra critic loss on real replayed states, bootstrapped from the imagined return.

There is an assert that `set(losses) == set(scales)` — **adding a loss term requires adding a matching `loss_scales` entry in `configs.yaml`**, or training dies at startup. The `rec` scale is expanded to one entry per decoder key (`agent.py:80-83`).

### Registries to extend

`dreamerv3/agent.py:41-49` holds three one-entry dicts mapping `config.{enc,dyn,dec}.typ` to a class. This is the extension point the research branches use: they replace `rssm.py` with `ssm.py` and register `tssm` (transformer dynamics) and `octssm` (object-centric, one latent per slot) alongside `rssm`.

### Parallel training

`--script parallel` (`embodied/run/parallel.py`) splits the loop into actor / learner / replay / env / logger processes communicating over `portal` sockets (`actor_addr`, `replay_addr`, `logger_addr` in `run`). Actor and learner can sit on different GPUs via `jax.policy_devices` / `jax.train_devices`; the learner pushes updated policy params to the actor. `--configs multicpu` fakes 8 devices on CPU (`jax.mock_devices`) for testing the sharded path.

## Conventions

- **Two-space indent**, `lambda`-heavy functional style, `f32`/`i32`/`sg`/`nn.cast` shorthands at module top. Match it.
- Networks are **ninjax** (`nj.Module`), not Flax/Haiku: submodules are created lazily inside methods via `self.sub('name', Class, ...)`, and hyperparameters are **class-level annotated attributes** with defaults (see `rssm.RSSM`), populated from the config dict. Adding a hyperparameter means adding the annotation *and* the `configs.yaml` key.
- Compute happens in `nn.COMPUTE_DTYPE` (`bfloat16` by default, `jax.compute_dtype`). Cast with `nn.cast`; there are asserts that intermediates carry the right dtype. Prioritized replay requires `bfloat16`/`float32` (asserted in `make_replay`).
- Danijar's external libraries do the heavy lifting and are not vendored: **`elements`** (config, flags, logger, checkpoint, paths, timers), **`ninjax`** (module system), **`portal`** (IPC for parallel mode), **`scope`** (metric viewer), **`granular`** (replay chunk storage).

## HomeGrid / language input (on `main`)

`main` now supports Dynalang-style language observations on HomeGrid (`--configs homegrid`, task `homegrid_{task,future,dynamics,corrections}`). The design mirrors Dynalang (`/samsung/projects/dynalang`): the `homegrid` pip package streams one T5 token id per env step (vocab 32100, `<pad>` between utterances); the adapter (`embodied/envs/homegrid.py`) emits it as a scalar discrete `token` obs (or `token_embed`, a 512-d T5 embedding, via `env.homegrid.lang`). No model changes were needed: `nn.DictConcat` already one-hots discrete vec keys into the encoder MLP, and the decoder's `DictHead` reconstructs them with a `categorical` head — which is exactly Dynalang's one-hot→MLP encoder and onehot reconstruction loss. Dynalang's separate LM loss is intentionally not ported (it defaults to scale 0.0 in Dynalang's own HomeGrid runs). `outs.Categorical.logp` casts events to int32 so bool observations (e.g. `is_read_step`) can be decoded. The full-size `homegrid` block (batch 16×256, deter 4096) needs a data-center GPU; on small GPUs use `--configs homegrid size12m --batch_size 8 --batch_length 64`.

## Object-centric branches (`korl` and ancestors)

If you check out `korl` or another OC branch, the following is true *in addition* to the above.

`rssm.py` is replaced by `dreamerv3/ssm.py`, which adds an `AbstractSSM` base and three registered dynamics: `rssm` (upstream), `tssm` (causal transformer with RoPE over a `max_context_length` window instead of the GRU), and `octssm` (`ObjectCentricTSSM` — requires a `slot` obs key; carry becomes `deter: (num_slots, deter)`, and the action is embedded as an extra "action slot"). The interleaved slot/time attention (`ObjectCentricDynamics` in `embodied/jax/nets.py`) alternates a block attending over the slot axis with a causal block attending over time. Heads become pluggable (`typ: mlp|transformer`), where the transformer head attention-aggregates over slots. The agent carry grows an `is_last` window so transformer attention cannot cross episode boundaries.

**The PyTorch/JAX bridge is the thing to understand.** The object-centric perception model (DINOSAUR-style `DinoV2saur`, or `SLATE`, under `embodied/torch/ocr/`) is **frozen, pretrained, and never touches JAX**. It runs in the env process: `BatchSlotExtractorEnv` (`embodied/core/wrappers.py`) calls the torch model on `obs['image']` and injects `obs['slot']` of shape `(n_slots, dim)` as an ordinary numpy observation. The agent just sees a `slot` key. So the boundary is numpy-over-pipes, and the extractor is configured entirely under `agent.batch_env.*` (`use_slot_extractor`, `slot_extractor.{typ,config_path,checkpoint_path,device}`). Because the slot encoder has no trainable JAX weights, `agent.py` excludes the encoder from the optimizer's module list when `dyn.typ == 'octssm'`. Use `agent.exclude_obs_keys{,_from_training}` to keep the raw image out of replay.

This means an OC run needs **both** a working JAX-CUDA install and a PyTorch install on the same machine, plus a pretrained OCR checkpoint — which is what the `install_*.txt` notes are about.
