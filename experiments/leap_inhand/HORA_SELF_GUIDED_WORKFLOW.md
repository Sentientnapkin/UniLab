# Self-Guided HORA Experiment Workflow

This runbook is the source of truth for continuing HORA experiments without an
LLM. Commands assume the repository is `/Users/School/Research/UniLab-hora` and
the branch is `feat/leap-hora`.

## Non-negotiable rules

1. Never select a checkpoint from one interactive rollout.
2. Give every run a new directory; never overwrite a prior run.
3. Sweep the same deterministic manifest after every training block.
4. Change one curriculum axis per block.
5. Do not train the zero-extrinsics control until exact extrinsics rotates.
6. Do not start Stage 2 until exact extrinsics beats the zero-extrinsics control.
7. Keep `run_config.json`, `run_summary.json`, every `evaluation_N.json`, and
   `selection.json` with the checkpoints.

The recurring `exclude-dependencies` warning from `uv` is currently harmless.

## One-command checkpoint selection

`sweep_leap_hora_checkpoints.py` discovers checkpoints, resumes interrupted
sweeps, evaluates missing checkpoints, applies stage-specific gates, and writes:

- `checkpoint_sweep/evaluation_N.json`: full deterministic metrics;
- `checkpoint_sweep/evaluation_N.log`: evaluator output and failures;
- `checkpoint_sweep/selection.json`: machine-readable ranking;
- `checkpoint_sweep/selection.md`: human-readable table and gate failures.

Checkpoint specifications accept integers and inclusive ranges, for example
`0:100:10,99`. Omit `--checkpoints` to evaluate every saved checkpoint. Re-run
the same command safely: completed evaluations are reused. Add `--force` only
when code, the manifest, or evaluation semantics changed.

## Phase 0: repository and baseline check

```bash
cd /Users/School/Research/UniLab-hora
git branch --show-current
git status --short
uv run --no-sync pytest -q tests/tasks/test_leap_hora.py -m 'not slow'
```

Expected branch: `feat/leap-hora`. Preserve the gait-assisted run separately.
Evaluate a gait checkpoint directly with `evaluate_leap_inhand.py` and the same
manifest whenever the manifest or metrics change.

## Phase 1: grasp-cache pipeline

The current validated artifact is
`leap_precision_grasps_v5_stage_a_weighted.npz`. To reproduce it:

```bash
uv run --no-sync python src/unilab/scripts/build_leap_hora_v5_candidates.py

uv run --no-sync python src/unilab/scripts/build_leap_hora_contact_cache.py \
  --input src/unilab/assets/caches/hora/leap_precision_grasps_v5_candidates.npz \
  --output-prefix src/unilab/assets/caches/hora/leap_precision_grasps_v5 \
  --max-control-steps 35000 --num-envs 64 --seed 42 \
  --cache-size 2000 --anchor-fraction 0.8 --middle-contact-fraction 0.4 \
  --minimum-raw-thumb-contact 0.5 --minimum-raw-middle-contact 0.5 \
  --minimum-raw-opposing-contact 0.5 --minimum-raw-opposing-fingers 2

uv run --no-sync python src/unilab/scripts/rebuild_leap_hora_stage_a.py
```

Read `leap_precision_grasps_v5_stage_a_weighted.json`. Required gates:

- 2,000 output rows;
- at least 95% unique source rows;
- exactly 80% anchor and 20% recoverable;
- at least 800 unique middle-enriched rows;
- zero invalid/marginal rows in Stage A.

Do not expand geometry until the current tier learns. The current middle-contact
expansion covers only the 32 mm radius, 70 mm cylinder.

## Stage A: stabilization and recovery

Train from scratch:

```bash
uv run train --algo ppo --task leap_inhand_hora_stage1 --sim mujoco \
  --profile stabilize \
  algo.max_iterations=400 algo.save_interval=25 algo.seed=1 \
  training.log_dir=/Users/School/Research/UniLab-hora/runs/hora_stage1/stabilize_v5_seed1 \
  training.no_play=true training.logger=tensorboard
```

Select automatically:

```bash
uv run --no-sync python src/unilab/scripts/sweep_leap_hora_checkpoints.py \
  --run-dir runs/hora_stage1/stabilize_v5_seed1 \
  --profile stabilize --stage stabilize \
  --checkpoints 0:375:25,399 --require-pass
```

The stabilization selector requires median time-to-fall of at least 16 seconds
in every split, at least 80% horizon completion, median valid-grasp fraction
0.70, median middle contact 0.40, and no palm/joint-limit exploitation. Among
passing checkpoints it balances valid grasp and middle contact using their
harmonic mean. The current selected transition checkpoint is 350.

## Stage B2: introduce signed rotation

Use the Stage A checkpoint recorded in `selection.json`, replacing `N` below:

```bash
uv run train --algo ppo --task leap_inhand_hora_stage1 --sim mujoco \
  --profile rotate_b2 \
  algo.load_run=/Users/School/Research/UniLab-hora/runs/hora_stage1/stabilize_v5_seed1/model_N.pt \
  algo.max_iterations=100 algo.save_interval=10 algo.seed=1 \
  training.log_dir=/Users/School/Research/UniLab-hora/runs/hora_stage1/rotate_b2_v5_seed1 \
  training.no_play=true training.logger=tensorboard

uv run --no-sync python src/unilab/scripts/sweep_leap_hora_checkpoints.py \
  --run-dir runs/hora_stage1/rotate_b2_v5_seed1 \
  --profile rotate_b2 --stage rotate \
  --checkpoints 0:90:10,99 --require-pass
```

The rotation selector first enforces survival/contact gates, then ranks net
committed rotation, directional speed, valid grasp, middle contact, and reversal
count. Stop B2 if no checkpoint has positive median net rotation and speed.
Increasing iterations is not a substitute for fixing a failed objective.

## Stage B3: cyclic regrasp and pose preservation

If B2 rotates but steadily deforms the hand, continue the selected B2 checkpoint
under the stronger HORA pose penalty:

```bash
uv run train --algo ppo --task leap_inhand_hora_stage1 --sim mujoco \
  --profile rotate_b3 \
  algo.load_run=/Users/School/Research/UniLab-hora/runs/hora_stage1/rotate_b2_v5_seed1/model_N.pt \
  algo.max_iterations=100 algo.save_interval=10 algo.seed=1 \
  training.log_dir=/Users/School/Research/UniLab-hora/runs/hora_stage1/rotate_b3_v5_seed1 \
  training.no_play=true training.logger=tensorboard

uv run --no-sync python src/unilab/scripts/sweep_leap_hora_checkpoints.py \
  --run-dir runs/hora_stage1/rotate_b3_v5_seed1 \
  --profile rotate_b3 --stage rotate \
  --checkpoints 0:90:10,99 --require-pass
```

Choose B3 only if it preserves B2's positive rotation while improving valid
grasp, reversals, or pose behavior. Otherwise retain the B2 checkpoint.

## Stage B4: LEAP pose-penalty bridge

The v5 B3 sweep selected checkpoint 499: it recovered middle-finger contact,
valid-grasp fraction, and positive signed rotation, but its median net rotation
and directional speed remained only 0.106 rad and 0.0059 rad/s. Continue from
that checkpoint with one change: relax the initial-grasp pose penalty from
`-0.3` to `-0.1`. This tests whether the literal Allegro coefficient is
preventing the LEAP hand from regrasping.

```bash
uv run train --algo ppo --task leap_inhand_hora_stage1 --sim mujoco \
  --profile rotate_b4 \
  algo.load_run=/Users/School/Research/UniLab-hora/runs/hora_stage1/rotate_b3_v5_seed1/model_499.pt \
  algo.max_iterations=50 algo.save_interval=5 algo.seed=1 \
  training.log_dir=/Users/School/Research/UniLab-hora/runs/hora_stage1/rotate_b4_v5_seed1 \
  training.no_play=true training.logger=tensorboard

uv run --no-sync python src/unilab/scripts/sweep_leap_hora_checkpoints.py \
  --run-dir runs/hora_stage1/rotate_b4_v5_seed1 \
  --profile rotate_b4 --stage rotate --require-pass
```

The sweep automatically discovers the saved checkpoint indices. Advance only
if B4 preserves the rotation-stage gates and materially exceeds B3 checkpoint
499. Use 0.25 rad median net rotation and 0.0125 rad/s median directional speed
as the minimum useful improvement targets. If B4 remains slow, retain B3 499
and run another bounded single-variable ablation instead of extending B4.

## Stage B5: stronger axial-velocity signal

The B4 sweep selected checkpoint 535 for survival and grasp quality, but it did
not beat B3 checkpoint 499 on rotation. Test a single conservative objective
change before altering action semantics: double `rotate` from `1.0` to `2.0`
while retaining B4's `retained=0.5` and `pose_diff=-0.1` settings.

```bash
uv run train --algo ppo --task leap_inhand_hora_stage1 --sim mujoco \
  --profile rotate_b5 \
  algo.load_run=/Users/School/Research/UniLab-hora/runs/hora_stage1/rotate_b4_v5_seed1/model_535.pt \
  algo.max_iterations=50 algo.save_interval=5 algo.seed=1 \
  training.log_dir=/Users/School/Research/UniLab-hora/runs/hora_stage1/rotate_b5_v5_seed1 \
  training.no_play=true training.logger=tensorboard

uv run --no-sync python src/unilab/scripts/sweep_leap_hora_checkpoints.py \
  --run-dir runs/hora_stage1/rotate_b5_v5_seed1 \
  --profile rotate_b5 --stage rotate --require-pass
```

Do not extend B5 merely because it passes the basic rotation-stage gates. It
must reach at least 0.25 rad median net rotation and 0.0125 rad/s directional
speed without a survival/contact regression. Reject gains caused primarily by
more reversals. If B5 remains static, reduce the artificial retention reward in
the next isolated ablation rather than increasing rotation again.

## Stage 1 exact-extrinsics gate and multi-seed confirmation

### Clean B8 PPO restart from the Stage A actor

When changing from the LEAP curriculum reward to the compact HORA reward, do
not fully resume a mature checkpoint. Full resume carries a critic, optimizer,
adaptive learning rate, iteration, and logger counters trained for the previous
objective. Use `mujoco_rotate_b8_restart` to load only the Stage A actor
(including its extrinsics encoder, hardware normalizer, and action
distribution) from checkpoint 350. The critic and optimizer are newly
initialized and training starts at iteration zero.

```bash
uv run train --algo ppo --task leap_inhand_hora_stage1 --sim mujoco \
  --profile rotate_b8_restart \
  algo.load_run=/Users/School/Research/UniLab-hora/runs/hora_stage1/stabilize_v5_seed1/model_350.pt \
  algo.max_iterations=200 algo.save_interval=10 algo.seed=1 \
  training.log_dir=/Users/School/Research/UniLab-hora/runs/hora_stage1/rotate_b8_restart_v5_seed1 \
  training.no_play=true training.logger=tensorboard

uv run --no-sync python src/unilab/scripts/sweep_leap_hora_checkpoints.py \
  --run-dir runs/hora_stage1/rotate_b8_restart_v5_seed1 \
  --profile rotate_b8_restart --stage rotate --require-pass
```

The restart should save checkpoints from zero rather than continuing from 350.
The console must say `Loading actor_only checkpoint`. If it says `full`, stop
the run because the critic and optimizer are being resumed unintentionally.

Before widening randomization, require a final exact-extrinsics checkpoint to
pass the project gates:

```bash
uv run --no-sync python src/unilab/scripts/sweep_leap_hora_checkpoints.py \
  --run-dir RUN_DIRECTORY --profile PROFILE --stage final --require-pass
```

The final selector requires 16-second median survival in every split, 80%
horizon completion, 6 radians median net rotation, 0.25 rad/s median directional
speed, no more than 10 median reversals, and no palm/joint-limit exploitation.
These are project gates, not HORA-paper claims.

Once seed 1 passes, repeat the identical train/sweep commands with seeds 2 and
3 and distinct run directories. Do not change hyperparameters between seeds.
Advance only when the median seed passes; never report only the best seed.

## Exact versus zero-extrinsics control

Only after exact extrinsics rotates, train the matched zero-latent control from
scratch with the same cache, reward, seed, iteration budget, and save interval:

```bash
uv run train --algo ppo --task leap_inhand_hora_stage1 --sim mujoco \
  --profile no_extrinsics \
  algo.max_iterations=400 algo.save_interval=25 algo.seed=1 \
  training.log_dir=/Users/School/Research/UniLab-hora/runs/hora_stage1/zero_seed1 \
  training.no_play=true training.logger=tensorboard

uv run --no-sync python src/unilab/scripts/sweep_leap_hora_checkpoints.py \
  --run-dir runs/hora_stage1/zero_seed1 \
  --profile no_extrinsics --stage final --require-pass
```

Compare the recommended checkpoints automatically:

```bash
uv run --no-sync python src/unilab/scripts/compare_leap_hora_selections.py \
  --exact runs/hora_stage1/EXACT_RUN/checkpoint_sweep/selection.json \
  --control runs/hora_stage1/zero_seed1/checkpoint_sweep/selection.json \
  --output runs/hora_stage1/exact_vs_zero_seed1.json --require-pass
```

Stage 1 has earned further randomization only if this exits successfully. The
default gate requires exact to improve median net rotation by 0.5 rad or speed
by 0.025 rad/s without losing more than 5% horizon completion. Repeat the
comparison across seeds; thresholds are CLI options, so changes remain explicit.

## Domain-randomization ladder

Introduce exactly one tier at a time and start each tier from the selected
checkpoint of the previous tier:

1. cylinder geometry and grasp family;
2. mass/inertia consistent with density;
3. friction and COM;
4. PD gains and encoder bias;
5. latency, action noise, and observation noise;
6. small disturbances.

Create a named task-owner YAML for each durable tier instead of accumulating a
long undocumented override string. Train exact and zero with matched owners,
then run the same `stage final` sweep. If a tier fails, narrow only the newly
introduced range or extend that tier; do not widen another axis.

The current branch does **not** yet contain multi-geometry middle-contact caches
or named randomization-tier owners. Treat those as implementation work, not as
an available experiment.

## Stage 2: proprioceptive adaptation

Stage 2 is not implemented in this branch. `HoraActor` accepts predicted
extrinsics, but there is currently no trajectory recorder, temporal adaptation
network, dataset trainer, iterative recollection loop, or predicted-latent
evaluation owner. Do not claim Stage 2 results or use placeholder commands.

Implementation must occur in this order:

1. freeze the selected exact Stage 1 actor and encoder;
2. record hardware-only joint/action histories plus target encoder latents;
3. train a temporal convolution or GRU predictor;
4. add predicted-extrinsics rollout inference;
5. recollect on-policy data and retrain;
6. evaluate exact, predicted, zero, shuffled, and once-frozen latents with the
   same manifest and checkpoint selector;
7. require predicted performance to reach at least 85% of exact performance.

Do not begin this implementation until exact Stage 1 clearly beats zero.

## Hardware gate

Hardware is blocked until the final exact and predicted Stage 2 policies pass
all simulation gates across at least five seeds. Before enabling commands:

- verify joint order, signs, zero offsets, units, and limits;
- reproduce measured command timing and PD semantics in simulation;
- enforce external joint/velocity/torque envelopes;
- return to a safe pose immediately after support loss;
- start with a tethered lightweight cylinder and low action scale;
- log joints, commands, timestamps, safety interventions, and video.

## Reading a failed sweep

Open `checkpoint_sweep/selection.md`. If every row fails:

- survival fails: return to the previous stage or improve grasp initialization;
- middle contact fails: repair cache composition before adding contact rewards;
- rotation/speed fails but survival passes: adjust the rotation curriculum,
  exploration, or action structure;
- reversals fail: strengthen direction commitment, not raw velocity reward;
- palm/joint-limit gates fail: treat the checkpoint as invalid;
- only OOD fails: do not advance randomization; consolidate the current tier.

Every experiment decision should be recoverable from configs, checkpoints,
evaluation JSON, and `selection.json` without relying on memory or video.
