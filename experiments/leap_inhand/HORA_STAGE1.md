# HORA LEAP Stage 1

Stage 1 trains the no-gait LEAP policy and privileged encoder jointly. The
first gate deliberately uses only the medium cylinder and nominal physics so a
failure to learn is attributable to the policy, action, or reward contract—not
to simultaneous domain widening.

## Experiment owners

`mujoco_nominal` is the exact-extrinsics treatment. `mujoco_no_extrinsics` is
the matched control: its actor has the same hardware observation and network
width, but receives a constant zero latent that is independent of privileged
properties. Both owners use training-split rows from the validated cache and
compile only the 32 mm radius, 70 mm long cylinder.

The declared object-mass normalization range is 0.020--0.120 kg, covering the
0.022--0.114 kg range exercised while validating the grasp cache. Tier 0 itself
uses nominal density, friction, COM, gains, latency, and observation noise.

## Bounded smoke runs

Run these before a long experiment. They verify PPO updates, checkpoint writes,
and the matched no-extrinsics control without claiming policy performance.

```bash
uv run train --algo ppo --task leap_inhand_hora_stage1 --sim mujoco \
  --profile nominal \
  algo.num_envs=8 algo.num_steps_per_env=8 algo.max_iterations=2 \
  algo.save_interval=1 algo.seed=1 training.no_play=true training.logger=tensorboard

uv run train --algo ppo --task leap_inhand_hora_stage1 --sim mujoco \
  --profile no_extrinsics \
  algo.num_envs=8 algo.num_steps_per_env=8 algo.max_iterations=2 \
  algo.save_interval=1 algo.seed=1 training.no_play=true training.logger=tensorboard
```

## First seed comparison

Do not select a checkpoint from an interactive rollout. Run the exact and
no-extrinsics owners with the same seeds and evaluate their checkpoints against
the fixed evaluation manifest. Start with seeds 1, 2, and 3; keep every other
override identical. The exact-extrinsics treatment must clearly outperform the
control before Stage 2 data collection begins.

```bash
uv run train --algo ppo --task leap_inhand_hora_stage1 --sim mujoco \
  --profile nominal algo.seed=1 training.no_play=true

uv run train --algo ppo --task leap_inhand_hora_stage1 --sim mujoco \
  --profile no_extrinsics algo.seed=1 training.no_play=true
```

Repeat both commands with `algo.seed=2` and `algo.seed=3`. Increase geometry,
mass, friction/COM, actuation/sensing, and disturbance diversity one tier at a
time only after the nominal task learns reliably across seeds.

## Deterministic evaluation

Evaluate exact and control checkpoints with the same versioned manifest. The
control command must retain `--profile no_extrinsics`; otherwise it would
reconstruct an exact-extrinsics actor around the control checkpoint.

```bash
uv run python -m unilab.scripts.evaluate_leap_inhand \
  --manifest experiments/leap_inhand/evaluation_manifest_v1.json \
  --checkpoint /absolute/path/to/exact/model_N.pt \
  --output /absolute/path/to/exact/evaluation.json \
  --policy-family hora --profile nominal

uv run python -m unilab.scripts.evaluate_leap_inhand \
  --manifest experiments/leap_inhand/evaluation_manifest_v1.json \
  --checkpoint /absolute/path/to/control/model_N.pt \
  --output /absolute/path/to/control/evaluation.json \
  --policy-family hora --profile no_extrinsics
```
