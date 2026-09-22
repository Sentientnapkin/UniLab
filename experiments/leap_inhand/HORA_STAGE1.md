# HORA LEAP Stage 1

Stage 1 trains the no-gait LEAP policy and privileged encoder jointly. The
first gate deliberately uses only the medium cylinder and nominal physics so a
failure to learn is attributable to the policy, action, or reward contract—not
to simultaneous domain widening.

## Pure accumulated-target contract

The production HORA action is a 20 Hz joint-target increment:

```text
effective_action[t] = delayed_and_noised(policy_action[t])
target[t + 1] = clip(target[t] + effective_action[t] / 24, actuator_limits)
```

Each episode starts with `target` equal to the sampled precision-grasp joint
pose. Zero actions preserve the preceding target. There is no scripted gait,
return-to-grasp factor, or anchor-relative target bound. A latency is sampled
once per episode, clears to a zero-filled command history at reset, and is
reported to the privileged-property provider in seconds.

Checkpoints trained before this contract was introduced used incompatible
bounded or leaky action semantics. Do not resume or evaluate them with the pure
owner. They remain historical artifacts only.

The first pure-integrator experiment used reward version 2: rotate `1.0`,
object linear velocity `-0.1`, pose deviation `-0.005`, estimated torque
`-0.05`, work `-0.10`, and terminal drop `-200.0`. Its checkpoint sweep found
a stable but slow policy near iteration 750, followed by a stability-for-speed
tradeoff rather than a sustainable regrasp cycle.

Reward version 3 keeps rotation at `1.0`, strengthens object linear velocity to
`-0.3`, pose deviation to `-0.01`, torque to `-0.1`, and work to `-1.0`, and
removes the explicit drop cost. A drop still terminates the episode and forfeits
future rotation reward. The direction matches HORA, but pose and work weights
are calibrated for the reward magnitudes observed with LEAP and MuJoCo rather
than copied blindly from Allegro/Isaac Gym.

The v2 TensorBoard audit below reports mean weighted reward rates before `dt`
scaling. Raw values are recovered by dividing each entry by its v2 weight.

| checkpoint | rotate | object linvel | pose deviation | torque | work | drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 750 | 0.02072 | -0.00119 | -0.00723 | -0.00104 | -0.00008 | -0.09766 |
| 900 | 0.01751 | -0.00093 | -0.00722 | -0.00086 | -0.00005 | 0.00000 |
| 1199 | 0.05437 | -0.00226 | -0.00299 | -0.00652 | -0.00178 | -0.19531 |

At checkpoint 750, applying HORA's literal `-0.3` pose coefficient would turn
the raw pose value of about `1.446` into a `-0.434` contribution against only
`+0.021` rotation. The v3 pose coefficient therefore increases only to
`-0.01`. Work increases more strongly because its raw value rises from about
`0.00075` at checkpoint 750 to `0.01785` at checkpoint 1199, making it a more
selective signal for the late aggressive failure mode.

The reward-v3 sweep showed the opposite failure mode from v2: all 52 screened
episodes dropped within 5.55 seconds, while median axial velocity reached about
`0.33` rad/s. Thumb contact remained high, but middle-finger contact fell to
roughly 3%, consistent with fast rotation through a degrading contact set rather
than a sustainable regrasp cycle. No v3 checkpoint qualifies for the full
15-case evaluation.

Reward version 4 is a bounded, literal-HORA reward pilot: rotate `1.0`, object
linear velocity `-0.3`, pose deviation `-0.3`, torque `-0.1`, work `-2.0`, and
drop `0.0`. This tests whether HORA's strong dense grasp and energy constraints
join v2's retention with v3's rotation. Train it from scratch for 400 iterations
and continue toward 1200 only if deterministic screening shows improving
survival together with positive rotation.

## Passive grasp audit

The schema-v2 cache is structurally sound: all 10,000 rows are finite and
inside actuator limits, the minimum normalized joint-limit margin is 8.36%, and
quaternion and cylinder-axis norm errors are below `6e-8`. The nominal Stage 1
owner has 2,736 eligible medium-cylinder rows from training families.

Reset application is exact. In a 64-environment nominal sample, the maximum
difference between the selected cache row and each of the action anchor,
observed initial joints, object position, and object quaternion was zero.
However, only 33/64 sampled training poses survived a fixed-target 20-second
rollout. The median reached the horizon, but the earliest failure occurred at
0.15 seconds. Survival by family in this sample was 10/23 balanced thumb/index/
middle, 20/35 index-forward load, and 3/6 outer-finger load.

The versioned 15-case manifest passive audit is stored at
`runs/hora_stage1/passive_grasp_audit/evaluation_full.json`. All five nominal
cases reached 20 seconds; 3/5 edge and 3/5 out-of-distribution cases reached 20
seconds. All four failures were vertical escapes, with no lateral-escape or
absolute-height failures. These results show that the cache is applied
correctly and contains good poses, but its original three two-second validation
trials do not guarantee the long-horizon stable initialization assumed by HORA.

## Long-horizon grasp qualification and Stage A

Before the next policy run, classify every medium-cylinder training row under a
literal zero command. The qualifier assigns four passive-retention tiers:

- anchor: at least 16 seconds;
- recoverable: at least 1 second but less than 16 seconds;
- marginal: at least 0.5 seconds but less than 1 second;
- invalid: less than 0.5 seconds.

It preserves the schema-v2 geometry/family metadata, writes every tier and a
machine-readable score report, and creates a 3,000-row Stage A cache. The Stage
A cache is balanced across training families and contains 80% anchors and 20%
recoverable rows within each family.

```bash
.venv/bin/python -m unilab.scripts.qualify_leap_hora_grasps \
  --num-envs 64 --seed 42 --max-control-steps 30000 \
  --input src/unilab/assets/caches/hora/leap_precision_grasps_v2.npz \
  --output-prefix src/unilab/assets/caches/hora/leap_precision_grasps_v3 \
  --stage-a-size 3000 --stage-a-anchor-fraction 0.8
```

Do not start Stage A if any training family has no anchor or recoverable rows;
the qualifier fails closed in that case. Inspect the printed tier and family
counts before training.

The contact-aware v5 audit combines the proven v3 anchor/recoverable rows with
the middle-opposition expansion. Rebuild Stage A offline from the completed
score report so the 2,000 rows use capacity-aware family targets rather than
repeating the legacy families' small middle-contact pools:

```bash
uv run --no-sync python src/unilab/scripts/rebuild_leap_hora_stage_a.py
```

The `mujoco_stabilize` owner consumes
`leap_precision_grasps_v5_stage_a_weighted.npz`. Its actor mean is initialized
to exactly zero (a hold command), its initial standard deviation is 0.03, and
its reward prioritizes grasp retention and recovery while retaining a small
axial-rotation term. Run the 400-iteration pilot only after qualification and
the offline rebuild succeed:

```bash
uv run train --algo ppo --task leap_inhand_hora_stage1 --sim mujoco \
  --profile stabilize \
  algo.max_iterations=400 algo.save_interval=25 algo.seed=1 \
  training.log_dir=/Users/School/Research/UniLab-hora/runs/hora_stage1/stabilize_v5_seed1 \
  training.no_play=true training.logger=tensorboard
```

Advance from Stage A based on held-out retention and recovery performance, not
iteration count alone. Rotation remains a secondary signal until the policy
can sustain the anchor population and improve the recoverable population.

### Stage B2 signed-rotation transition

The first Stage B attempt (`rotate=0.5`, `retained=1.0`) preserved survival but
did not learn commanded-direction rotation. Across its fixed-manifest sweep,
median signed rotation remained negative while retention contributed roughly
40--100 times more reward per step than rotation. Do not continue a B1
checkpoint.

`mujoco_rotate_b2` restarts from the selected Stage A checkpoint 350, raises
signed axial rotation to `2.0`, and reduces dense retention to `0.1`. All
physical penalties and the `-20` terminal drop cost remain unchanged. Run only
100 iterations initially and save densely; stop unless signed rotation becomes
positive within 25--50 iterations.

```bash
.venv/bin/train --algo ppo --task leap_inhand_hora_stage1 --sim mujoco \
  --profile rotate_b2 \
  algo.load_run=/Users/School/Research/UniLab-hora/runs/hora_stage1/stabilize_seed1/model_350.pt \
  algo.max_iterations=100 algo.save_interval=10 algo.seed=1 \
  training.log_dir=/Users/School/Research/UniLab-hora/runs/hora_stage1/rotate_b2_seed1 \
  training.no_play=true training.logger=tensorboard
```

### Stage B3 cyclic-regrasp test

B2 checkpoint 390 is the selected directional checkpoint: 14/15 fixed-manifest
episodes reached the horizon and 13/15 had positive net rotation. Continuing
B2 increased pose deformation and reduced valid-grasp fraction instead of
forming a sustainable gait. `mujoco_rotate_b3` therefore keeps B2's rotation
and retention weights but restores HORA's literal `-0.3` initial-pose penalty.

```bash
.venv/bin/train --algo ppo --task leap_inhand_hora_stage1 --sim mujoco \
  --profile rotate_b3 \
  algo.load_run=/Users/School/Research/UniLab-hora/runs/hora_stage1/rotate_b2_seed1/model_390.pt \
  algo.max_iterations=100 algo.save_interval=10 algo.seed=1 \
  training.log_dir=/Users/School/Research/UniLab-hora/runs/hora_stage1/rotate_b3_seed1 \
  training.no_play=true training.logger=tensorboard
```

Select B3 only if it preserves positive net rotation while reducing pose
deviation and preventing the valid-grasp regression seen after B2 checkpoint
390. If speed remains flat, treat the next blocker as exploration or action
structure rather than increasing reward scale again.

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

## Literal-HORA reward-v4 pilot

Do not select a checkpoint from an interactive rollout. Train exact extrinsics
from scratch and screen fixed checkpoints before spending the matched control
budget. Never resume a bounded/leaky-action checkpoint into this owner.

```bash
uv run train --algo ppo --task leap_inhand_hora_stage1 --sim mujoco \
  --profile nominal \
  algo.max_iterations=400 algo.save_interval=25 algo.seed=1 \
  training.log_dir=/absolute/path/to/hora_reward_v4_exact_seed1 \
  training.no_play=true training.logger=tensorboard
```

After the exact treatment demonstrates multi-grasp rotation, train its matched
zero-extrinsics control from scratch:

```bash
uv run train --algo ppo --task leap_inhand_hora_stage1 --sim mujoco \
  --profile no_extrinsics \
  algo.max_iterations=1200 algo.save_interval=50 algo.seed=1 \
  training.log_dir=/absolute/path/to/hora_pure_integrator_zero_seed1 \
  training.no_play=true training.logger=tensorboard
```

Only after exact clearly exceeds zero should both treatments be repeated with
seeds 2 and 3. Increase geometry, mass, friction/COM, actuation/sensing, and
disturbance diversity one tier at a time only after the nominal task learns
reliably across seeds.

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
