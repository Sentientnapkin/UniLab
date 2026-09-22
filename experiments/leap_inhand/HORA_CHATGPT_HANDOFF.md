# HORA-Style LEAP Hand: Complete ChatGPT Handoff

Last updated: **2026-09-21 15:25 PDT**

This document is a self-contained handoff for continuing the HORA-style LEAP
Hand project in a normal ChatGPT conversation that cannot inspect the local
repository, run commands, or see old Codex context. It records the design,
implemented code, experiment history, current live state, commands, decision
rules, and unfinished work.

## Paste this into a new ChatGPT conversation

Upload or paste this entire file, then use this opening prompt:

> You are helping me continue a HORA-style in-hand rotation project for the
> 16-DoF LEAP Hand in UniLab. Treat the attached handoff as the source of truth.
> You do not have direct access to my filesystem or terminal, so do not claim
> that you inspected a file or run. Ask me to paste the relevant config,
> `selection.md`, evaluation JSON, training output, or visual observation when
> needed. Help me reason about one controlled experiment at a time, give exact
> commands using the paths in the handoff, and update your recommendation from
> measured survival, rotation, contact, and reward-component evidence. Preserve
> the gait-assisted task as a baseline. Do not start Stage 2, broad domain
> randomization, or hardware deployment until the documented gates pass.

After every new run, provide ChatGPT with:

1. `checkpoint_sweep/selection.md` in full;
2. the relevant part of `run_config.json` if command-line overrides were used;
3. the exact starting checkpoint and training command;
4. what the interactive viewer showed, especially which fingers moved, when
   contact was lost, whether rotation was sustained, and how the object fell;
5. any TensorBoard reward-component trends if the sweep result is ambiguous.

Do not rely on conversational memory alone. Record every durable decision in a
repo Markdown file or named YAML profile.

## Executive status

The project has a working Stage 1 architecture, a geometry-aware validated
grasp pipeline, a contact-enriched Stage A cache, deterministic checkpoint
sweeps, and a sequence of stabilization/rotation experiments. The learned
policy can preserve useful grasps and can learn some signed rotation, but it has
**not** learned sustained cyclic regrasping and remains far below the project
rotation gates.

The most important findings are:

- The original grasp cache was structurally correct but did not guarantee
  long-horizon passive retention or middle-finger contact.
- The v5 weighted cache fixed the contact topology problem. Stage A checkpoint
  350 is a strong stabilization starting point.
- Increasing the axial-rotation weight eventually taught motion, but the policy
  exploited it by spinning briefly and dropping the object.
- A full PPO resume into the literal-HORA reward failed because it carried a
  critic, optimizer, adaptive learning rate, iteration counter, and value
  function trained for the old objective.
- An **actor-only B8 restart from Stage A checkpoint 350** is now running. It
  keeps the useful actor and encoder but resets the critic and optimizer.

At the timestamp above, this live run existed:

```text
runs/hora_stage1/rotate_b8_restart_v5_seed1
```

It had produced checkpoints `model_0.pt` through at least `model_120.pt`. The
planned budget is 200 iterations and no deterministic sweep had yet been run.
Do not restart or overwrite this directory. Wait for completion, then sweep it.

Stage 2 adaptation is **not implemented**. Hardware deployment is blocked.

## Repository and Git safety

HORA worktree:

```text
/Users/School/Research/UniLab-hora
```

Branch:

```text
feat/leap-hora
```

Remotes:

```text
origin   https://github.com/Sentientnapkin/UniLab.git
upstream https://github.com/Motphys/UniLab.git
```

`origin` is the user's fork. `upstream` is the main UniLab repository. Do not
push or commit this experimental work to upstream or an upstream main branch.
The original checkout remains separate so another task can work there in
parallel. This worktree exists specifically to isolate HORA work.

Recent committed foundation:

```text
7ea1f60e feat(leap): add HORA stage-one experiment gate
34dda35f data(leap): add validated HORA grasp cache
ae9b04ae feat(leap): validate HORA grasp physics
a1d798ab feat(leap): add bounded HORA grasp collector
71fe7f3f feat(leap): add HORA precision grasp cache v2
d308a067 feat(leap): add HORA stage-one policy skeleton
a150df3c feat(leap): add gait rotation and grasp cache workflows
```

The worktree also contains substantial **uncommitted** HORA experiment work,
including the v5 caches, B2-B8 profiles, checkpoint sweep tools, actor-only
resume support, tests, and documentation. Never use `git reset --hard`, `git
clean`, or `git checkout --` on this worktree. Inspect `git status --short`
before any commit and separate generated data, code, and experiment configs
carefully.

Useful safety check:

```bash
cd /Users/School/Research/UniLab-hora
git branch --show-current
git status --short
git remote -v
```

Expected branch is `feat/leap-hora`.

## Project goal

Build a HORA-inspired in-hand cylinder rotation system for the 16-DoF LEAP
Hand while preserving UniLab's existing gait-assisted policy as a working
baseline. This is not intended to be a line-for-line Allegro-to-LEAP port. The
target is HORA's learning structure:

1. Train a privileged Stage 1 expert from stable precision grasps. A learned
   encoder maps simulator-only object/dynamics properties to a compact latent.
2. Freeze the Stage 1 actor and privileged encoder.
3. Train a Stage 2 temporal adaptation module that predicts that latent using
   only proprioception and command history available on real hardware.

UniLab continues to own the LEAP asset, MuJoCo backend, manager-based
environment, and RSL-RL integration.

Primary references:

- Paper: <https://arxiv.org/abs/2210.04887>
- Project: <https://haozhi.io/hora/>
- Authors' implementation: <https://github.com/HaozhiQi/Hora>

## Project success gates

Before hardware deployment, require held-out simulation evaluation over at
least five seeds:

- median time-to-fall at least 16 seconds, which is 80% of the 20-second
  horizon;
- median net committed-axis rotation at least 6 radians per episode;
- median directional speed at least 0.25 rad/s, with 0.5 rad/s as a stretch
  target;
- no palm-support exploitation;
- low reversal frequency after direction commitment;
- no finger persistently fully extended or pinned at a joint limit;
- Stage 2 predicted-extrinsics performance at least 85% of Stage 1 exact
  performance;
- separate in-distribution, edge-of-range, and out-of-distribution reporting.

These are project gates, not performance claims from the HORA paper.

## Existing gait-assisted baseline

The existing UniLab task uses:

- 16-DoF LEAP Hand and a cylindrical object;
- 20 Hz policy control and 400-step/20-second episodes;
- a recurrent actor with deployable joint/target/phase observations;
- an explicit cyclic gait reference plus learned residual actions;
- contact-aware drop/grasp logic and bidirectional accumulated-rotation reward;
- domain randomization over mass, friction, COM, PD gains, encoder bias, delay,
  and action noise.

The historically best working gait family before cache qualification was:

```text
/Users/School/Research/leap_first_steps/runs/grasp_diverse_cache_ft_s1/model_4375.pt
```

Do not mutate the gait task into HORA. It is a separate baseline. The old
episode-anchor experiment is also separate; changing a mature policy's action
anchor invalidated its learned semantics.

## What the HORA paper contributes

HORA's Stage 1 actor receives a short history of joint positions, a short
history of commanded actions, and an 8-dimensional latent produced from
privileged object properties. The privileged encoder and actor train jointly
with PPO. Stage 2 freezes them and predicts the latent from a longer
proprioception/action history.

Important paper details used here:

- 20 Hz policy and 400-step episodes;
- three recent joint/action frames for the base actor;
- an 8D latent encoded from a 9D privileged property vector in the paper;
- clipped target-axis angular velocity in `[-0.5, 0.5]`;
- reward coefficients approximately: rotation `+1`, initial-pose deviation
  `-0.3`, object linear velocity `-0.3`, torque `-0.1`, work `-2`;
- no prescribed finger gait and no reward requiring three fingertips to remain
  in contact;
- termination when the object falls rather than a necessary explicit drop
  penalty.

The clipping discourages a max-speed instability. Pose restoration helps a
finger break and re-establish contact instead of monotonically deforming the
hand. Linear-velocity, torque, and work costs bias the policy toward a smooth,
stable emergent gait.

Do not copy Allegro poses, gains, or exact numeric balance blindly. LEAP has
different kinematics, thumb reach, geometry, limits, and actuation. The paper
also used far more parallel simulation and grasp data. The public HORA repo
notes that reward numbers can differ between code versions.

## Implemented Stage 1 architecture

Task name:

```text
leap_inhand_hora_stage1
```

### Action semantics

`LeapHoraPositionAction` implements a pure accumulated target at 20 Hz:

```text
effective_action[t] = delay_and_noise(policy_action[t])
target[t + 1] = clip(target[t] + effective_action[t] / 24, joint_limits)
```

Key properties:

- 16 position-target increments;
- action scale `1/24 = 0.0416667` radians per policy action unit per step;
- each episode starts with target equal to the sampled grasp joint pose;
- a zero action keeps the previous target;
- no scripted cyclic gait;
- no leaky return toward the initial anchor;
- no small anchor-relative box that prevents accumulating a regrasp cycle;
- clipping only to physical actuator limits;
- low-level PD remains at simulator substep rate;
- latency is sampled once per episode and command history is reset cleanly.

This replaced earlier bounded/leaky semantics. Old checkpoints trained under
those semantics are incompatible historical artifacts and must not be resumed
into the pure-integrator task.

### Actor observations and network

The deployable actor portion uses:

- three recent measured joint-position frames;
- three recent commanded target/action frames;
- the learned extrinsics latent.

The current privileged property provider has 10 normalized fields:

1. cylinder radius;
2. cylinder length;
3. object mass;
4. sliding friction;
5. torsional friction;
6. COM offset x;
7. COM offset y;
8. COM offset z;
9. effective PD-gain scale;
10. latency.

`ExtrinsicsEncoder` is an MLP `10 -> 32 -> 32 -> 8` with ELU activations.
`HoraActor` consumes hardware history plus the 8D latent and currently uses
hidden sizes `512 -> 256 -> 128 -> 16` with ELU. The encoder is inside the actor
module, so actor-only checkpoint loading also transfers the encoder, actor
normalizer, distribution state, and policy weights.

The environment's flat observation contains the 10 raw properties, while the
checkpoint actor consumes the encoded 8D replacement. This produces the known
dimension distinction:

```text
checkpoint actor input = 104
flat environment observation = 106
```

Interactive playback must use `interactive.policy_obs_mode=actor` (or the CLI
equivalent) so it reconstructs the correct actor interface.

The critic has a separate privileged observation group. Privileged information
must not leak directly into the deployable hardware observation history.

### Exact and zero extrinsics

“Exact extrinsics” does not mean manually known physical parameters are fed
straight to the motor policy. During Stage 1, simulator-only properties are
normalized and encoded to the 8D latent; the actor receives that exact encoder
output. It is the privileged upper bound and the teacher target for Stage 2.

“Zero extrinsics” uses the same actor hardware observation and compatible
network width but gives it a constant zero latent. This is the matched robust
domain-randomization baseline. If exact does not outperform zero, the latent is
not proving useful adaptation.

Do not spend on a new zero-extrinsics run until the exact policy learns
meaningful sustained rotation. Existing old `zero_seed1` artifacts are not a
substitute for a matched final experiment.

## Reward implementation and interpretation

The core HORA-style reward components are:

- clipped signed angular velocity along the commanded hand-frame axis;
- squared deviation from the episode's initial hand pose;
- object linear-velocity penalty;
- estimated torque penalty;
- estimated mechanical-work penalty;
- geometric fall termination.

Contact state is primarily a failure/diagnostic signal, not a prescribed gait.
Current LEAP curriculum profiles sometimes use dense retained-grasp shaping,
but literal B8 disables it.

Implementation details matter:

- `hora_axial_rotation` returns clipped signed axial velocity and is not gated
  by valid grasp;
- inherited frontier-related parameters visible in the config are ignored by
  this simple HORA rotation function;
- the manager scales reward by `dt = 0.05`;
- torque is implemented as a sum of squared estimated torques;
- work is currently the square of summed `torque * joint_velocity`, so it is
  directionally HORA-inspired but not bit-identical to every paper/code version;
- the separate frontier reward supports valid-grasp-gated progress, direction
  locking, reversal/stall penalties, and angle milestones, but its B8 weight is
  zero.

To compare raw component magnitudes, use the logged weighted reward rate before
`dt` scaling and divide by its coefficient. For example, if weighted
`pose_diff = -0.03` and its weight is `-0.1`, the mean raw pose cost is about
`0.3`. This reveals whether a coefficient is numerically dominating rather
than merely looking large on paper. Always distinguish:

```text
raw statistic -> multiplied by configured weight -> multiplied by dt
```

The purpose is to diagnose objective balance. If rotation contributes `+0.02`
while pose contributes `-0.4`, exploration toward motion is strongly opposed.
If rotation rises while retention and survival collapse, the policy has found
a spin-and-drop shortcut rather than a gait.

## Grasp-cache pipeline

HORA assumes stable precision-grasp initialization. This proved decisive for
LEAP.

### Original 1,000-row pilot

- 1,000 validated grasps;
- three validation trials;
- family counts: balanced thumb/index/middle 210, distal outer opposition 166,
  high thumb opposition 124, index-forward load 411, outer-finger load 89;
- geometry counts: `(r=.028,l=.06)` 257, `(r=.032,l=.07)` 401,
  `(r=.036,l=.08)` 342.

### Schema-v2 10,000-row cache

- 10,000 validated grasps;
- train 7,112, validation 1,224, held-out 1,664;
- families: balanced 1,914, distal outer 1,664, high thumb 1,224,
  index-forward 4,198, outer-finger 1,000;
- geometries: small 2,612, medium 3,856, large 3,532;
- validation trials: three.

The cache was finite, inside limits, and correctly applied on reset. However,
short validation did not imply 20-second passive retention. A nominal 64-env
sample retained only 33/64 for the full horizon. The fixed 15-case audit had
5/5 nominal, 3/5 edge, and 3/5 OOD horizon completions; all four failures were
vertical escapes.

### Passive qualification tiers

Zero-command rollouts classify rows as:

- anchor: at least 16 seconds;
- recoverable: at least 1 second but less than 16;
- marginal: at least 0.5 second but less than 1;
- invalid: less than 0.5 second.

### Active v5 Stage A cache

Path:

```text
src/unilab/assets/caches/hora/leap_precision_grasps_v5_stage_a_weighted.npz
```

Composition:

- 2,000 rows;
- 1,996 unique source rows, only 0.2% duplication;
- 1,600 anchors and 400 recoverable rows;
- family counts: balanced 400, index-forward 560, outer-finger 240,
  middle-opposition expansion 800;
- all 800 middle-enriched rows are unique.

This cache is the current Stage A source and was the major contact-topology
improvement. The middle expansion currently covers only the medium cylinder
`r=.032 m, l=.070 m`.

Reproduction scripts:

```bash
cd /Users/School/Research/UniLab-hora

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

Do not rebuild the cache casually while comparing runs. A cache change alters
the experimental distribution.

## Deterministic evaluation system

Manifest:

```text
experiments/leap_inhand/evaluation_manifest_v1.json
```

It contains 15 deterministic cases: five in-distribution, five edge-of-range,
and five OOD. This is a useful development screen, not adequate final
statistics. Final claims require at least five policy seeds and larger
evaluation samples.

Primary sweep script:

```text
src/unilab/scripts/sweep_leap_hora_checkpoints.py
```

It:

- discovers `model_*.pt` automatically when `--checkpoints` is omitted;
- resumes interrupted sweeps and reuses existing evaluations;
- evaluates each checkpoint against the same manifest;
- applies `stabilize`, `rotate`, or `final` gates;
- writes `evaluation_N.json`, `evaluation_N.log`, `selection.json`, and
  `selection.md` under `<run>/checkpoint_sweep/`.

Use `--force` only when code, manifest, or evaluation semantics changed.

The safest sweep form omits explicit checkpoint indices:

```bash
uv run --no-sync python src/unilab/scripts/sweep_leap_hora_checkpoints.py \
  --run-dir RUN_DIRECTORY --profile PROFILE --stage rotate --require-pass
```

Why: full-resume runs preserve global iteration indices, while actor-only
restart runs begin from zero. Explicit ranges must name files that actually
exist. The range syntax is inclusive, e.g. `350:440:10`, and adding `,99` asks
for literal `model_99.pt`; it is not shorthand for “the last checkpoint.”

### Stabilization selector

Requires:

- median time-to-fall at least 16 seconds in every split;
- horizon completion at least 0.80;
- median valid-grasp fraction at least 0.70;
- median middle-finger contact at least 0.40;
- palm support and joint-limit occupancy no more than 0.02.

Passing checkpoints are ranked using safety and a balance of valid-grasp and
middle-contact fractions.

### Rotation selector

Requires survival first, including 16-second split medians, 0.80 horizon
completion, valid grasp at least 0.65, middle contact at least 0.25, and no
palm/joint-limit exploitation. It then ranks net rotation, directional speed,
valid grasp, middle contact, and reversals. A selector can recommend the least
bad safe checkpoint even if it is not materially better than an earlier run;
always compare absolute metrics and the prior global best.

### Final selector

Requires the project gates: 16-second split survival, 0.80 horizon completion,
6-radian median net rotation, 0.25 rad/s speed, at most 10 median reversals, and
no palm/joint-limit exploitation.

Metric meanings:

- **time to fall:** retention/safety, not rotation quality;
- **net committed-axis rotation:** signed useful progress after direction
  commitment; primary task metric;
- **accumulated rotation:** can be inflated by oscillation, so never use alone;
- **directional speed:** net rotation per elapsed time in committed direction;
- **reversals:** detects back-and-forth exploitation;
- **valid-grasp fraction:** fraction of rollout satisfying support/contact
  validity;
- **middle contact fraction:** important LEAP topology diagnostic, not itself a
  mandated paper reward;
- **palm support:** must remain near zero;
- **joint-limit occupancy:** detects pinned or fully extended fingers.

## Experiment history and lessons

### Early pure-integrator/reward experiments

The first pure-integrator reward used rotation `1`, object linear velocity
`-0.1`, pose `-0.005`, torque `-0.05`, work `-0.1`, and drop `-200`. Around
iteration 750 it found stability but little speed; later it traded stability
for faster motion rather than learning a sustainable regrasp cycle.

Reward v3 strengthened linear velocity to `-0.3`, pose to `-0.01`, torque to
`-0.1`, work to `-1`, and removed explicit drop cost. It found roughly
0.33 rad/s but all 52 screened episodes fell within 5.55 seconds and
middle-finger contact was about 3%: fast destructive motion, not a gait.

These results motivated explicit passive-grasp qualification and the v5 cache.

### Stage A: stabilization/recovery

Run:

```text
runs/hora_stage1/stabilize_v5_seed1
```

Live profile values:

```text
retained +1.0
rotate   +0.1
pose     -0.03
linvel   -0.3
torque   -0.1
work     -1.0
drop     -20
actor initial mean 0, initial std 0.03
```

Checkpoint 350 was selected:

- 13/15 horizon completions = 0.867;
- minimum split median time-to-fall = 20 seconds;
- median valid-grasp fraction = 0.8575;
- median middle-contact fraction = 0.685;
- palm and joint-limit gates passed.

Other useful checkpoints:

- 300: horizon 0.867, valid 0.786, middle 0.710;
- 399: horizon 0.867, valid 0.968, middle 0.453;
- 275: horizon 0.80, valid 0.790, middle 0.542.

Under the old cache, checkpoint 350's middle contact was only 0.0075. The new
cache's 0.685 is strong evidence that initialization/contact distribution was
the bottleneck.

Global stable starting point remains:

```text
runs/hora_stage1/stabilize_v5_seed1/model_350.pt
```

### B2 through B6: cautious rotation curriculum

Current live profile inheritance is:

| Profile | Rotate | Retained | Pose | Linvel | Torque | Work | Drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stabilize | 0.1 | 1.0 | -0.03 | -0.3 | -0.1 | -1 | -20 |
| B2 | 1.0 | 0.5 | -0.03 | -0.3 | -0.1 | -1 | -20 |
| B3 | 1.0 | 0.5 | -0.3 | -0.3 | -0.1 | -1 | -20 |
| B4 | 1.0 | 0.5 | -0.1 | -0.3 | -0.1 | -1 | -20 |
| B5 | 2.0 | 0.5 | -0.1 | -0.3 | -0.1 | -1 | -20 |
| B6 | 2.0 | 0.1 | -0.15 | -0.3 | -0.1 | -1 | -20 |

Some older prose in `HORA_STAGE1.md` says B2 rotation was `2.0`; the YAML and
saved `run_config.json` are authoritative for actual runs. Always inspect a
run's saved config when there is a discrepancy.

B3 run:

```text
runs/hora_stage1/rotate_b3_v5_seed1
```

Checkpoint 499 was the first clear contact/direction improvement:

- horizon 12/15 = 0.80;
- median net rotation 0.1064 rad;
- directional speed 0.00592 rad/s;
- valid grasp 0.83;
- middle contact 0.3925;
- median reversals 8;
- no palm/joint-limit exploitation.

It was substantially better than B3 checkpoint 400, which had negative net
rotation, but remained far below a useful gait.

B4 run started from B3 checkpoint 499. Checkpoint 535 improved holding but lost
rotation:

- net 0.0753 rad;
- speed 0.00426 rad/s;
- horizon 13/15;
- valid 0.965;
- middle 0.295;
- reversals 4.

B5 doubled rotation weight. Checkpoint 545 remained similar:

- net 0.0717 rad;
- speed 0.00382 rad/s;
- horizon 13/15;
- valid 0.98;
- middle 0.297.

Later B5 checkpoint 580 reached net 0.101 rad but middle contact collapsed to
0.0075 and it failed. More rotation weight alone did not create a gait.

B6 reduced retained shaping and used pose `-0.15`. Checkpoint 594 was the best
overall balance found before the B8 restart:

- net 0.1363 rad;
- speed 0.00764 rad/s;
- horizon 12/15 = 0.80;
- valid 0.932;
- middle 0.385;
- reversals 3.

Run path:

```text
runs/hora_stage1/rotate_b6_v5_seed1/model_594.pt
```

A further 50-iteration continuation selected checkpoint 643 by its local
ranking, but it failed the survival gate: horizon 11/15, valid 0.703, net
0.0914. Do not treat 643 as better than 594 merely because it was selected
inside the failed continuation.

### B7: high-rotation failure

B7 started from B6 checkpoint 594 with:

```text
rotate +3.0, retained +0.1, pose -0.1
```

It clearly learned increasing motion but increasingly dropped the object:

| checkpoint | median net rotation | horizon fraction |
| ---: | ---: | ---: |
| 600 | 0.074 | 0.667 |
| 640 | 0.148 | 0.667 |
| 660 | 0.177 | 0.400 |
| 680 | 0.274 | 0.200 |
| 700 | 0.330 | 0.267 |
| 720 | 0.269 | 0.400 |
| 740 | 0.308 | 0.333 |
| 760 | 0.439 | 0.133 |
| 780 | 0.316 | 0.067 |
| 793 | 0.281 | 0.133 |

All checkpoints failed. Interactive viewing confirmed that the policy learned
rotation but not the contact-release/re-establishment cycle; it spun and
dropped. Reward logs showed rotation increasing while retention fell, pose/work
costs grew, and the grasp deteriorated. This is why further rotation-weight
increases are not recommended.

### B8 full-resume failure

B8 uses the paper-inspired compact balance:

```text
rotate +1.0
pose -0.3
linvel -0.3
torque -0.1
work -2.0
retained 0
frontier 0
drop 0
```

The first B8 experiment fully resumed B6 checkpoint 594. Every checkpoint
failed. Horizon completion stayed roughly 0.47-0.67, minimum split median
time-to-fall fell to about 2.35-5.1 seconds, and net rotation remained small.
Training reward deteriorated while pose deviation and invalid grasps grew.

This was not interpreted as “the paper reward just needs more time.” A full
resume reused an incompatible value function, optimizer state, adaptive
learning rate near `1e-5`, iteration count, and logger state from the old
objective. That contaminated the experiment.

### B8 actor-only restart

The training stack now supports:

```yaml
training:
  resume_mode: actor_only
```

Actor-only loading transfers the Stage A actor and encoder but creates a fresh
critic and optimizer, resets learning progress to iteration zero, and uses the
fresh PPO learning-rate schedule. Focused loader/profile tests passed.

The current experiment starts from Stage A checkpoint 350 rather than B6 594,
because Stage A is the strongest neutral grasp-maintenance prior and is less
specialized to the old shaping objective.

## Immediate next action

Do **not** relaunch training if the B8 restart is still running. Confirm that
the terminal completed and that `model_199.pt` or the final expected checkpoint
exists. Then run:

```bash
cd /Users/School/Research/UniLab-hora

uv run --no-sync python src/unilab/scripts/sweep_leap_hora_checkpoints.py \
  --run-dir runs/hora_stage1/rotate_b8_restart_v5_seed1 \
  --profile rotate_b8_restart --stage rotate --require-pass
```

The report will be written to:

```text
/Users/School/Research/UniLab-hora/runs/hora_stage1/rotate_b8_restart_v5_seed1/checkpoint_sweep/selection.md
```

Paste that full file into the new ChatGPT conversation. Compare it against:

- Stage A 350 for survival/contact;
- B3 499 for the first safe positive rotation;
- B6 594 for the current best balance;
- B7 for the known spin-and-drop failure signature.

Do not accept “passes rotate selector” as sufficient. Look for a trajectory
where positive net rotation and speed rise over checkpoints while horizon,
valid grasp, and middle contact remain intact.

If the 200-iteration restart is still trending safely upward but remains below
the project final gates, a bounded continuation may be justified. If survival
steadily collapses as rotation rises, extending it is unlikely to create the
missing gait.

## Interactive playback

On macOS, MuJoCo passive viewing must run under `mjpython`; ordinary `python`
raises:

```text
RuntimeError: launch_passive requires that the Python script be run under
mjpython on macOS
```

For B8 restart checkpoint `N`:

```bash
cd /Users/School/Research/UniLab-hora

uv run --no-sync mjpython src/unilab/scripts/play_interactive.py \
  --algo ppo --task leap_inhand_hora_stage1 --sim mujoco_rotate_b8_restart \
  algo.load_run=/Users/School/Research/UniLab-hora/runs/hora_stage1/rotate_b8_restart_v5_seed1 \
  algo.checkpoint=N training.device=mps \
  interactive.action_mode=policy \
  interactive.policy_obs_mode=actor \
  interactive.show_reward_debug=true \
  interactive.reward_debug_show_velocity=true
```

Controls are printed by the script and include Space to pause, `N` to
single-step, `+/-` for speed, Backspace to reset, and Escape to quit.

Use video only as a diagnostic after deterministic metrics. Specifically note:

- whether thumb and middle contacts alternate or simply disappear;
- whether one outer finger remains fully extended;
- whether the hand makes repeatable contact-release-recontact cycles;
- whether the object translates vertically before falling;
- whether apparent rotation is a single transient impulse;
- whether recovery occurs after small slips;
- whether motion is in the commanded sign with few reversals.

## Decision tree after B8 restart

### Case 1: survival and contacts pass, rotation improves coherently

Continue a bounded block with the same objective before changing coefficients.
Sweep densely and stop if valid grasp or horizon completion declines. The
policy needs evidence of a learning trend, not merely a visually pleasing
episode.

### Case 2: safe but nearly static

Inspect raw reward-component magnitudes and action standard deviation. Determine
whether pose/work dominates rotation or exploration is too small. Change one
axis only. Possible tests include a modest exploration adjustment or a
carefully bounded bridge curriculum. Do not immediately return to rotation
weight 3, which already produced the drop shortcut.

### Case 3: rotation rises while survival/contact falls

Reject it as the B7 failure mode. More time or more rotation weight is not the
first response. Inspect whether the simple velocity reward provides no signal
for valid-grasp progress. The next experiment may be an explicitly labeled
**hybrid frontier curriculum**, not a claim of literal HORA reward:

- start from the best stable checkpoint;
- use a modest simple rotation term;
- add valid-grasp-gated frontier progress/direction commitment;
- retain strong pose/work regularization;
- use only a small retention term if necessary;
- return to compact HORA reward after a cyclic behavior exists, if possible.

This is a deliberate LEAP-specific scaffold. Record it as a hybrid baseline.

### Case 4: immediate catastrophic regression

Verify actor-only loading actually occurred and the console said:

```text
Loading actor_only checkpoint from ...model_350.pt
```

Inspect `run_config.json` for `resume_mode: actor_only`, the exact starting
checkpoint, reward values, action semantics, and cache path. Do not assume the
command used the intended profile.

## Zero-extrinsics and multi-seed plan

Only after exact extrinsics shows meaningful sustained rotation:

1. freeze the exact configuration and budget;
2. train the matched zero-latent policy with the same cache, seed, randomization,
   reward, iteration count, and checkpoint frequency;
3. compare identical manifests;
4. require exact to improve net rotation by at least 0.5 rad or speed by at
   least 0.025 rad/s without losing more than 5% horizon completion;
5. repeat exact and zero on seeds 2 and 3 initially;
6. ultimately report at least five seeds.

Comparison tool:

```bash
uv run --no-sync python src/unilab/scripts/compare_leap_hora_selections.py \
  --exact runs/hora_stage1/EXACT_RUN/checkpoint_sweep/selection.json \
  --control runs/hora_stage1/ZERO_RUN/checkpoint_sweep/selection.json \
  --output runs/hora_stage1/exact_vs_zero_seed1.json --require-pass
```

Do not claim that adaptation helps merely because exact works. It must beat the
matched no-extrinsics control.

## Domain-randomization ladder

Current learning is primarily the nominal medium-cylinder tier. Do not widen
several axes at once. After nominal exact-versus-zero succeeds, introduce one
tier per named YAML owner:

1. geometry and grasp family;
2. mass and inertia consistent with geometry/density;
3. friction and COM offset;
4. PD gains and encoder bias;
5. action delay/noise and observation noise;
6. small external disturbances.

Sample one coherent physical parameter set per episode. Preserve a fixed-domain
anchor population. Match exact and zero treatments at every tier. The branch
does not yet contain multi-geometry middle-contact caches or durable owners for
all these tiers.

## Stage 2 adaptation: unimplemented plan

The actor interface can accept predicted extrinsics, but the following still
need implementation:

1. freeze the selected Stage 1 actor and privileged encoder;
2. record rollout windows containing only measured joints and commanded
   targets/actions available on hardware;
3. record the privileged encoder output as the supervised target latent;
4. train a compact temporal convolution or GRU predictor;
5. run the frozen actor using predicted latents;
6. recollect data under the induced predicted-latent policy distribution;
7. retrain iteratively;
8. evaluate exact, predicted, zero, shuffled, once-estimated/frozen, and direct
   physical-parameter prediction ablations;
9. require predicted closed-loop rotation and survival to reach at least 85%
   of exact.

Command history is essential because commanded-versus-measured discrepancy is
what makes dynamics identifiable. Contact, object pose, simulator velocity,
mass, and friction cannot enter the deployable Stage 2 history unless
equivalent hardware sensing exists.

Do not start this work until exact Stage 1 clearly exceeds zero extrinsics.

## Hardware gate

Hardware use remains blocked. Before deployment:

- verify LEAP joint ordering, signs, zero offsets, limits, and units;
- measure command/update timing and reproduce it in simulation;
- match target semantics and PD gains;
- enforce external joint, velocity, and torque safety limits;
- detect loss of support and immediately command a safe pose;
- start with a tethered lightweight cylinder and low action scale;
- log measured joints, commands, timestamps, safety interventions, and video;
- expand geometry, mass, and duration only after repeated safe trials.

Domain randomization cannot replace calibration, timing measurement, or a
safety supervisor.

## Important files

Documentation:

```text
experiments/leap_inhand/HORA_CHATGPT_HANDOFF.md
experiments/leap_inhand/HORA_STAGE1.md
experiments/leap_inhand/HORA_SELF_GUIDED_WORKFLOW.md
experiments/leap_inhand/evaluation_manifest_v1.json
```

Task and terms:

```text
src/unilab/tasks/manipulation/leap_inhand/hora.py
src/unilab/tasks/manipulation/leap_inhand/hora_grasp.py
src/unilab/tasks/manipulation/leap_inhand/manager_terms.py
src/unilab/tasks/manipulation/leap_inhand/evaluation.py
src/unilab/tasks/manipulation/leap_inhand/passive_qualification.py
```

PPO/task configuration:

```text
src/unilab/conf/ppo/config.yaml
src/unilab/conf/ppo/task/leap_inhand_hora_stage1/base.yaml
src/unilab/conf/ppo/task/leap_inhand_hora_stage1/mujoco_nominal.yaml
src/unilab/conf/ppo/task/leap_inhand_hora_stage1/mujoco_no_extrinsics.yaml
src/unilab/conf/ppo/task/leap_inhand_hora_stage1/mujoco_stabilize.yaml
src/unilab/conf/ppo/task/leap_inhand_hora_stage1/mujoco_rotate_b2.yaml
src/unilab/conf/ppo/task/leap_inhand_hora_stage1/mujoco_rotate_b3.yaml
src/unilab/conf/ppo/task/leap_inhand_hora_stage1/mujoco_rotate_b4.yaml
src/unilab/conf/ppo/task/leap_inhand_hora_stage1/mujoco_rotate_b5.yaml
src/unilab/conf/ppo/task/leap_inhand_hora_stage1/mujoco_rotate_b6.yaml
src/unilab/conf/ppo/task/leap_inhand_hora_stage1/mujoco_rotate_b7.yaml
src/unilab/conf/ppo/task/leap_inhand_hora_stage1/mujoco_rotate_b8.yaml
src/unilab/conf/ppo/task/leap_inhand_hora_stage1/mujoco_rotate_b8_restart.yaml
```

Training/evaluation tools:

```text
src/unilab/scripts/train_rsl_rl.py
src/unilab/scripts/evaluate_leap_inhand.py
src/unilab/scripts/sweep_leap_hora_checkpoints.py
src/unilab/scripts/compare_leap_hora_selections.py
src/unilab/scripts/play_interactive.py
src/unilab/scripts/audit_leap_hora_rollout.py
src/unilab/scripts/qualify_leap_hora_grasps.py
src/unilab/scripts/build_leap_hora_v5_candidates.py
src/unilab/scripts/build_leap_hora_contact_cache.py
src/unilab/scripts/rebuild_leap_hora_stage_a.py
```

Tests:

```text
tests/tasks/test_leap_hora.py
tests/scripts/test_train_scripts.py
```

## Common command and tooling pitfalls

### `uv` warning

The installed `uv` may warn that this `pyproject.toml` field is unknown:

```text
exclude-dependencies = ["torchvision"]
```

This warning has been harmless for the current environment. `uv run --no-sync`
or direct `.venv/bin/...` tools avoid environment re-resolution. Do not edit
dependency policy solely to silence it during an experiment.

### Playback CLI

`play_interactive.py` requires explicit `--algo`, `--task`, and `--sim`. It does
not take a generic `--config` path as a replacement. For task variants,
`--sim mujoco_rotate_b8_restart` selects the owner. Hydra-style values such as
`algo.load_run=...` follow the fixed flags.

### Checkpoint indexing

- Full resume preserves the old global counter.
- Actor-only resume starts at iteration zero.
- A 100-iteration run commonly ends at `model_99.pt` only when the save logic
  writes that final checkpoint; never assume it exists.
- Omit `--checkpoints` in sweep commands unless there is a reason to specify an
  exact subset.

### Evaluation profile must match training

Using `nominal` to evaluate a zero-extrinsics checkpoint can reconstruct the
wrong actor interface. Always use the profile that created the run and inspect
`run_config.json`.

### Interactive success is not checkpoint selection

One rollout can look excellent by chance. Always select from the deterministic
manifest first, then use the viewer to explain the measured behavior.

## Tests before code changes are trusted

Fast task tests:

```bash
cd /Users/School/Research/UniLab-hora
uv run --no-sync pytest -q tests/tasks/test_leap_hora.py -m 'not slow'
```

When modifying resume/loading logic, also run the focused slow loader tests in
`tests/scripts/test_train_scripts.py`. When changing actions, reward terms,
observations, reset/cache logic, or evaluation semantics, run the directly
related tests and force fresh sweeps because old evaluation JSON may no longer
be comparable.

## Experiment discipline

1. Never overwrite a run directory.
2. Keep `run_config.json`, checkpoints, TensorBoard events, evaluator logs,
   evaluation JSON, and selections together.
3. Change one scientific variable per block whenever possible.
4. Record the starting checkpoint and whether loading was full or actor-only.
5. Use the same manifest for comparisons.
6. Compare against the previous global best, not only the immediately preceding
   failed run.
7. Reject palm support, joint-limit pinning, high reversal oscillation, and
   spin-then-drop shortcuts even if accumulated rotation is high.
8. Do not widen object diversity until stable-grasp initialization works at the
   current tier.
9. Do not add a scripted gait to rescue HORA without labeling it a separate
   hybrid baseline.
10. Do not begin Stage 2 until exact Stage 1 beats zero extrinsics.
11. Do not deploy any actor/adaptation observation that contains simulator-only
    state.

## Questions the next ChatGPT should ask after receiving new results

When the B8 restart sweep is pasted, answer these in order:

1. Did any checkpoint pass every rotation-stage survival/contact gate?
2. Which checkpoint has the best net rotation and speed among passing rows?
3. Are rotation and speed improving over training, or is selection a noisy
   isolated peak?
4. Does horizon completion, minimum split time-to-fall, valid-grasp fraction,
   or middle contact decline as rotation rises?
5. Are reversals increasing?
6. Does the result materially beat B6 checkpoint 594, not merely its own nearby
   checkpoints?
7. What does the viewer show at the best safe checkpoint and at the fastest
   failing checkpoint?
8. Do raw pose/work/rotation magnitudes explain the tradeoff?
9. Is a same-objective continuation justified, or is an explicitly labeled
   hybrid frontier curriculum the next controlled test?

The desired outcome is not “more motion.” It is repeatable, commanded-direction
rotation while continuously repairing the grasp over the full episode.
