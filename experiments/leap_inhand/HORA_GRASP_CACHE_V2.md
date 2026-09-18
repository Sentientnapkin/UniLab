# HORA LEAP precision-grasp cache v2

The Phase 1 collector is a zero-action environment loop. It does not construct
an actor, PPO runner, optimizer, checkpoint, or training log.

## Inspect the resolved collection plan

This command requires no MuJoCo installation and performs no simulation:

```bash
uv run python -m unilab.scripts.generate_leap_hora_grasps \
  --describe-only \
  --num-envs 63 \
  --target 10000 \
  --seed 42 \
  --max-control-steps 100000 \
  --output src/unilab/assets/caches/hora/leap_precision_grasps_v2.npz
```

## Run the full bounded collection

Install the repository's MuJoCo extra first. Then run:

```bash
uv run python -m unilab.scripts.generate_leap_hora_grasps \
  --num-envs 63 \
  --target 10000 \
  --seed 42 \
  --max-control-steps 100000 \
  --output src/unilab/assets/caches/hora/leap_precision_grasps_v2.npz
```

The control-step cap is mandatory. Reaching it is an error, but closing the
environment still writes a partial cache when at least one grasp was accepted.
The full target stops normally through `RunComplete` as soon as 10,000 diverse
grasps have been saved. The final JSON reports accepted rows, validation
rejections, near-duplicate rejections, and accepted counts by geometry, family,
and split.

## Validation contract

Each candidate is sampled once and replayed for three trials. Every trial must:

- reach the two-second timeout without geometric escape;
- retain thumb plus at least two opposing-finger contacts;
- avoid sustained palm support;
- avoid contact between any pair of finger subtrees;
- finish below the configured linear- and angular-speed limits; and
- remain outside the configured joint-limit margin.

Mass and the matching analytic solid-cylinder inertia are resampled from
density on every trial. Sliding friction is also resampled on every trial.
Candidates are diversity-filtered within their geometry using joint RMS,
object position, and cylinder-axis orientation; spin around the symmetry axis
is ignored.

Self-collision is read through six named MuJoCo subtree-contact sensors bound by
the public scene sensor interface. Task code does not inspect raw backend
contact buffers.

## Output contract

The output is a compressed NumPy `.npz` archive. `allow_pickle=False` is
sufficient to load every field.

| Field | Shape | Meaning |
| --- | --- | --- |
| `schema_version` | scalar | Integer `2` |
| `joint_pos` | `(N, 16)` | Settled LEAP joint pose, radians |
| `object_pos` | `(N, 3)` | Settled object world position, metres |
| `object_quat` | `(N, 4)` | Settled wxyz object quaternion |
| `cylinder_radius` | `(N,)` | Radius, metres |
| `cylinder_length` | `(N,)` | Full cylinder length, metres |
| `family_id` | `(N,)` | Index into `family_names` |
| `split` | `(N,)` | Index into `split_names` |
| `validation_trials` | `(N,)` | Successful trials required for the row |
| `mass_min`, `mass_max` | `(N,)` | Observed validation mass bounds, kg |
| `sliding_friction_min`, `sliding_friction_max` | `(N,)` | Observed validation friction bounds |
| `family_names` | `(F,)` | Canonical family names |
| `split_names` | `(3,)` | `train`, `validation`, `held_out` |

The archive loader rejects non-finite values, invalid quaternions, non-positive
dimensions or physics bounds, unknown split IDs, and any family that appears in
more than one split.
