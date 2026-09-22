"""Expand the HORA grasp dataset around diverse qualified middle-contact seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from unilab.scripts.generate_leap_hora_grasps import (
    collection_plan,
    compose_collection_config,
    run_collection,
)
from unilab.tasks.manipulation.leap_inhand.hora_grasp import HoraGraspCache


def _cylinder_axes(quaternions: np.ndarray) -> np.ndarray:
    quat = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    w, x, y, z = quat.T
    return np.stack(
        (2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)),
        axis=1,
    )


def select_middle_contact_seed_indices(
    source: HoraGraspCache,
    scores: Any,
    *,
    seed_count: int,
    minimum_raw_contact: float,
) -> np.ndarray:
    """Select diverse source-cache rows from qualified raw middle contacts."""
    score_indices = np.asarray(scores["cache_index"], dtype=np.int64)
    tier = np.asarray(scores["tier_id"])
    thumb = np.asarray(scores["mean_raw_thumb_contact_fraction"])
    opposing = np.asarray(scores["mean_raw_opposing_contact_fraction"])
    support = np.count_nonzero(opposing >= minimum_raw_contact, axis=1)
    qualified = (
        np.isin(tier, (2, 3))
        & (thumb >= minimum_raw_contact)
        & (opposing[:, 1] >= minimum_raw_contact)
        & (support >= 2)
    )
    candidate_indices = score_indices[qualified]
    if candidate_indices.size < seed_count:
        raise RuntimeError(
            f"only {candidate_indices.size} qualified middle-contact seeds are available; "
            f"requested {seed_count}"
        )

    rows = source.take(candidate_indices)
    features = np.concatenate(
        (
            rows.joint_pos,
            rows.object_pos,
            _cylinder_axes(rows.object_quat),
        ),
        axis=1,
        dtype=np.float64,
    )
    scale = np.std(features, axis=0)
    scale[scale < 1.0e-6] = 1.0
    features = (features - np.mean(features, axis=0)) / scale
    raw_middle = opposing[qualified, 1]
    family = rows.family_id

    selected_local: list[int] = []
    # First cover every source family that contains qualified middle contact.
    for family_id in np.unique(family):
        members = np.flatnonzero(family == family_id)
        selected_local.append(int(members[np.argmax(raw_middle[members])]))
        if len(selected_local) == seed_count:
            break
    while len(selected_local) < seed_count:
        selected_features = features[np.asarray(selected_local)]
        distances = np.min(
            np.sum(np.square(features[:, None, :] - selected_features[None, :, :]), axis=2),
            axis=1,
        )
        distances[np.asarray(selected_local)] = -np.inf
        selected_local.append(int(np.argmax(distances)))
    return candidate_indices[np.asarray(selected_local, dtype=np.int64)]


def compose_middle_collection_config(args: argparse.Namespace):
    source_path = args.source.resolve()
    score_path = args.scores.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source cache not found: {source_path}")
    if not score_path.is_file():
        raise FileNotFoundError(f"contact score file not found: {score_path}")
    source = HoraGraspCache.load(source_path)
    with np.load(score_path) as scores:
        seed_indices = select_middle_contact_seed_indices(
            source,
            scores,
            seed_count=args.seed_count,
            minimum_raw_contact=args.minimum_raw_contact,
        )
    seeds = source.take(seed_indices)
    cfg = compose_collection_config(
        num_envs=args.num_envs,
        target=args.target,
        output_path=args.output,
        seed=args.seed,
    )
    families = []
    for position, source_index in enumerate(seed_indices):
        families.append(
            {
                "name": f"middle_opposition_seed_{position:02d}",
                "split": "train",
                "joint_pos": [float(value) for value in seeds.joint_pos[position]],
                "object_pos": [float(value) for value in seeds.object_pos[position]],
                "object_quat": [float(value) for value in seeds.object_quat[position]],
            }
        )
    reset = "env.events.reset_hand_object.params"
    OmegaConf.update(cfg, f"{reset}.canonical_families", families)
    OmegaConf.update(cfg, f"{reset}.cylinder_geometries", [[0.032, 0.070]])
    OmegaConf.update(cfg, f"{reset}.candidate_joint_noise", args.joint_noise)
    OmegaConf.update(cfg, f"{reset}.candidate_object_xy_noise", args.object_xy_noise)
    OmegaConf.update(cfg, f"{reset}.candidate_object_z_noise", args.object_z_noise)
    OmegaConf.update(cfg, f"{reset}.candidate_object_tilt_noise_deg", args.object_tilt_noise_deg)
    recorder = "env.recorders.grasp_cache.params"
    OmegaConf.update(
        cfg,
        f"{recorder}.minimum_raw_thumb_contact_fraction",
        args.minimum_raw_contact,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        f"{recorder}.minimum_raw_middle_contact_fraction",
        args.minimum_raw_contact,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        f"{recorder}.minimum_raw_opposing_contact_fraction",
        args.minimum_raw_contact,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        f"{recorder}.minimum_raw_opposing_fingers",
        2,
        force_add=True,
    )
    return cfg, seed_indices


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--seed-count", type=int, default=6)
    parser.add_argument("--max-control-steps", type=int, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("src/unilab/assets/caches/hora/leap_precision_grasps_v2.npz"),
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path(
            "src/unilab/assets/caches/hora/leap_precision_grasps_v4_contact_pilot_scores.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/unilab/assets/caches/hora/leap_precision_grasps_middle_expansion_2k.npz"),
    )
    parser.add_argument("--minimum-raw-contact", type=float, default=0.5)
    parser.add_argument("--joint-noise", type=float, default=0.08)
    parser.add_argument("--object-xy-noise", type=float, default=0.004)
    parser.add_argument("--object-z-noise", type=float, default=0.002)
    parser.add_argument("--object-tilt-noise-deg", type=float, default=8.0)
    parser.add_argument("--describe-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 0.0 <= args.minimum_raw_contact <= 1.0:
        raise ValueError("--minimum-raw-contact must be in [0, 1]")
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive")
    cfg, seed_indices = compose_middle_collection_config(args)
    plan = collection_plan(cfg, max_control_steps=args.max_control_steps)
    print(
        json.dumps(
            {
                "collection_plan": plan.as_dict(),
                "middle_contact_contract": {
                    "minimum_raw_contact_fraction": args.minimum_raw_contact,
                    "minimum_raw_opposing_fingers": 2,
                    "source_cache_indices": [int(value) for value in seed_indices],
                    "joint_noise": args.joint_noise,
                    "object_xy_noise": args.object_xy_noise,
                    "object_z_noise": args.object_z_noise,
                    "object_tilt_noise_deg": args.object_tilt_noise_deg,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.describe_only:
        return 0
    result = run_collection(cfg, max_control_steps=args.max_control_steps)
    print(json.dumps({"collection_result": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
