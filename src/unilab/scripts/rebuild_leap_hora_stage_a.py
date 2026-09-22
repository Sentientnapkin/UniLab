"""Rebuild a HORA Stage A cache offline from passive-qualification scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from unilab.tasks.manipulation.leap_inhand.hora_grasp import HoraGraspCache
from unilab.tasks.manipulation.leap_inhand.hora_passive_qualification import (
    select_contact_balanced_indices,
)

_DEFAULT_TARGETS = (
    "balanced_thumb_index_middle=400",
    "index_forward_load=560",
    "outer_finger_load=240",
    "middle_opposition_expansion=800",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("src/unilab/assets/caches/hora/leap_precision_grasps_v5_candidates.npz"),
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("src/unilab/assets/caches/hora/leap_precision_grasps_v5_scores.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/unilab/assets/caches/hora/leap_precision_grasps_v5_stage_a_weighted.npz"),
    )
    parser.add_argument(
        "--family-target",
        action="append",
        default=None,
        metavar="NAME=COUNT",
        help="Exact output rows for one family; defaults to the v5 capacity-aware allocation.",
    )
    parser.add_argument("--anchor-fraction", type=float, default=0.8)
    parser.add_argument("--middle-contact-fraction", type=float, default=0.4)
    parser.add_argument("--minimum-raw-thumb-contact", type=float, default=0.5)
    parser.add_argument("--minimum-raw-middle-contact", type=float, default=0.5)
    parser.add_argument("--minimum-raw-opposing-contact", type=float, default=0.5)
    parser.add_argument("--minimum-raw-opposing-fingers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _parse_targets(values: list[str], cache: HoraGraspCache) -> dict[int, int]:
    by_name: dict[str, int] = {}
    for value in values:
        name, separator, raw_count = value.rpartition("=")
        if not separator or not name:
            raise ValueError(f"invalid family target {value!r}; expected NAME=COUNT")
        if name in by_name:
            raise ValueError(f"duplicate family target for {name!r}")
        try:
            count = int(raw_count)
        except ValueError as error:
            raise ValueError(f"invalid family target count in {value!r}") from error
        if count <= 0:
            raise ValueError(f"family target count must be positive in {value!r}")
        by_name[name] = count
    known = {cache.family_names[int(family)] for family in np.unique(cache.family_id)}
    if set(by_name) != known:
        raise ValueError(
            "family targets must name every cache family exactly once; "
            f"expected={sorted(known)}, received={sorted(by_name)}"
        )
    return {cache.family_names.index(name): count for name, count in by_name.items()}


def _load_scores(path: Path, cache: HoraGraspCache) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"qualification score file does not exist: {path.resolve()}")
    with np.load(path, allow_pickle=False) as payload:
        scores = {name: np.array(payload[name], copy=True) for name in payload.files}
    if not bool(scores["complete"].item()):
        raise ValueError("qualification scores are incomplete")
    indices = scores["cache_index"].astype(np.int64)
    if indices.ndim != 1 or np.unique(indices).size != indices.size:
        raise ValueError("qualification cache_index must contain unique one-dimensional rows")
    if indices.size and (np.any(indices < 0) or np.any(indices >= cache.size)):
        raise ValueError("qualification cache_index is out of range")
    if not np.array_equal(scores["family_id"], cache.family_id[indices]):
        raise ValueError("qualification family IDs do not match the input cache")
    return scores


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    for name in (
        "anchor_fraction",
        "middle_contact_fraction",
        "minimum_raw_thumb_contact",
        "minimum_raw_middle_contact",
        "minimum_raw_opposing_contact",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if not 1 <= args.minimum_raw_opposing_fingers <= 3:
        raise ValueError("--minimum-raw-opposing-fingers must be in [1, 3]")

    cache = HoraGraspCache.load(args.input)
    scores = _load_scores(args.scores, cache)
    targets = _parse_targets(args.family_target or list(_DEFAULT_TARGETS), cache)
    cache_size = sum(targets.values())
    indices = scores["cache_index"].astype(np.int64)
    tier = np.full(cache.size, -1, dtype=np.int8)
    raw_thumb = np.full(cache.size, np.nan, dtype=np.float64)
    raw_opposing = np.full((cache.size, 3), np.nan, dtype=np.float64)
    tier[indices] = scores["tier_id"]
    raw_thumb[indices] = scores["mean_raw_thumb_contact_fraction"]
    raw_opposing[indices] = scores["mean_raw_opposing_contact_fraction"]

    selected, enriched = select_contact_balanced_indices(
        eligible=indices,
        tier=tier,
        family_id=cache.family_id,
        raw_thumb_fraction=raw_thumb,
        raw_opposing_fraction=raw_opposing,
        cache_size=cache_size,
        anchor_fraction=args.anchor_fraction,
        middle_contact_fraction=args.middle_contact_fraction,
        minimum_raw_thumb_contact_fraction=args.minimum_raw_thumb_contact,
        minimum_raw_middle_contact_fraction=args.minimum_raw_middle_contact,
        minimum_raw_opposing_contact_fraction=args.minimum_raw_opposing_contact,
        minimum_raw_opposing_fingers=args.minimum_raw_opposing_fingers,
        seed=args.seed,
        family_targets=targets,
    )
    output = args.output.resolve()
    cache.take(selected).save(output)
    selection_path = output.with_name(f"{output.stem}_selection.npz")
    np.savez_compressed(
        selection_path,
        source_cache_path=np.asarray(str(args.input.resolve())),
        score_path=np.asarray(str(args.scores.resolve())),
        source_cache_index=selected,
        middle_enriched=enriched,
    )
    family_counts = {
        cache.family_names[int(family)]: int(np.count_nonzero(cache.family_id[selected] == family))
        for family in np.unique(cache.family_id[selected])
    }
    tier_names = tuple(str(value) for value in scores["tier_names"].tolist())
    report = {
        "output_path": str(output),
        "selection_path": str(selection_path),
        "rows": int(selected.size),
        "unique_source_rows": int(np.unique(selected).size),
        "duplicate_fraction": float(1.0 - np.unique(selected).size / selected.size),
        "family_counts": family_counts,
        "tier_counts": {
            name: int(np.count_nonzero(tier[selected] == tier_id))
            for tier_id, name in enumerate(tier_names)
        },
        "middle_enriched_rows": int(np.count_nonzero(enriched)),
        "unique_middle_enriched_source_rows": int(np.unique(selected[enriched]).size),
        "raw_contact": {
            "thumb_mean": float(np.mean(raw_thumb[selected])),
            "middle_mean": float(np.mean(raw_opposing[selected, 1])),
            "middle_threshold_fraction": float(
                np.mean(raw_opposing[selected, 1] >= args.minimum_raw_middle_contact)
            ),
            "enriched_middle_minimum": float(np.min(raw_opposing[selected[enriched], 1])),
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stage_a_rebuild": report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
