"""Merge proven general grasps with the middle-contact expansion for v5 qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from unilab.tasks.manipulation.leap_inhand.hora_grasp import (
    HORA_GRASP_SPLITS,
    HoraGraspCache,
    merge_hora_grasp_caches,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--general-cache",
        action="append",
        type=Path,
        default=None,
        help="Qualified general-purpose cache; may be supplied more than once.",
    )
    parser.add_argument(
        "--middle-cache",
        type=Path,
        default=Path("src/unilab/assets/caches/hora/leap_precision_grasps_middle_expansion_2k.npz"),
    )
    parser.add_argument(
        "--middle-family-name",
        default="middle_opposition_expansion",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/unilab/assets/caches/hora/leap_precision_grasps_v5_candidates.npz"),
    )
    parser.add_argument("--keep-exact-duplicates", action="store_true")
    return parser


def _counts(cache: HoraGraspCache) -> dict[str, object]:
    return {
        "rows": cache.size,
        "family_counts": {
            cache.family_names[int(family)]: int(np.count_nonzero(cache.family_id == family))
            for family in np.unique(cache.family_id)
        },
        "split_counts": {
            name: int(np.count_nonzero(cache.split == split_id))
            for split_id, name in enumerate(HORA_GRASP_SPLITS)
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    general_paths = args.general_cache or [
        Path("src/unilab/assets/caches/hora/leap_precision_grasps_v3_anchor.npz"),
        Path("src/unilab/assets/caches/hora/leap_precision_grasps_v3_recoverable.npz"),
    ]
    paths = [*general_paths, args.middle_cache]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"HORA grasp cache does not exist: {path.resolve()}")
    if args.output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    if not args.middle_family_name:
        raise ValueError("--middle-family-name must be non-empty")

    caches = [HoraGraspCache.load(path) for path in paths]
    middle_map = {name: args.middle_family_name for name in caches[-1].family_names}
    merged = merge_hora_grasp_caches(
        caches,
        family_name_maps=[*[None] * len(general_paths), middle_map],
        deduplicate=not args.keep_exact_duplicates,
    )
    output = args.output.resolve()
    merged.save(output)
    report = {
        "output_path": str(output),
        "manifest_path": str(output.with_suffix(".json")),
        "sources": [
            {"path": str(path.resolve()), **_counts(cache)}
            for path, cache in zip(paths, caches, strict=True)
        ],
        "middle_family_name": args.middle_family_name,
        "exact_duplicates_removed": int(sum(cache.size for cache in caches) - merged.size),
        "merged": _counts(merged),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"merge_result": report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
