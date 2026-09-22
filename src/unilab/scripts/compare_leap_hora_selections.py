"""Compare exact and control HORA selection reports with explicit go/no-go gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _recommended(report: dict[str, Any]) -> dict[str, Any]:
    checkpoint = report["recommended_checkpoint"]
    return next(row for row in report["rankings"] if row["checkpoint"] == checkpoint)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-rotation-advantage", type=float, default=0.5)
    parser.add_argument("--minimum-speed-advantage", type=float, default=0.025)
    parser.add_argument("--maximum-horizon-regression", type=float, default=0.05)
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    exact_report = json.loads(args.exact.read_text(encoding="utf-8"))
    control_report = json.loads(args.control.read_text(encoding="utf-8"))
    exact = _recommended(exact_report)
    control = _recommended(control_report)
    deltas = {
        name: float(exact["metrics"][name]) - float(control["metrics"][name])
        for name in (
            "horizon_fraction",
            "minimum_split_time_to_fall",
            "valid_grasp_fraction",
            "middle_contact_fraction",
            "net_rotation",
            "directional_speed",
            "direction_reversals",
        )
    }
    failures = []
    if not exact["passed"]:
        failures.append("the recommended exact checkpoint does not pass its selection gates")
    if deltas["horizon_fraction"] < -args.maximum_horizon_regression:
        failures.append(
            f"horizon regression {deltas['horizon_fraction']:.6g} exceeds "
            f"{args.maximum_horizon_regression:.6g}"
        )
    if (
        deltas["net_rotation"] < args.minimum_rotation_advantage
        and deltas["directional_speed"] < args.minimum_speed_advantage
    ):
        failures.append(
            "exact does not exceed either the rotation or directional-speed advantage gate"
        )
    result = {
        "schema_version": 1,
        "go": not failures,
        "failures": failures,
        "thresholds": {
            "minimum_rotation_advantage": args.minimum_rotation_advantage,
            "minimum_speed_advantage": args.minimum_speed_advantage,
            "maximum_horizon_regression": args.maximum_horizon_regression,
        },
        "exact": {
            "selection_path": str(args.exact.resolve()),
            "checkpoint": exact["checkpoint"],
            "metrics": exact["metrics"],
        },
        "control": {
            "selection_path": str(args.control.resolve()),
            "checkpoint": control["checkpoint"],
            "metrics": control["metrics"],
        },
        "exact_minus_control": deltas,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_pass and failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
