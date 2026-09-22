"""Run, resume, rank, and report deterministic HORA checkpoint sweeps."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from unilab.tasks.manipulation.leap_inhand.evaluation import (
    HORA_CHECKPOINT_SELECTION_DEFAULTS,
    rank_hora_checkpoint_evaluations,
)

_CHECKPOINT_RE = re.compile(r"model_(\d+)\.pt$")


def parse_checkpoint_spec(spec: str) -> list[int]:
    """Parse comma-separated integers and inclusive START:STOP:STEP ranges."""
    result: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            value = int(token)
            if value < 0:
                raise ValueError("checkpoint numbers must be nonnegative")
            result.add(value)
            continue
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(f"invalid checkpoint range {token!r}; expected START:STOP:STEP")
        start, stop, step = (int(value) for value in parts)
        if start < 0 or stop < start or step <= 0:
            raise ValueError(f"invalid checkpoint range {token!r}")
        result.update(range(start, stop + 1, step))
    if not result:
        raise ValueError("checkpoint specification selected no checkpoints")
    return sorted(result)


def discover_checkpoints(run_dir: Path) -> list[int]:
    checkpoints = []
    for path in run_dir.glob("model_*.pt"):
        match = _CHECKPOINT_RE.match(path.name)
        if match:
            checkpoints.append(int(match.group(1)))
    if not checkpoints:
        raise FileNotFoundError(f"no model_N.pt checkpoints found in {run_dir.resolve()}")
    return sorted(set(checkpoints))


def _threshold_overrides(args: argparse.Namespace) -> dict[str, float]:
    result = {}
    for name in HORA_CHECKPOINT_SELECTION_DEFAULTS[args.stage]:
        value = getattr(args, name)
        if value is not None:
            result[name] = float(value)
    return result


def _markdown(report: dict[str, Any], run_dir: Path) -> str:
    lines = [
        f"# HORA checkpoint selection: {run_dir.name}",
        "",
        f"Stage: `{report['stage']}`",
        "",
        f"Recommended checkpoint: **{report['recommended_checkpoint']}** "
        f"(gates {'passed' if report['recommended_passed'] else 'FAILED'})",
        "",
        "| Rank | Checkpoint | Pass | Horizon | Min split TTF | Valid grasp | Middle | "
        "Net rotation | Speed | Reversals |",
        "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rankings"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['rank']} | {row['checkpoint']} | "
            f"{'yes' if row['passed'] else 'no'} | {metrics['horizon_fraction']:.3f} | "
            f"{metrics['minimum_split_time_to_fall']:.2f} | "
            f"{metrics['valid_grasp_fraction']:.3f} | "
            f"{metrics['middle_contact_fraction']:.3f} | {metrics['net_rotation']:.3f} | "
            f"{metrics['directional_speed']:.3f} | {metrics['direction_reversals']:.1f} |"
        )
    lines.extend(("", "## Gate failures", ""))
    for row in report["rankings"]:
        if row["gate_failures"]:
            lines.append(f"- Checkpoint {row['checkpoint']}: " + "; ".join(row["gate_failures"]))
    if not any(row["gate_failures"] for row in report["rankings"]):
        lines.append("- None.")
    lines.extend(("", "## Thresholds", ""))
    lines.extend(f"- `{name}`: {value}" for name, value in report["thresholds"].items())
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--stage", choices=tuple(HORA_CHECKPOINT_SELECTION_DEFAULTS), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/leap_inhand/evaluation_manifest_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to RUN_DIR/checkpoint_sweep.",
    )
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="Comma list/ranges such as 200:400:25,399; defaults to every saved checkpoint.",
    )
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-evaluate existing reports.")
    parser.add_argument("--require-pass", action="store_true")
    for stage_thresholds in HORA_CHECKPOINT_SELECTION_DEFAULTS.values():
        for name in stage_thresholds:
            option = f"--{name.replace('_', '-')}"
            if option not in parser._option_string_actions:
                parser.add_argument(option, type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    output_dir = (args.output_dir or run_dir / "checkpoint_sweep").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = (
        parse_checkpoint_spec(args.checkpoints)
        if args.checkpoints
        else discover_checkpoints(run_dir)
    )
    missing = [value for value in checkpoints if not (run_dir / f"model_{value}.pt").is_file()]
    if missing:
        raise FileNotFoundError(f"requested checkpoints do not exist: {missing}")

    failures: dict[int, str] = {}
    if not args.analyze_only:
        for checkpoint in checkpoints:
            evaluation_path = output_dir / f"evaluation_{checkpoint}.json"
            if evaluation_path.is_file() and not args.force:
                print(f"[sweep] checkpoint {checkpoint}: reuse {evaluation_path}")
                continue
            log_path = output_dir / f"evaluation_{checkpoint}.log"
            command = [
                sys.executable,
                "-m",
                "unilab.scripts.evaluate_leap_inhand",
                "--manifest",
                str(args.manifest.resolve()),
                "--checkpoint",
                str(run_dir / f"model_{checkpoint}.pt"),
                "--output",
                str(evaluation_path),
                "--policy-family",
                "hora",
                "--profile",
                args.profile,
            ]
            print(f"[sweep] checkpoint {checkpoint}: evaluating")
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                failures[checkpoint] = f"exit={completed.returncode}; log={log_path}"
                print(f"[sweep] checkpoint {checkpoint}: FAILED ({failures[checkpoint]})")

    evaluations = []
    for checkpoint in checkpoints:
        path = output_dir / f"evaluation_{checkpoint}.json"
        if path.is_file():
            evaluations.append((checkpoint, path, json.loads(path.read_text(encoding="utf-8"))))
        elif checkpoint not in failures:
            failures[checkpoint] = f"missing report: {path}"
    if not evaluations:
        raise RuntimeError(f"no successful checkpoint evaluations; failures={failures}")
    report = rank_hora_checkpoint_evaluations(
        evaluations,
        stage=args.stage,
        thresholds=_threshold_overrides(args),
    )
    report.update(
        {
            "run_dir": str(run_dir),
            "profile": args.profile,
            "manifest": str(args.manifest.resolve()),
            "requested_checkpoints": checkpoints,
            "evaluation_failures": {str(key): value for key, value in failures.items()},
        }
    )
    json_path = output_dir / "selection.json"
    markdown_path = output_dir / "selection.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report, run_dir), encoding="utf-8")
    print(f"[sweep] selection JSON: {json_path}")
    print(f"[sweep] selection report: {markdown_path}")
    print(
        f"[sweep] recommended checkpoint: {report['recommended_checkpoint']} "
        f"(passed={report['recommended_passed']})"
    )
    if args.require_pass and not report["recommended_passed"]:
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
