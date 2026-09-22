"""Compare HORA checkpoints with deterministic per-finger rollout diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from unilab.cli import build_command
from unilab.scripts.evaluate_leap_inhand import build_case_overrides
from unilab.tasks.manipulation.leap_inhand.evaluation import (
    load_evaluation_manifest,
    summarize_episode_reports,
    summarize_hora_rollout_audits,
)


def build_audit_overrides(case: dict[str, Any], output_path: Path) -> list[str]:
    """Build the additional recorder overrides for one fixed rollout."""
    return [
        "++env.recorders.hora_rollout_audit.func="
        "unilab.tasks.manipulation.leap_inhand.evaluation.LeapHoraRolloutAuditRecorder",
        f"++env.recorders.hora_rollout_audit.params.output_path={output_path}",
        f"++env.recorders.hora_rollout_audit.params.case_id={case['id']}",
        f"++env.recorders.hora_rollout_audit.params.split={case['split']}",
        f"++env.recorders.hora_rollout_audit.params.grasp_index={case['grasp_index']}",
        "++env.recorders.hora_rollout_audit.params.state_term_name=dropped",
        "++env.recorders.hora_rollout_audit.params.action_name=hand",
    ]


def _parse_runs(values: list[list[str]]) -> list[tuple[str, str, Path]]:
    labels: set[str] = set()
    runs: list[tuple[str, str, Path]] = []
    for label, profile, checkpoint_value in values:
        if label in labels:
            raise ValueError(f"duplicate run label {label!r}")
        labels.add(label)
        checkpoint = Path(checkpoint_value).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        runs.append((label, profile, checkpoint))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        required=True,
        metavar=("LABEL", "PROFILE", "CHECKPOINT"),
        help="Checkpoint label, task profile, and checkpoint path; repeat to compare runs.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Run only this manifest case; repeat to select multiple cases.",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_evaluation_manifest(args.manifest)
    runs = _parse_runs(args.run)
    cases = manifest["cases"]
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["id"] in selected]
        missing = selected - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"unknown manifest case ids: {sorted(missing)}")

    root = args.output.parent / f"{args.output.stem}_episodes"
    summaries: dict[str, Any] = {}
    for label, profile, checkpoint in runs:
        evaluation_reports: list[dict[str, Any]] = []
        audit_reports: list[dict[str, Any]] = []
        for case in cases:
            case_dir = root / label
            evaluation_path = case_dir / f"{case['id']}_evaluation.json"
            audit_path = case_dir / f"{case['id']}_audit.json"
            overrides = build_case_overrides(
                case,
                evaluation_path,
                policy_family="hora",
                checkpoint=str(checkpoint),
            )
            overrides.extend(build_audit_overrides(case, audit_path))
            overrides.append(f"training.device={args.device}")
            command = build_command(
                mode="eval",
                algo="ppo",
                task="leap_inhand_hora_stage1",
                sim="mujoco",
                profile=profile,
                overrides=overrides,
                load_run=None,
                render_mode="none",
            )
            if args.dry_run:
                print(" ".join(str(part) for part in command))
                continue
            subprocess.run(
                command,
                check=True,
                env={**os.environ, "UNILAB_EXPORT_POLICY": "0"},
            )
            evaluation_report = json.loads(evaluation_path.read_text(encoding="utf-8"))
            evaluation_report["domain"] = case
            audit_report = json.loads(audit_path.read_text(encoding="utf-8"))
            audit_report["domain"] = case
            evaluation_reports.append(evaluation_report)
            audit_reports.append(audit_report)
        if not args.dry_run:
            summaries[label] = {
                "profile": profile,
                "checkpoint": str(checkpoint),
                "performance": summarize_episode_reports(evaluation_reports),
                "rollout_audit": summarize_hora_rollout_audits(audit_reports),
                "episodes": [
                    {"evaluation": evaluation, "audit": audit}
                    for evaluation, audit in zip(evaluation_reports, audit_reports, strict=True)
                ],
            }

    if args.dry_run:
        return 0
    payload = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "runs": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
