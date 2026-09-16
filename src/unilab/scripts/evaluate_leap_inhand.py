"""Run deterministic LEAP checkpoint evaluation from a versioned manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from unilab.cli import build_command
from unilab.tasks.manipulation.leap_inhand.evaluation import (
    load_evaluation_manifest,
    summarize_episode_reports,
)


def _scalar(value: float) -> str:
    return format(float(value), ".12g")


def build_case_overrides(
    case: dict[str, Any],
    output_path: Path,
    *,
    policy_family: str,
    checkpoint: str = "-1",
) -> list[str]:
    """Translate one manifest row into fixed Hydra environment overrides."""
    if policy_family != "hora" and (
        case["cylinder_radius"] != 0.032 or case["cylinder_length"] != 0.070
    ):
        raise ValueError(
            "the current LEAP scene supports only radius=0.032 m and length=0.070 m; "
            "geometry-aware scene materialization belongs to Phase 1"
        )
    if case["torsional_friction"] != 0.05:
        raise ValueError("the current fixed-domain evaluator supports torsional_friction=0.05")
    if case["action_delay_steps"] not in {0, 1}:
        raise ValueError("the current fixed-domain evaluator supports 0 or 1 delay steps")
    com = [_scalar(value) for value in case["com"]]
    gain = float(case["pd_gain_scale"])
    overrides = [
        f"algo.load_run={checkpoint}",
        "algo.num_envs=1",
        f"algo.seed={case['seed']}",
        f"+env.seed={case['seed']}",
        "training.play_env_num=1",
        "training.play_render_mode=none",
        "training.play_steps=400",
        # The manifest intentionally fixes/noises policy-I/O fields differently
        # from the training snapshot; the evaluator records those differences.
        "training.sim2sim_strict=false",
        "+training.headless_play_collection=true",
        "+env.events.reset_hand_object.params.grasp_cache_sampling=sequential",
        f"+env.events.reset_hand_object.params.grasp_cache_start_index={case['grasp_index']}",
        "env.events.reset_hand_object.params.randomization_fraction=0.0",
        "env.events.reset_hand_object.params.joint_noise=0.0",
        "env.events.reset_hand_object.params.object_xy_noise=0.0",
        "env.events.reset_hand_object.params.object_z_noise=0.0",
        "env.events.reset_hand_object.params.object_tilt_noise_deg=0.0",
        "env.events.reset_hand_object.params.object_velocity_noise=0.0",
        "env.events.reset_hand_object.params.object_angular_velocity_noise=0.0",
        f"env.events.cylinder_friction.params.ranges=[{_scalar(case['sliding_friction'])},{_scalar(case['sliding_friction'])}]",
        f"env.events.cylinder_com.params.com_range.x=[{com[0]},{com[0]}]",
        f"env.events.cylinder_com.params.com_range.y=[{com[1]},{com[1]}]",
        f"env.events.cylinder_com.params.com_range.z=[{com[2]},{com[2]}]",
        f"env.events.pd_gains.params.kp_range=[{_scalar(gain)},{_scalar(gain)}]",
        f"env.events.pd_gains.params.kd_range=[{_scalar(0.1 * gain)},{_scalar(0.1 * gain)}]",
        "env.events.pd_gains.params.operation=abs",
        f"env.events.encoder_bias.params.bias_range=[{_scalar(case['encoder_bias'])},{_scalar(case['encoder_bias'])}]",
        f"env.observations.critic.terms.rotation.params.joint_noise={_scalar(case['observation_noise_std'])}",
        "++env.recorders.evaluation.func=unilab.tasks.manipulation.leap_inhand.evaluation.LeapEvaluationRecorder",
        f"++env.recorders.evaluation.params.output_path={output_path}",
        f"++env.recorders.evaluation.params.case_id={case['id']}",
        f"++env.recorders.evaluation.params.split={case['split']}",
        f"++env.recorders.evaluation.params.grasp_index={case['grasp_index']}",
        "++env.recorders.evaluation.params.state_term_name=dropped",
        "++env.recorders.evaluation.params.rotation_reward_name="
        + ("frontier" if policy_family == "hora" else "rotate"),
        "++env.recorders.evaluation.params.action_name=hand",
    ]
    if policy_family == "hora":
        density = case["object_mass"] / (
            math.pi * case["cylinder_radius"] ** 2 * case["cylinder_length"]
        )
        overrides.extend(
            (
                "env.events.reset_hand_object.params.grasp_cache_path=caches/hora/leap_precision_grasps_v2.npz",
                "env.events.reset_hand_object.params.cache_splits=[train,validation,held_out]",
                "+env.events.reset_hand_object.params.cylinder_geometry_override="
                f"[{_scalar(case['cylinder_radius'])},{_scalar(case['cylinder_length'])}]",
                "env.events.cylinder_mass_inertia.params.density_range="
                f"[{_scalar(density)},{_scalar(density)}]",
                f"env.actions.hand.latency_steps_range=[{case['action_delay_steps']},{case['action_delay_steps']}]",
                f"env.actions.hand.action_noise_std_schedule=[{{step:0,std:{_scalar(case['action_noise_std'])}}}]",
                f"env.observations.actor.terms.hardware.params.joint_noise={_scalar(case['observation_noise_std'])}",
            )
        )
    else:
        delay_probability = float(case["action_delay_steps"])
        mass_scale = case["object_mass"] / 0.05
        overrides.extend(
            (
                "env.events.reset_hand_object.params.grasp_cache_path=caches/leap_cylinder_grasp_diverse_5k.npy",
                "env.events.cylinder_mass_inertia.params.scale_range="
                f"[{_scalar(mass_scale)},{_scalar(mass_scale)}]",
                f"env.actions.hand.action_delay_probability_schedule=[{{step:0,probability:{_scalar(delay_probability)}}}]",
                f"env.actions.hand.action_noise_std_schedule=[{{step:0,std:{_scalar(case['action_noise_std'])}}}]",
                f"env.observations.actor.terms.proprio.params.joint_noise={_scalar(case['observation_noise_std'])}",
            )
        )
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy-family", choices=("gait", "hora"), required=True)
    parser.add_argument(
        "--task-owner",
        default=None,
        help="Optional PPO task owner; defaults to the matching gait or HORA owner.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Run only the named manifest case; repeat to select several cases.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_evaluation_manifest(args.manifest)
    task = args.task_owner or (
        "leap_inhand_hora_stage1" if args.policy_family == "hora" else "leap_inhand_dynamics"
    )
    episode_dir = args.output.parent / f"{args.output.stem}_episodes"
    reports: list[dict[str, Any]] = []
    cases = manifest["cases"]
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["id"] in selected]
        missing = selected - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"unknown manifest case ids: {sorted(missing)}")
    for case in cases:
        case_output = episode_dir / f"{case['id']}.json"
        overrides = build_case_overrides(
            case,
            case_output,
            policy_family=args.policy_family,
            checkpoint=args.checkpoint,
        )
        command = build_command(
            mode="eval",
            algo="ppo",
            task=task,
            sim="mujoco",
            profile=None,
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
        episode = json.loads(case_output.read_text(encoding="utf-8"))
        episode["domain"] = case
        case_output.write_text(
            json.dumps(episode, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        reports.append(episode)
    if args.dry_run:
        return 0
    summary = summarize_episode_reports(reports)
    summary["manifest"] = str(args.manifest.resolve())
    summary["checkpoint"] = str(Path(args.checkpoint).resolve())
    summary["policy_family"] = args.policy_family
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
