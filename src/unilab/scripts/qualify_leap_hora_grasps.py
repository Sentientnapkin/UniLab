"""Classify HORA schema-v2 grasps by long-horizon passive retention."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.base.run_control import RunComplete
from unilab.tasks.manipulation.leap_inhand.hora_grasp import HoraGraspCache

_REPO_ROOT = Path(__file__).parents[3]
_CONFIG_ROOT = _REPO_ROOT / "src" / "unilab" / "conf" / "ppo"


@dataclass(frozen=True)
class PassiveQualificationPlan:
    num_envs: int
    seed: int
    max_control_steps: int
    input_path: Path
    output_prefix: Path
    eligible_grasps: int
    family_counts: dict[str, int]
    anchor_minimum_seconds: float
    recoverable_minimum_seconds: float
    marginal_minimum_seconds: float
    stage_a_size: int
    stage_a_anchor_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_envs": self.num_envs,
            "seed": self.seed,
            "max_control_steps": self.max_control_steps,
            "input_path": str(self.input_path),
            "output_prefix": str(self.output_prefix),
            "eligible_grasps": self.eligible_grasps,
            "family_counts": self.family_counts,
            "tier_thresholds_seconds": {
                "anchor": self.anchor_minimum_seconds,
                "recoverable": self.recoverable_minimum_seconds,
                "marginal": self.marginal_minimum_seconds,
            },
            "stage_a_size": self.stage_a_size,
            "stage_a_anchor_fraction": self.stage_a_anchor_fraction,
        }


def compose_qualification_config(
    *,
    num_envs: int,
    seed: int,
    input_path: Path,
    output_prefix: Path,
    anchor_minimum_seconds: float,
    recoverable_minimum_seconds: float,
    marginal_minimum_seconds: float,
    stage_a_size: int,
    stage_a_anchor_fraction: float,
) -> DictConfig:
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if input_path.suffix != ".npz":
        raise ValueError("input must end in .npz")
    resolved_input = input_path.resolve()
    if not resolved_input.is_file():
        raise FileNotFoundError(f"HORA grasp cache does not exist: {resolved_input}")
    if output_prefix.suffix:
        raise ValueError("output-prefix must not have a file extension")
    if stage_a_size <= 0:
        raise ValueError("stage-a-size must be positive")
    if not 0.0 <= stage_a_anchor_fraction <= 1.0:
        raise ValueError("stage-a-anchor-fraction must be in [0, 1]")
    if not 0.0 <= marginal_minimum_seconds < recoverable_minimum_seconds < anchor_minimum_seconds:
        raise ValueError("tier thresholds must satisfy 0 <= marginal < recoverable < anchor")

    with initialize_config_dir(config_dir=str(_CONFIG_ROOT), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["task=leap_inhand_hora_stage1/mujoco_nominal"],
        )
    resolved_prefix = output_prefix.resolve()
    OmegaConf.update(cfg, "algo.num_envs", num_envs)
    OmegaConf.update(cfg, "algo.seed", seed)
    OmegaConf.update(cfg, "env.seed", seed, force_add=True)
    OmegaConf.update(
        cfg,
        "env.events.reset_hand_object.params.grasp_cache_path",
        str(resolved_input),
    )
    OmegaConf.update(
        cfg,
        "env.events.reset_hand_object.params.grasp_cache_sampling",
        "sequential",
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "env.events.reset_hand_object.params.grasp_cache_start_index",
        0,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "env.recorders.passive_qualification",
        {
            "func": (
                "unilab.tasks.manipulation.leap_inhand.hora_passive_qualification."
                "LeapHoraPassiveQualificationRecorder"
            ),
            "params": {
                "reset_event_name": "reset_hand_object",
                "grasp_term_name": "dropped",
                "input_cache_path": str(resolved_input),
                "output_prefix": str(resolved_prefix),
                "anchor_minimum_seconds": anchor_minimum_seconds,
                "recoverable_minimum_seconds": recoverable_minimum_seconds,
                "marginal_minimum_seconds": marginal_minimum_seconds,
                "training_cache_size": stage_a_size,
                "training_anchor_fraction": stage_a_anchor_fraction,
                "sampling_seed": seed,
            },
        },
        force_add=True,
    )
    return cfg


def qualification_plan(cfg: DictConfig, *, max_control_steps: int) -> PassiveQualificationPlan:
    if max_control_steps <= 0:
        raise ValueError("max-control-steps must be positive")
    reset = cfg.env.events.reset_hand_object.params
    recorder = cfg.env.recorders.passive_qualification.params
    cache = HoraGraspCache.load(Path(str(recorder.input_cache_path)))
    radius, length = (float(value) for value in reset.cylinder_geometry_override)
    train_id = 0
    eligible = (
        (cache.split == train_id)
        & np.isclose(cache.cylinder_radius, radius)
        & np.isclose(cache.cylinder_length, length)
    )
    indices = np.flatnonzero(eligible)
    family_counts = {
        cache.family_names[int(family)]: int(np.count_nonzero(cache.family_id[indices] == family))
        for family in np.unique(cache.family_id[indices])
    }
    return PassiveQualificationPlan(
        num_envs=int(cfg.algo.num_envs),
        seed=int(cfg.algo.seed),
        max_control_steps=max_control_steps,
        input_path=Path(str(recorder.input_cache_path)),
        output_prefix=Path(str(recorder.output_prefix)),
        eligible_grasps=int(indices.size),
        family_counts=family_counts,
        anchor_minimum_seconds=float(recorder.anchor_minimum_seconds),
        recoverable_minimum_seconds=float(recorder.recoverable_minimum_seconds),
        marginal_minimum_seconds=float(recorder.marginal_minimum_seconds),
        stage_a_size=int(recorder.training_cache_size),
        stage_a_anchor_fraction=float(recorder.training_anchor_fraction),
    )


def run_qualification(cfg: DictConfig, *, max_control_steps: int) -> dict[str, object]:
    plan = qualification_plan(cfg, max_control_steps=max_control_steps)
    adapter = BackendAdapter(cfg, root_dir=_REPO_ROOT, algo_name="ppo")
    env = create_env(
        cfg,
        num_envs=plan.num_envs,
        env_cfg_override=adapter.build_task_env_cfg_override(),
        sim_backend="mujoco",
        task_name="LeapInhandHoraStage1",
    )
    try:
        control_step = 0
        try:
            env.init_state()
            action_shape = env.action_space.shape
            if action_shape is None:
                raise ValueError("HORA environment action space must declare a shape")
            actions = np.zeros(
                (plan.num_envs, *(int(value) for value in action_shape)), dtype=np.float32
            )
            for control_step in range(1, max_control_steps + 1):
                env.step(actions)
        except RunComplete as completion:
            return {
                "status": "complete",
                "reason": completion.reason,
                "control_steps": control_step,
                **dict(completion.summary),
            }
        raise RuntimeError(
            "passive qualification reached max-control-steps before all eligible grasps "
            "completed; partial scores were saved on close"
        )
    finally:
        env.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-control-steps", type=int, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("src/unilab/assets/caches/hora/leap_precision_grasps_v2.npz"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("src/unilab/assets/caches/hora/leap_precision_grasps_v3"),
    )
    parser.add_argument("--anchor-minimum-seconds", type=float, default=16.0)
    parser.add_argument("--recoverable-minimum-seconds", type=float, default=1.0)
    parser.add_argument("--marginal-minimum-seconds", type=float, default=0.5)
    parser.add_argument("--stage-a-size", type=int, default=3000)
    parser.add_argument("--stage-a-anchor-fraction", type=float, default=0.8)
    parser.add_argument("--describe-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = compose_qualification_config(
        num_envs=args.num_envs,
        seed=args.seed,
        input_path=args.input,
        output_prefix=args.output_prefix,
        anchor_minimum_seconds=args.anchor_minimum_seconds,
        recoverable_minimum_seconds=args.recoverable_minimum_seconds,
        marginal_minimum_seconds=args.marginal_minimum_seconds,
        stage_a_size=args.stage_a_size,
        stage_a_anchor_fraction=args.stage_a_anchor_fraction,
    )
    plan = qualification_plan(cfg, max_control_steps=args.max_control_steps)
    print(json.dumps({"qualification_plan": plan.as_dict()}, indent=2, sort_keys=True))
    if args.describe_only:
        return 0
    result = run_qualification(cfg, max_control_steps=args.max_control_steps)
    print(json.dumps({"qualification_result": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
