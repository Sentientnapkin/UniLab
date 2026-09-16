"""Generate HORA LEAP precision grasps without constructing a PPO runner.

The environment holds each sampled candidate with a zero action. Its recorder
owns repeated randomized-physics validation, diversity filtering, cache writes,
and the normal :class:`~unilab.base.run_control.RunComplete` stop condition.
"""

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
from unilab.tasks.manipulation.leap_inhand.hora_grasp import (
    HORA_GRASP_CACHE_SCHEMA_VERSION,
    HORA_GRASP_SPLITS,
)

_REPO_ROOT = Path(__file__).parents[3]
_CONFIG_ROOT = _REPO_ROOT / "src" / "unilab" / "conf" / "ppo"


@dataclass(frozen=True)
class HoraGraspCollectionPlan:
    """Resolved, backend-independent collection contract."""

    num_envs: int
    target: int
    seed: int
    max_control_steps: int
    output_path: Path
    validation_trials: int
    geometries: tuple[tuple[float, float], ...]
    families: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HORA_GRASP_CACHE_SCHEMA_VERSION,
            "num_envs": self.num_envs,
            "collection_target": self.target,
            "seed": self.seed,
            "max_control_steps": self.max_control_steps,
            "output_path": str(self.output_path),
            "validation_trials": self.validation_trials,
            "geometries": [
                {"radius": radius, "length": length} for radius, length in self.geometries
            ],
            "families": [{"name": name, "split": split} for name, split in self.families],
            "split_names": list(HORA_GRASP_SPLITS),
        }


def compose_collection_config(
    *,
    num_envs: int,
    target: int,
    output_path: Path,
    seed: int,
) -> DictConfig:
    """Compose the dedicated owner and apply bounded collection overrides."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if target <= 0:
        raise ValueError("target must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if output_path.suffix != ".npz":
        raise ValueError("output must end in .npz")
    with initialize_config_dir(config_dir=str(_CONFIG_ROOT), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=leap_inhand_hora_grasp/mujoco"])
    OmegaConf.update(cfg, "algo.num_envs", num_envs)
    OmegaConf.update(cfg, "algo.seed", seed)
    OmegaConf.update(cfg, "env.seed", seed, force_add=True)
    OmegaConf.update(cfg, "env.recorders.grasp_cache.params.collection_target", target)
    OmegaConf.update(
        cfg,
        "env.recorders.grasp_cache.params.output_path",
        str(output_path.resolve()),
    )
    return cfg


def collection_plan(cfg: DictConfig, *, max_control_steps: int) -> HoraGraspCollectionPlan:
    """Extract the user-visible output contract without creating a backend."""
    if max_control_steps <= 0:
        raise ValueError("max_control_steps must be positive")
    reset = cfg.env.events.reset_hand_object.params
    recorder = cfg.env.recorders.grasp_cache.params
    geometries = tuple((float(row[0]), float(row[1])) for row in reset.cylinder_geometries)
    families = tuple((str(row.name), str(row.split)) for row in reset.canonical_families)
    return HoraGraspCollectionPlan(
        num_envs=int(cfg.algo.num_envs),
        target=int(recorder.collection_target),
        seed=int(cfg.algo.seed),
        max_control_steps=max_control_steps,
        output_path=Path(str(recorder.output_path)),
        validation_trials=int(recorder.validation_trials),
        geometries=geometries,
        families=families,
    )


def run_collection(cfg: DictConfig, *, max_control_steps: int) -> dict[str, object]:
    """Run zero-action physics until the recorder completes or the safety cap fires."""
    plan = collection_plan(cfg, max_control_steps=max_control_steps)
    adapter = BackendAdapter(cfg, root_dir=_REPO_ROOT, algo_name="ppo")
    env_override = adapter.build_task_env_cfg_override()
    env = create_env(
        cfg,
        num_envs=plan.num_envs,
        env_cfg_override=env_override,
        sim_backend="mujoco",
        task_name="LeapInhandHoraGrasp",
    )
    try:
        control_step = 0
        try:
            env.init_state()
            raw_action_shape = env.action_space.shape
            if raw_action_shape is None:
                raise ValueError("HORA grasp environment action space must declare a shape")
            action_shape = tuple(int(value) for value in raw_action_shape)
            actions = np.zeros((plan.num_envs, *action_shape), dtype=np.float32)
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
            "HORA grasp collection reached max_control_steps before its target; "
            f"partial cache will be saved on close at {plan.output_path}"
        )
    finally:
        env.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--target", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-control-steps", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/unilab/assets/caches/hora/leap_precision_grasps_v2.npz"),
    )
    parser.add_argument(
        "--describe-only",
        action="store_true",
        help="Compose and print the collection/output contract without loading MuJoCo.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = compose_collection_config(
        num_envs=args.num_envs,
        target=args.target,
        output_path=args.output,
        seed=args.seed,
    )
    plan = collection_plan(cfg, max_control_steps=args.max_control_steps)
    print(json.dumps({"collection_plan": plan.as_dict()}, indent=2, sort_keys=True))
    if args.describe_only:
        return 0
    summary = run_collection(cfg, max_control_steps=args.max_control_steps)
    print(json.dumps({"collection_result": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
