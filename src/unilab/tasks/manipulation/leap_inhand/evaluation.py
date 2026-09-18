"""Deterministic evaluation manifests and machine-readable LEAP reports."""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.managers import RecorderTerm, RecorderTermCfg

from .hora import LeapHoraPositionAction, joint_limit_occupancy
from .manager_terms import (
    LeapBidirectionalFrontierReward,
    LeapGraspTermination,
    LeapIncrementalPositionAction,
    LeapRotationObservation,
    _name,
    _positive_int,
)

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


MANIFEST_FIELDS = frozenset(
    {
        "id",
        "split",
        "seed",
        "grasp_index",
        "cylinder_radius",
        "cylinder_length",
        "object_mass",
        "sliding_friction",
        "torsional_friction",
        "com",
        "pd_gain_scale",
        "encoder_bias",
        "action_delay_steps",
        "action_noise_std",
        "observation_noise_std",
    }
)


def load_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    """Load and fail closed on an incomplete or ambiguous evaluation manifest."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "cases"}:
        raise ValueError("evaluation manifest must contain only schema_version and cases")
    if payload["schema_version"] != 1:
        raise ValueError(f"unsupported evaluation manifest schema {payload['schema_version']!r}")
    cases = payload["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation manifest cases must be a non-empty list")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != MANIFEST_FIELDS:
            missing = sorted(MANIFEST_FIELDS - set(case)) if isinstance(case, dict) else []
            extra = sorted(set(case) - MANIFEST_FIELDS) if isinstance(case, dict) else []
            raise ValueError(
                f"manifest case {index} fields mismatch: missing={missing}, extra={extra}"
            )
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"manifest case {index} id must be non-empty and unique")
        seen.add(case_id)
        if case["split"] not in {"in_distribution", "edge_of_range", "out_of_distribution"}:
            raise ValueError(f"manifest case {case_id!r} has invalid split")
        for field in ("seed", "grasp_index", "action_delay_steps"):
            value = case[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"manifest case {case_id!r} {field} must be nonnegative integer")
        com = np.asarray(case["com"], dtype=np.float64)
        if com.shape != (3,) or not np.isfinite(com).all():
            raise ValueError(f"manifest case {case_id!r} com must be a finite XYZ triple")
        numeric = MANIFEST_FIELDS - {
            "id",
            "split",
            "seed",
            "grasp_index",
            "com",
            "action_delay_steps",
        }
        for field in numeric:
            value = case[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"manifest case {case_id!r} {field} must be finite numeric")
    return payload


def summarize_episode_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize episode reports globally and by requested domain split."""
    if not reports:
        raise ValueError("at least one episode report is required")
    metric_names = (
        "time_to_fall_seconds",
        "net_committed_axis_rotation",
        "accumulated_rotation",
        "directional_speed",
        "mean_axial_rotation_velocity",
        "object_linear_velocity",
        "estimated_torque",
        "estimated_work",
        "valid_grasp_fraction",
        "thumb_contact_fraction",
        "index_contact_fraction",
        "middle_contact_fraction",
        "outer_contact_fraction",
        "palm_support_fraction",
        "joint_limit_occupancy",
        "direction_reversals",
    )

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "episodes": len(rows),
            "metrics": {
                name: {
                    "median": float(statistics.median(float(row[name]) for row in rows)),
                    "mean": float(statistics.fmean(float(row[name]) for row in rows)),
                }
                for name in metric_names
            },
        }

    splits = sorted({str(report["split"]) for report in reports})
    return {
        "schema_version": 1,
        "overall": summarize(reports),
        "by_split": {
            split: summarize([report for report in reports if report["split"] == split])
            for split in splits
        },
        "episodes": list(reports),
    }


class LeapEvaluationRecorder(RecorderTerm):
    """Write one complete JSON episode report for a deterministic rollout."""

    _ALLOWED_PARAMS = frozenset(
        {
            "output_path",
            "case_id",
            "split",
            "grasp_index",
            "state_term_name",
            "rotation_reward_name",
            "action_name",
        }
    )

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        term = type(self).__name__
        unexpected = set(cfg.params) - self._ALLOWED_PARAMS
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        self._output_path = Path(_name(term, "output_path", cfg.params.get("output_path")))
        self._case_id = _name(term, "case_id", cfg.params.get("case_id"))
        self._split = _name(term, "split", cfg.params.get("split"))
        grasp_index = cfg.params.get("grasp_index")
        if isinstance(grasp_index, bool) or not isinstance(grasp_index, int):
            raise TypeError(f"{term} grasp_index must be a nonnegative integer")
        self._grasp_index = _positive_int(term, "grasp_index_plus_one", grasp_index + 1) - 1
        state_name = _name(term, "state_term_name", cfg.params.get("state_term_name"))
        state_cfg = env.termination_manager.get_term_cfg(state_name)
        if not isinstance(state_cfg.func, LeapGraspTermination):
            raise TypeError(f"{term} state term must be LeapGraspTermination")
        self._grasp = state_cfg.func
        observation = self._grasp.observation
        if not isinstance(observation, LeapRotationObservation):
            raise TypeError(f"{term} requires LeapRotationObservation")
        self._state = observation
        rotation_name = _name(term, "rotation_reward_name", cfg.params.get("rotation_reward_name"))
        rotation = env.reward_manager.get_term_cfg(rotation_name).func
        if not isinstance(rotation, LeapBidirectionalFrontierReward):
            raise TypeError(f"{term} rotation reward must be LeapBidirectionalFrontierReward")
        self._rotation = rotation
        action_name = _name(term, "action_name", cfg.params.get("action_name"))
        action = env.action_manager.get_term(action_name)
        if not isinstance(action, LeapIncrementalPositionAction):
            raise TypeError(f"{term} action must be a LEAP position action")
        self._action_name = action_name
        self._sum_fields = {
            name: np.zeros(env.num_envs, dtype=np.float64)
            for name in (
                "object_linear_velocity",
                "mean_axial_rotation_velocity",
                "estimated_torque",
                "estimated_work",
                "valid_grasp_fraction",
                "thumb_contact_fraction",
                "index_contact_fraction",
                "middle_contact_fraction",
                "outer_contact_fraction",
                "palm_support_fraction",
                "joint_limit_occupancy",
            )
        }
        self._steps = np.zeros(env.num_envs, dtype=np.int64)
        self._written = False

    def _accumulate(self, ids: np.ndarray) -> None:
        if not len(ids):
            return
        state = self._state
        grasp = self._grasp
        values = {
            "object_linear_velocity": np.linalg.norm(state.object_linvel, axis=1),
            # HORA reports the episode-average *unclipped* omega dot desired-axis
            # value in simulation. Keep it alongside UniLab's net/frontier gates.
            "mean_axial_rotation_velocity": np.sum(
                state.object_angvel * grasp.cylinder_axis_w, axis=1
            )
            * ~grasp.fallen,
            "estimated_torque": np.sum(np.square(state.torques), axis=1),
            "estimated_work": np.square(np.sum(state.torques * state.dof_vel, axis=1)),
            "valid_grasp_fraction": grasp.valid_grasp,
            "thumb_contact_fraction": grasp.thumb_contact,
            "index_contact_fraction": grasp.opposing_contacts[:, 0],
            "middle_contact_fraction": grasp.opposing_contacts[:, 1],
            "outer_contact_fraction": grasp.opposing_contacts[:, 2],
            "palm_support_fraction": grasp.palm_supported,
            "joint_limit_occupancy": joint_limit_occupancy(
                cast("ManagerBasedRlEnv", self._env), self._action_name
            ),
        }
        for name, value in values.items():
            self._sum_fields[name][ids] += np.asarray(value)[ids]
        self._steps[ids] += 1

    def record_pre_reset(self, env_ids: np.ndarray) -> None:
        self._accumulate(env_ids)
        if self._written:
            return
        env_id = int(env_ids[0])
        steps = max(int(self._steps[env_id]), 1)
        elapsed_seconds = float(self._steps[env_id] * self._env.step_dt)
        net_rotation = float(self._rotation.held_angle[env_id])
        env = cast(Any, self._env)
        report: dict[str, Any] = {
            "schema_version": 1,
            "case_id": self._case_id,
            "split": self._split,
            "seed": int(env.cfg.seed),
            "grasp_index": self._grasp_index,
            "time_to_fall_seconds": elapsed_seconds,
            "net_committed_axis_rotation": net_rotation,
            "accumulated_rotation": float(self._rotation.frontier_angle[env_id]),
            "directional_speed": net_rotation / max(elapsed_seconds, self._env.step_dt),
            "direction_reversals": int(self._rotation.direction_reversals[env_id]),
            "terminated": bool(env.reset_terminated[env_id]),
            "timed_out": bool(env.reset_time_outs[env_id]),
        }
        report.update(
            {name: float(values[env_id] / steps) for name, values in self._sum_fields.items()}
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._written = True

    def record_post_step(self) -> None:
        active = np.flatnonzero(~self._env.reset_buf)
        self._accumulate(active)


__all__ = [
    "LeapEvaluationRecorder",
    "load_evaluation_manifest",
    "summarize_episode_reports",
]
