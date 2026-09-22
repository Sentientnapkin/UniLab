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

HORA_CHECKPOINT_SELECTION_DEFAULTS: dict[str, dict[str, float]] = {
    "stabilize": {
        "minimum_split_time_to_fall": 16.0,
        "minimum_horizon_fraction": 0.8,
        "minimum_valid_grasp_fraction": 0.7,
        "minimum_middle_contact_fraction": 0.4,
        "maximum_palm_support_fraction": 0.02,
        "maximum_joint_limit_occupancy": 0.02,
    },
    "rotate": {
        "minimum_split_time_to_fall": 16.0,
        "minimum_horizon_fraction": 0.8,
        "minimum_valid_grasp_fraction": 0.65,
        "minimum_middle_contact_fraction": 0.25,
        "minimum_net_rotation": 0.0,
        "minimum_directional_speed": 0.0,
        "maximum_palm_support_fraction": 0.02,
        "maximum_joint_limit_occupancy": 0.02,
    },
    "final": {
        "minimum_split_time_to_fall": 16.0,
        "minimum_horizon_fraction": 0.8,
        "minimum_valid_grasp_fraction": 0.65,
        "minimum_middle_contact_fraction": 0.25,
        "minimum_net_rotation": 6.0,
        "minimum_directional_speed": 0.25,
        "maximum_direction_reversals": 10.0,
        "maximum_palm_support_fraction": 0.02,
        "maximum_joint_limit_occupancy": 0.02,
    },
}


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


def rank_hora_checkpoint_evaluations(
    evaluations: Sequence[tuple[int, str | Path, Mapping[str, Any]]],
    *,
    stage: str,
    thresholds: Mapping[str, float] | None = None,
    horizon_seconds: float = 20.0,
) -> dict[str, Any]:
    """Apply explicit gates and rank deterministic HORA checkpoint reports."""
    if stage not in HORA_CHECKPOINT_SELECTION_DEFAULTS:
        raise ValueError(f"unknown HORA checkpoint-selection stage {stage!r}")
    if not evaluations:
        raise ValueError("at least one checkpoint evaluation is required")
    configured = dict(HORA_CHECKPOINT_SELECTION_DEFAULTS[stage])
    if thresholds is not None:
        unknown = set(thresholds) - set(configured)
        if unknown:
            raise ValueError(f"unknown HORA checkpoint thresholds: {sorted(unknown)}")
        configured.update({name: float(value) for name, value in thresholds.items()})

    rows: list[dict[str, Any]] = []
    for checkpoint, path, report in evaluations:
        overall = cast(Mapping[str, Any], report["overall"])
        overall_metrics = cast(Mapping[str, Any], overall["metrics"])
        by_split = cast(Mapping[str, Any], report["by_split"])

        def median(name: str) -> float:
            return float(cast(Mapping[str, Any], overall_metrics[name])["median"])

        split_time_to_fall = {
            str(split): float(
                cast(Mapping[str, Any], cast(Mapping[str, Any], summary)["metrics"])[
                    "time_to_fall_seconds"
                ]["median"]
            )
            for split, summary in by_split.items()
        }
        episodes = cast(Sequence[Mapping[str, Any]], report["episodes"])
        horizon_fraction = float(
            np.mean(
                [
                    float(episode["time_to_fall_seconds"]) >= horizon_seconds - 1.0e-6
                    for episode in episodes
                ]
            )
        )
        metrics = {
            "minimum_split_time_to_fall": min(split_time_to_fall.values()),
            "horizon_fraction": horizon_fraction,
            "valid_grasp_fraction": median("valid_grasp_fraction"),
            "middle_contact_fraction": median("middle_contact_fraction"),
            "palm_support_fraction": median("palm_support_fraction"),
            "joint_limit_occupancy": median("joint_limit_occupancy"),
            "net_rotation": median("net_committed_axis_rotation"),
            "directional_speed": median("directional_speed"),
            "direction_reversals": median("direction_reversals"),
            "object_linear_velocity": median("object_linear_velocity"),
            "estimated_torque": median("estimated_torque"),
        }
        contact_sum = metrics["valid_grasp_fraction"] + metrics["middle_contact_fraction"]
        metrics["contact_retention_harmonic_mean"] = (
            0.0
            if contact_sum <= 0.0
            else 2.0
            * metrics["valid_grasp_fraction"]
            * metrics["middle_contact_fraction"]
            / contact_sum
        )
        failures: list[str] = []
        lower_bounds = {
            "minimum_split_time_to_fall": "minimum_split_time_to_fall",
            "minimum_horizon_fraction": "horizon_fraction",
            "minimum_valid_grasp_fraction": "valid_grasp_fraction",
            "minimum_middle_contact_fraction": "middle_contact_fraction",
            "minimum_net_rotation": "net_rotation",
            "minimum_directional_speed": "directional_speed",
        }
        upper_bounds = {
            "maximum_palm_support_fraction": "palm_support_fraction",
            "maximum_joint_limit_occupancy": "joint_limit_occupancy",
            "maximum_direction_reversals": "direction_reversals",
        }
        for threshold_name, metric_name in lower_bounds.items():
            if threshold_name in configured and metrics[metric_name] < configured[threshold_name]:
                failures.append(
                    f"{metric_name}={metrics[metric_name]:.6g} < {configured[threshold_name]:.6g}"
                )
        for threshold_name, metric_name in upper_bounds.items():
            if threshold_name in configured and metrics[metric_name] > configured[threshold_name]:
                failures.append(
                    f"{metric_name}={metrics[metric_name]:.6g} > {configured[threshold_name]:.6g}"
                )

        if stage == "stabilize":
            rank_key = (
                not failures,
                metrics["horizon_fraction"],
                metrics["minimum_split_time_to_fall"],
                metrics["contact_retention_harmonic_mean"],
                metrics["valid_grasp_fraction"],
                metrics["middle_contact_fraction"],
                -metrics["object_linear_velocity"],
                -metrics["estimated_torque"],
            )
        else:
            rank_key = (
                not failures,
                metrics["horizon_fraction"],
                metrics["minimum_split_time_to_fall"],
                metrics["net_rotation"],
                metrics["directional_speed"],
                metrics["valid_grasp_fraction"],
                metrics["middle_contact_fraction"],
                -metrics["direction_reversals"],
            )
        rows.append(
            {
                "checkpoint": int(checkpoint),
                "evaluation_path": str(Path(path).resolve()),
                "passed": not failures,
                "gate_failures": failures,
                "metrics": metrics,
                "split_time_to_fall": split_time_to_fall,
                "_rank_key": rank_key,
            }
        )

    rows.sort(key=lambda row: cast(tuple[Any, ...], row["_rank_key"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        del row["_rank_key"]
    return {
        "schema_version": 1,
        "stage": stage,
        "thresholds": configured,
        "recommended_checkpoint": rows[0]["checkpoint"],
        "recommended_passed": rows[0]["passed"],
        "rankings": rows,
    }


FINGER_JOINT_SLICES = {
    "index": slice(0, 4),
    "middle": slice(4, 8),
    "outer": slice(8, 12),
    "thumb": slice(12, 16),
}


def summarize_hora_rollout_audits(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize per-finger HORA rollout diagnostics across fixed episodes."""
    if not reports:
        raise ValueError("at least one rollout audit report is required")
    metric_names = (
        "raw_contact_fraction",
        "graced_contact_fraction",
        "policy_action_rms",
        "target_step_rms_rad",
        "target_travel_rad",
        "target_span_rad",
        "measured_step_rms_rad",
        "measured_travel_rad",
        "measured_span_rad",
        "tracking_error_rms_rad",
        "estimated_torque_rms",
        "estimated_torque_saturation_fraction",
        "joint_limit_occupancy",
        "raw_contact_transitions",
    )

    fingers: dict[str, Any] = {}
    for finger in FINGER_JOINT_SLICES:
        rows = [cast(Mapping[str, Any], report["fingers"])[finger] for report in reports]
        fingers[finger] = {
            "episodes_with_initial_raw_contact": int(
                sum(bool(row["initial_raw_contact"]) for row in rows)
            ),
            "metrics": {
                name: {
                    "median": float(statistics.median(float(row[name]) for row in rows)),
                    "mean": float(statistics.fmean(float(row[name]) for row in rows)),
                }
                for name in metric_names
            },
        }
    return {"episodes": len(reports), "fingers": fingers}


class LeapHoraRolloutAuditRecorder(RecorderTerm):
    """Record per-finger policy, target-tracking, torque, and contact diagnostics."""

    _ALLOWED_PARAMS = frozenset(
        {"output_path", "case_id", "split", "grasp_index", "state_term_name", "action_name"}
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
        if isinstance(grasp_index, bool) or not isinstance(grasp_index, int) or grasp_index < 0:
            raise TypeError(f"{term} grasp_index must be a nonnegative integer")
        self._grasp_index = grasp_index

        state_name = _name(term, "state_term_name", cfg.params.get("state_term_name"))
        state_cfg = env.termination_manager.get_term_cfg(state_name)
        if not isinstance(state_cfg.func, LeapGraspTermination):
            raise TypeError(f"{term} state term must be LeapGraspTermination")
        self._grasp = state_cfg.func
        self._state = self._grasp.observation

        action_name = _name(term, "action_name", cfg.params.get("action_name"))
        action = env.action_manager.get_term(action_name)
        if not isinstance(action, LeapHoraPositionAction):
            raise TypeError(f"{term} action must be LeapHoraPositionAction")
        self._action = action

        shape = (env.num_envs, action.action_dim)
        self._sum_action_sq = np.zeros(shape, dtype=np.float64)
        self._sum_target_step_sq = np.zeros(shape, dtype=np.float64)
        self._sum_target_travel = np.zeros(shape, dtype=np.float64)
        self._sum_measured_step_sq = np.zeros(shape, dtype=np.float64)
        self._sum_measured_travel = np.zeros(shape, dtype=np.float64)
        self._sum_tracking_error_sq = np.zeros(shape, dtype=np.float64)
        self._sum_torque_sq = np.zeros(shape, dtype=np.float64)
        self._torque_saturated_steps = np.zeros(shape, dtype=np.int64)
        self._joint_limit_steps = np.zeros(shape, dtype=np.int64)
        self._target_min = np.array(action.target, dtype=np.float64, copy=True)
        self._target_max = np.array(action.target, dtype=np.float64, copy=True)
        self._measured_min = np.array(self._state.dof_pos, dtype=np.float64, copy=True)
        self._measured_max = np.array(self._state.dof_pos, dtype=np.float64, copy=True)
        self._previous_target = np.array(action.target, dtype=np.float64, copy=True)
        self._previous_measured = np.array(self._state.dof_pos, dtype=np.float64, copy=True)
        self._initial_target = np.array(action.target, dtype=np.float64, copy=True)
        self._initial_measured = np.array(self._state.dof_pos, dtype=np.float64, copy=True)

        self._raw_contact_steps = np.zeros((env.num_envs, 4), dtype=np.int64)
        self._graced_contact_steps = np.zeros((env.num_envs, 4), dtype=np.int64)
        self._raw_contact_transitions = np.zeros((env.num_envs, 4), dtype=np.int64)
        self._initial_raw_contact = np.zeros((env.num_envs, 4), dtype=np.bool_)
        self._previous_raw_contact = np.zeros((env.num_envs, 4), dtype=np.bool_)
        self._seen_contact = np.zeros(env.num_envs, dtype=np.bool_)
        self._steps = np.zeros(env.num_envs, dtype=np.int64)
        self._written = False

    def _contacts(self) -> tuple[np.ndarray, np.ndarray]:
        raw = np.column_stack((self._grasp.raw_opposing_contacts, self._grasp.raw_thumb_contact))
        graced = np.column_stack((self._grasp.opposing_contacts, self._grasp.thumb_contact))
        return raw, graced

    def _accumulate(self, ids: np.ndarray) -> None:
        if not len(ids):
            return
        target = np.asarray(self._action.target, dtype=np.float64)
        measured = np.asarray(self._state.dof_pos, dtype=np.float64)
        action = np.asarray(self._action.raw_action, dtype=np.float64)
        torque = np.asarray(self._state.torques, dtype=np.float64)
        # Recorder construction precedes the episode reset that installs a
        # cached grasp. Establish each environment's baseline from its first
        # post-reset sample so the reset jump is not counted as policy motion.
        unseen = ids[~self._seen_contact[ids]]
        if len(unseen):
            self._previous_target[unseen] = target[unseen]
            self._previous_measured[unseen] = measured[unseen]
            self._initial_target[unseen] = target[unseen]
            self._initial_measured[unseen] = measured[unseen]
            self._target_min[unseen] = target[unseen]
            self._target_max[unseen] = target[unseen]
            self._measured_min[unseen] = measured[unseen]
            self._measured_max[unseen] = measured[unseen]
        target_step = target - self._previous_target
        measured_step = measured - self._previous_measured
        tracking_error = target - measured
        span = self._action.ctrl_upper - self._action.ctrl_lower
        near_limit = (measured <= self._action.ctrl_lower + 0.02 * span) | (
            measured >= self._action.ctrl_upper - 0.02 * span
        )

        self._sum_action_sq[ids] += np.square(action[ids])
        self._sum_target_step_sq[ids] += np.square(target_step[ids])
        self._sum_target_travel[ids] += np.abs(target_step[ids])
        self._sum_measured_step_sq[ids] += np.square(measured_step[ids])
        self._sum_measured_travel[ids] += np.abs(measured_step[ids])
        self._sum_tracking_error_sq[ids] += np.square(tracking_error[ids])
        self._sum_torque_sq[ids] += np.square(torque[ids])
        self._torque_saturated_steps[ids] += np.abs(torque[ids]) >= 0.5 - 1.0e-6
        self._joint_limit_steps[ids] += near_limit[ids]
        self._target_min[ids] = np.minimum(self._target_min[ids], target[ids])
        self._target_max[ids] = np.maximum(self._target_max[ids], target[ids])
        self._measured_min[ids] = np.minimum(self._measured_min[ids], measured[ids])
        self._measured_max[ids] = np.maximum(self._measured_max[ids], measured[ids])

        raw, graced = self._contacts()
        if len(unseen):
            self._initial_raw_contact[unseen] = raw[unseen]
            self._previous_raw_contact[unseen] = raw[unseen]
            self._seen_contact[unseen] = True
        self._raw_contact_steps[ids] += raw[ids]
        self._graced_contact_steps[ids] += graced[ids]
        self._raw_contact_transitions[ids] += raw[ids] != self._previous_raw_contact[ids]
        self._previous_raw_contact[ids] = raw[ids]
        self._previous_target[ids] = target[ids]
        self._previous_measured[ids] = measured[ids]
        self._steps[ids] += 1

    @staticmethod
    def _joint_values(values: np.ndarray, env_id: int, steps: int, *, root: bool) -> list[float]:
        result = values[env_id] / steps
        if root:
            result = np.sqrt(result)
        return [float(value) for value in result]

    @staticmethod
    def _finger_mean(values: Sequence[float], joint_slice: slice) -> float:
        return float(statistics.fmean(values[joint_slice]))

    def _finger_report(
        self, env_id: int, steps: int, finger_index: int, joint_slice: slice
    ) -> dict[str, Any]:
        action_rms = self._joint_values(self._sum_action_sq, env_id, steps, root=True)
        target_step_rms = self._joint_values(self._sum_target_step_sq, env_id, steps, root=True)
        measured_step_rms = self._joint_values(self._sum_measured_step_sq, env_id, steps, root=True)
        tracking_rms = self._joint_values(self._sum_tracking_error_sq, env_id, steps, root=True)
        torque_rms = self._joint_values(self._sum_torque_sq, env_id, steps, root=True)
        target_travel = [float(value) for value in self._sum_target_travel[env_id]]
        measured_travel = [float(value) for value in self._sum_measured_travel[env_id]]
        target_span = [
            float(value) for value in self._target_max[env_id] - self._target_min[env_id]
        ]
        measured_span = [
            float(value) for value in self._measured_max[env_id] - self._measured_min[env_id]
        ]
        torque_saturation = [float(value) for value in self._torque_saturated_steps[env_id] / steps]
        joint_limits = [float(value) for value in self._joint_limit_steps[env_id] / steps]
        return {
            "initial_raw_contact": bool(self._initial_raw_contact[env_id, finger_index]),
            "raw_contact_fraction": float(self._raw_contact_steps[env_id, finger_index] / steps),
            "graced_contact_fraction": float(
                self._graced_contact_steps[env_id, finger_index] / steps
            ),
            "raw_contact_transitions": int(self._raw_contact_transitions[env_id, finger_index]),
            "policy_action_rms": self._finger_mean(action_rms, joint_slice),
            "target_step_rms_rad": self._finger_mean(target_step_rms, joint_slice),
            "target_travel_rad": self._finger_mean(target_travel, joint_slice),
            "target_span_rad": self._finger_mean(target_span, joint_slice),
            "measured_step_rms_rad": self._finger_mean(measured_step_rms, joint_slice),
            "measured_travel_rad": self._finger_mean(measured_travel, joint_slice),
            "measured_span_rad": self._finger_mean(measured_span, joint_slice),
            "tracking_error_rms_rad": self._finger_mean(tracking_rms, joint_slice),
            "estimated_torque_rms": self._finger_mean(torque_rms, joint_slice),
            "estimated_torque_saturation_fraction": self._finger_mean(
                torque_saturation, joint_slice
            ),
            "joint_limit_occupancy": self._finger_mean(joint_limits, joint_slice),
            "per_joint": {
                "policy_action_rms": action_rms[joint_slice],
                "target_step_rms_rad": target_step_rms[joint_slice],
                "target_travel_rad": target_travel[joint_slice],
                "target_span_rad": target_span[joint_slice],
                "measured_step_rms_rad": measured_step_rms[joint_slice],
                "measured_travel_rad": measured_travel[joint_slice],
                "measured_span_rad": measured_span[joint_slice],
                "tracking_error_rms_rad": tracking_rms[joint_slice],
                "estimated_torque_rms": torque_rms[joint_slice],
                "estimated_torque_saturation_fraction": torque_saturation[joint_slice],
                "joint_limit_occupancy": joint_limits[joint_slice],
            },
        }

    def record_pre_reset(self, env_ids: np.ndarray) -> None:
        self._accumulate(env_ids)
        if self._written:
            return
        env_id = int(env_ids[0])
        steps = max(int(self._steps[env_id]), 1)
        joint_names = [
            self._action._entity.joint_names[int(index)] for index in self._action.joint_ids
        ]
        report = {
            "schema_version": 1,
            "case_id": self._case_id,
            "split": self._split,
            "grasp_index": self._grasp_index,
            "steps": int(self._steps[env_id]),
            "duration_seconds": float(self._steps[env_id] * self._env.step_dt),
            "joint_names": joint_names,
            "torque_saturation_definition": "abs(clipped PD torque estimate) >= 0.5 - 1e-6",
            "fingers": {
                name: self._finger_report(env_id, steps, finger_index, joint_slice)
                for finger_index, (name, joint_slice) in enumerate(FINGER_JOINT_SLICES.items())
            },
        }
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._written = True

    def record_post_step(self) -> None:
        active = np.flatnonzero(~self._env.reset_buf)
        self._accumulate(active)


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
        self._maximum_vertical_drop = np.zeros(env.num_envs, dtype=np.float64)
        self._maximum_lateral_drift = np.zeros(env.num_envs, dtype=np.float64)
        self._minimum_object_height = np.full(env.num_envs, np.inf, dtype=np.float64)
        self._maximum_pose_deviation = np.zeros(env.num_envs, dtype=np.float64)
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
        self._maximum_vertical_drop[ids] = np.maximum(
            self._maximum_vertical_drop[ids], grasp.vertical_drop[ids]
        )
        self._maximum_lateral_drift[ids] = np.maximum(
            self._maximum_lateral_drift[ids], grasp.lateral_drift[ids]
        )
        self._minimum_object_height[ids] = np.minimum(
            self._minimum_object_height[ids], state.object_pos[ids, 2]
        )
        pose_deviation = np.sum(np.square(state.dof_pos - state.init_pose), axis=1)
        self._maximum_pose_deviation[ids] = np.maximum(
            self._maximum_pose_deviation[ids], pose_deviation[ids]
        )
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
            "maximum_vertical_drop": float(self._maximum_vertical_drop[env_id]),
            "maximum_lateral_drift": float(self._maximum_lateral_drift[env_id]),
            "minimum_object_height": float(self._minimum_object_height[env_id]),
            "maximum_pose_deviation_l2": float(self._maximum_pose_deviation[env_id]),
            "termination_causes": {
                "vertical_escape": bool(
                    self._maximum_vertical_drop[env_id] >= self._grasp._termination_vertical_drop
                ),
                "lateral_escape": bool(
                    self._maximum_lateral_drift[env_id] >= self._grasp._termination_lateral_drift
                ),
                "absolute_height": bool(
                    self._minimum_object_height[env_id] < self._grasp._absolute_minimum_height
                ),
            },
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
    "FINGER_JOINT_SLICES",
    "HORA_CHECKPOINT_SELECTION_DEFAULTS",
    "LeapEvaluationRecorder",
    "LeapHoraRolloutAuditRecorder",
    "load_evaluation_manifest",
    "rank_hora_checkpoint_evaluations",
    "summarize_episode_reports",
    "summarize_hora_rollout_audits",
]
