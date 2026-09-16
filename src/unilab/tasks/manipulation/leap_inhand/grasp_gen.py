"""Stable-grasp collection terms and registry owner for the LEAP Hand."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.run_control import RunComplete
from unilab.dtype_config import get_global_dtype
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env
from unilab.managers import ManagerTermBase, ManagerTermBaseCfg, RecorderTerm, RecorderTermCfg

from .manager_terms import (
    LeapGraspTermination,
    LeapIncrementalPositionAction,
    LeapIncrementalPositionActionCfg,
)

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv
    from unilab.managers.action_manager import ActionManager
    from unilab.managers.termination_manager import TerminationManager

    class _GraspEnv(ManagerBasedRlEnv, Protocol):
        @property
        def common_step_counter(self) -> int: ...

        @property
        def episode_length_buf(self) -> np.ndarray: ...

        @property
        def action_manager(self) -> ActionManager: ...

        @property
        def termination_manager(self) -> TerminationManager: ...

        @property
        def reset_terminated(self) -> np.ndarray: ...

        @property
        def reset_time_outs(self) -> np.ndarray: ...

        @property
        def extras(self) -> dict[str, Any]: ...


def _name(term: str, field: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{term} {field} must be a non-empty string")
    return value


def _real(term: str, field: str, value: Any, *, minimum: float = 0.0) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{term} {field} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{term} {field} must be finite and at least {minimum}")
    return result


def _positive_int(term: str, field: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{term} {field} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{term} {field} must be positive")
    return result


def _bool(term: str, field: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{term} {field} must be boolean")
    return bool(value)


@dataclass(kw_only=True)
class LeapGraspHoldActionCfg(LeapIncrementalPositionActionCfg):
    """LEAP action config that holds the joint pose sampled by the reset event."""

    reset_target_from_joint_state: bool = True

    def build(self, env: ManagerBasedRlEnv) -> LeapGraspHoldAction:
        return LeapGraspHoldAction(self, env)


class LeapGraspHoldAction(LeapIncrementalPositionAction):
    """Hold each environment's sampled pose instead of the canonical reset command."""


class LeapGraspQualityTermination(ManagerTermBase):
    """Reject candidates that are not motionless, valid grasps after settling."""

    _ALLOWED_PARAMS = frozenset(
        {
            "state_term_name",
            "minimum_validation_seconds",
            "maximum_linear_speed",
            "maximum_angular_speed",
            "enabled",
        }
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: _GraspEnv):
        super().__init__(env)
        term = type(self).__name__
        unexpected = set(cfg.params) - self._ALLOWED_PARAMS
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        self._state_term_name = _name(term, "state_term_name", cfg.params.get("state_term_name"))
        # TerminationManager constructs class terms before assigning itself to
        # the environment. Resolve the sibling grasp term lazily once manager
        # construction has completed.
        self._grasp: LeapGraspTermination | None = None
        self._minimum_validation_seconds = _real(
            term,
            "minimum_validation_seconds",
            cfg.params.get("minimum_validation_seconds"),
        )
        self._maximum_linear_speed = _real(
            term, "maximum_linear_speed", cfg.params.get("maximum_linear_speed")
        )
        self._maximum_angular_speed = _real(
            term, "maximum_angular_speed", cfg.params.get("maximum_angular_speed")
        )
        self._enabled = _bool(term, "enabled", cfg.params.get("enabled"))
        self.linear_speed = np.zeros(env.num_envs, dtype=get_global_dtype())
        self.angular_speed = np.zeros(env.num_envs, dtype=get_global_dtype())
        self.validation_started = np.zeros(env.num_envs, dtype=np.bool_)
        self.stable = np.zeros(env.num_envs, dtype=np.bool_)
        self._disabled = np.zeros(env.num_envs, dtype=np.bool_)
        self._last_counter = int(env.common_step_counter)

    @property
    def grasp(self) -> LeapGraspTermination:
        if self._grasp is not None:
            return self._grasp
        env = cast("_GraspEnv", self._env)
        grasp = env.termination_manager.get_term_cfg(self._state_term_name).func
        if not isinstance(grasp, LeapGraspTermination):
            raise TypeError(
                f"{type(self).__name__} termination term {self._state_term_name!r} must be "
                "LeapGraspTermination, "
                f"got {type(grasp).__name__}"
            )
        self._grasp = grasp
        return grasp

    @property
    def last_counter(self) -> int:
        return self._last_counter

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.linear_speed[ids] = 0.0
        self.angular_speed[ids] = 0.0
        self.validation_started[ids] = False
        self.stable[ids] = False
        self._last_counter = int(cast("_GraspEnv", self._env).common_step_counter)

    def __call__(self, env: _GraspEnv, **params: Any) -> np.ndarray:
        del params
        if self.grasp.last_counter != int(env.common_step_counter):
            raise RuntimeError(
                f"{type(self).__name__} requires its LEAP grasp state term to run first"
            )
        state = self.grasp.observation
        self.linear_speed[:] = np.linalg.norm(state.object_linvel, axis=1)
        self.angular_speed[:] = np.linalg.norm(state.object_angvel, axis=1)
        self.validation_started[:] = (
            env.episode_length_buf * env.step_dt >= self._minimum_validation_seconds
        )
        self.stable[:] = (
            self.grasp.valid_grasp
            & (self.linear_speed <= self._maximum_linear_speed)
            & (self.angular_speed <= self._maximum_angular_speed)
        )
        self._last_counter = int(env.common_step_counter)
        if not self._enabled:
            return self._disabled
        return self.validation_started & ~self.stable


class LeapGraspQualityMetric(ManagerTermBase):
    """Expose one cached stable-grasp condition through MetricsManager."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: _GraspEnv):
        super().__init__(env)
        term = type(self).__name__
        unexpected = set(cfg.params) - {"quality_term_name", "condition"}
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        quality_name = _name(term, "quality_term_name", cfg.params.get("quality_term_name"))
        quality = env.termination_manager.get_term_cfg(quality_name).func
        if not isinstance(quality, LeapGraspQualityTermination):
            raise TypeError(
                f"{term} termination term {quality_name!r} must be "
                f"LeapGraspQualityTermination, got {type(quality).__name__}"
            )
        condition = _name(term, "condition", cfg.params.get("condition"))
        values = {
            "valid_grasp": quality.grasp.valid_grasp,
            "stable": quality.stable,
            "linear_speed": quality.linear_speed,
            "angular_speed": quality.angular_speed,
        }
        try:
            self._value = values[condition]
        except KeyError:
            raise ValueError(
                f"{term} condition must be one of {sorted(values)}, got {condition!r}"
            ) from None
        self._quality = quality
        self._quality_name = quality_name

    def __call__(self, env: _GraspEnv, **params: Any) -> np.ndarray:
        del params
        if self._quality.last_counter != int(env.common_step_counter):
            raise RuntimeError(
                f"{type(self).__name__} term {self._quality_name!r} was not computed for "
                f"control step {env.common_step_counter}"
            )
        return np.asarray(self._value, dtype=get_global_dtype())


class LeapGraspRecorder(RecorderTerm):
    """Collect stable timeout states into a diversity-filtered 23-D cache.

    A validated seed cache may be included in the output and used as the
    novelty reference.  A generated row is considered a near-duplicate only
    when it is simultaneously close in enabled joint, object-position, and
    cylinder-axis criteria.  Cylinder-axis distance deliberately ignores spin
    around the cylinder's symmetry axis.
    """

    _ALLOWED_PARAMS = frozenset(
        {
            "quality_term_name",
            "output_path",
            "collection_target",
            "auto_save",
            "initial_cache_path",
            "minimum_joint_rms_rad",
            "minimum_position_distance_m",
            "minimum_axis_angle_deg",
        }
    )

    def __init__(self, cfg: RecorderTermCfg, env: _GraspEnv):
        super().__init__(cfg, env)
        term = type(self).__name__
        unexpected = set(cfg.params) - self._ALLOWED_PARAMS
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        quality_name = _name(term, "quality_term_name", cfg.params.get("quality_term_name"))
        quality = env.termination_manager.get_term_cfg(quality_name).func
        if not isinstance(quality, LeapGraspQualityTermination):
            raise TypeError(
                f"{term} termination term {quality_name!r} must be "
                f"LeapGraspQualityTermination, got {type(quality).__name__}"
            )
        self._quality = quality
        output = Path(_name(term, "output_path", cfg.params.get("output_path")))
        self._output_path = output if output.is_absolute() else Path(ASSETS_ROOT_PATH) / output
        self._target = _positive_int(term, "collection_target", cfg.params.get("collection_target"))
        self._auto_save = _bool(term, "auto_save", cfg.params.get("auto_save"))
        self._minimum_joint_rms_rad = _real(
            term,
            "minimum_joint_rms_rad",
            cfg.params.get("minimum_joint_rms_rad", 0.0),
        )
        self._minimum_position_distance_m = _real(
            term,
            "minimum_position_distance_m",
            cfg.params.get("minimum_position_distance_m", 0.0),
        )
        self._minimum_axis_angle_rad = np.deg2rad(
            _real(
                term,
                "minimum_axis_angle_deg",
                cfg.params.get("minimum_axis_angle_deg", 0.0),
            )
        )
        initial_cache_value = cfg.params.get("initial_cache_path")
        initial_states = np.empty((0, 23), dtype=np.float32)
        if initial_cache_value is not None:
            initial_path = Path(_name(term, "initial_cache_path", initial_cache_value))
            if not initial_path.is_absolute():
                initial_path = Path(ASSETS_ROOT_PATH) / initial_path
            if initial_path == self._output_path:
                raise ValueError(f"{term} initial_cache_path and output_path must differ")
            if not initial_path.is_file():
                raise FileNotFoundError(f"{term} initial cache does not exist: {initial_path}")
            initial_states = np.asarray(np.load(initial_path), dtype=np.float32)
            self._validate_rows(initial_states, label="initial cache")
            if initial_states.shape[0] > self._target:
                raise ValueError(
                    f"{term} initial cache has {initial_states.shape[0]} rows, exceeding "
                    f"collection_target {self._target}"
                )
        self._seed_count = int(initial_states.shape[0])
        self._saved_states: list[np.ndarray] = []
        if self._seed_count:
            self._saved_states.append(initial_states.copy())
        self._novelty_rows = initial_states.copy()
        self._rejected_near_duplicates = 0
        self._generated_count = 0
        self._cache_saved = False
        self._target_notified = False

    @property
    def total_saved_grasps(self) -> int:
        return int(sum(states.shape[0] for states in self._saved_states))

    @property
    def cache_saved(self) -> bool:
        return self._cache_saved

    @property
    def output_path(self) -> Path:
        return self._output_path

    def _log(self, name: str, value: float) -> None:
        env = cast("_GraspEnv", self._env)
        env.extras.setdefault("log", {})[name] = value

    @staticmethod
    def _validate_rows(rows: np.ndarray, *, label: str) -> None:
        if rows.ndim != 2 or rows.shape[1] != 23:
            raise ValueError(
                f"LeapGraspRecorder {label} must have shape (N, 23), got {rows.shape}"
            )
        if not np.isfinite(rows).all():
            raise ValueError(f"LeapGraspRecorder {label} contains NaN or Inf")
        if rows.shape[0] and np.any(np.linalg.norm(rows[:, 19:23], axis=1) <= 1.0e-8):
            raise ValueError(f"LeapGraspRecorder {label} contains a zero-norm quaternion")

    @staticmethod
    def _cylinder_axes(rows: np.ndarray) -> np.ndarray:
        quat = np.asarray(rows[:, 19:23], dtype=np.float64)
        quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        w, x, y, z = quat.T
        return np.stack(
            (
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ),
            axis=1,
        )

    def _filter_novel(self, rows: np.ndarray) -> np.ndarray:
        thresholds_enabled = (
            self._minimum_joint_rms_rad > 0.0
            or self._minimum_position_distance_m > 0.0
            or self._minimum_axis_angle_rad > 0.0
        )
        if not thresholds_enabled:
            return rows

        accepted: list[np.ndarray] = []
        references = self._novelty_rows
        for row in rows:
            if references.shape[0] == 0:
                is_duplicate = False
            else:
                near = np.ones(references.shape[0], dtype=np.bool_)
                if self._minimum_joint_rms_rad > 0.0:
                    joint_rms = np.sqrt(np.mean(np.square(references[:, :16] - row[:16]), axis=1))
                    near &= joint_rms < self._minimum_joint_rms_rad
                if self._minimum_position_distance_m > 0.0:
                    position_distance = np.linalg.norm(references[:, 16:19] - row[16:19], axis=1)
                    near &= position_distance < self._minimum_position_distance_m
                if self._minimum_axis_angle_rad > 0.0:
                    reference_axes = self._cylinder_axes(references)
                    candidate_axis = self._cylinder_axes(row[None, :])[0]
                    alignment = np.abs(reference_axes @ candidate_axis)
                    axis_angle = np.arccos(np.clip(alignment, 0.0, 1.0))
                    near &= axis_angle < self._minimum_axis_angle_rad
                is_duplicate = bool(np.any(near))
            if is_duplicate:
                self._rejected_near_duplicates += 1
                continue
            accepted.append(row)
            references = np.concatenate((references, row[None, :]), axis=0)

        if not accepted:
            return np.empty((0, 23), dtype=np.float32)
        novel = np.asarray(accepted, dtype=np.float32)
        self._novelty_rows = references
        return novel

    def _save_cache(self, *, force: bool = False) -> None:
        if self._cache_saved:
            return
        total = self.total_saved_grasps
        if (not force and total < self._target) or total == 0:
            return
        all_states = np.concatenate(self._saved_states, axis=0).astype(np.float32)
        all_states = all_states[: self._target]
        self._validate_rows(all_states, label="collected cache")
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._output_path, all_states)
        self._cache_saved = True
        self._log("grasp_cache/saved", 1.0)
        self._log("grasp_cache/num_states", float(all_states.shape[0]))

    def _stop_collection(self) -> None:
        if self._target_notified or self.total_saved_grasps < self._target:
            return
        total = self.total_saved_grasps
        self._target_notified = True
        self._log("grasp/target_reached", 1.0)
        print(
            "[LeapInhandRotationGrasp] Grasp collection target reached "
            f"({total}/{self._target}). Collection completed."
        )
        raise RunComplete(
            reason="grasp_collection_target_reached",
            summary={
                "collected_grasps": total,
                "saved_grasps": min(total, self._target),
                "grasp_collection_target": self._target,
                "seed_grasps": self._seed_count,
                "generated_grasps": self._generated_count,
                "rejected_near_duplicates": self._rejected_near_duplicates,
            },
        )

    def record_pre_reset(self, env_ids: np.ndarray) -> None:
        env = cast("_GraspEnv", self._env)
        ids = np.asarray(env_ids, dtype=np.intp)
        success = env.reset_time_outs[ids] & ~env.reset_terminated[ids] & self._quality.stable[ids]
        success_ids = ids[np.flatnonzero(success)]
        if success_ids.size == 0:
            return
        if self._quality.last_counter != int(env.common_step_counter):
            raise RuntimeError(
                f"{type(self).__name__} quality state was not computed for control step "
                f"{env.common_step_counter}"
            )
        observation = self._quality.grasp.observation
        rows = np.concatenate(
            (
                observation.dof_pos[success_ids],
                observation.object_pos[success_ids],
                observation.object_quat[success_ids],
            ),
            axis=1,
            dtype=np.float32,
        )
        if rows.shape != (success_ids.size, 23):
            raise ValueError(
                f"{type(self).__name__} expected collected rows shape "
                f"({success_ids.size}, 23), got {rows.shape}"
            )
        self._validate_rows(rows, label="candidate rows")
        rows = self._filter_novel(rows)
        if rows.shape[0]:
            remaining = self._target - self.total_saved_grasps
            rows = rows[:remaining]
            self._saved_states.append(rows)
            self._generated_count += int(rows.shape[0])
        self._save_cache()
        self._log("grasp/cache_size", float(self.total_saved_grasps))
        self._log("grasp/generated_count", float(self._generated_count))
        self._log("grasp/rejected_near_duplicates", float(self._rejected_near_duplicates))
        self._stop_collection()

    def close(self) -> None:
        self._save_cache(force=self._auto_save)


registry.register_env_config("LeapInhandRotationGrasp", ManagerBasedRlEnvCfg)
registry.register_env("LeapInhandRotationGrasp", make_manager_based_rl_env, sim_backend="mujoco")


__all__ = [
    "LeapGraspHoldAction",
    "LeapGraspHoldActionCfg",
    "LeapGraspQualityMetric",
    "LeapGraspQualityTermination",
    "LeapGraspRecorder",
]
