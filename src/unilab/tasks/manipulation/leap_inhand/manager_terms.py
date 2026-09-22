"""Manager-Based terms for Leap in-hand object rotation.

Hydra owns the production task declaration.  These terms use only the public
Entity facade and the community manager lifecycle; they do not inspect backend
objects or physical state layouts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from etils import epath
from unisim.backend.base import DebugPrimitive

from unilab.assets import ASSETS_ROOT_PATH
from unilab.dtype_config import get_global_dtype
from unilab.managers import ActionTerm, ActionTermCfg, ManagerTermBase, ManagerTermBaseCfg
from unilab.utils.geometry import np_normalize_axis, np_quat_angular_velocity_from_pair
from unilab.utils.rotation import (
    np_quat_apply_batched,
    np_quat_from_angle_axis,
    np_quat_mul,
)

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv
    from unilab.managers.action_manager import ActionManager
    from unilab.managers.event_manager import EventManager
    from unilab.managers.observation_manager import ObservationManager
    from unilab.managers.reward_manager import RewardManager
    from unilab.managers.termination_manager import TerminationManager

    class _LeapEnv(ManagerBasedRlEnv, Protocol):
        @property
        def episode_length_buf(self) -> np.ndarray: ...

        @property
        def common_step_counter(self) -> int: ...

        @property
        def step_dt(self) -> float: ...

        @property
        def action_manager(self) -> ActionManager: ...

        @property
        def observation_manager(self) -> ObservationManager: ...

        @property
        def termination_manager(self) -> TerminationManager: ...

        @property
        def reward_manager(self) -> RewardManager: ...

        @property
        def event_manager(self) -> EventManager: ...


def _real(
    term: str,
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{term} {name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{term} {name} must be finite")
    if minimum is not None and (result <= minimum if strict_minimum else result < minimum):
        relation = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{term} {name} must be {relation} {minimum}")
    return result


def _name(term: str, name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{term} {name} must be a non-empty string")
    return value


def _names(term: str, name: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} {name} must be a sequence of names")
    result = tuple(_name(term, f"{name}[{index}]", item) for index, item in enumerate(value))
    if not result:
        raise ValueError(f"{term} {name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{term} {name} must not contain duplicate names")
    return result


def _positive_int(term: str, name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{term} {name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{term} {name} must be positive")
    return result


def _name_groups(term: str, name: str, value: Any) -> tuple[tuple[str, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} {name} must be a sequence of name sequences")
    groups = tuple(_names(term, f"{name}[{index}]", group) for index, group in enumerate(value))
    if not groups:
        raise ValueError(f"{term} {name} must not be empty")
    flat = tuple(item for group in groups for item in group)
    if len(set(flat)) != len(flat):
        raise ValueError(f"{term} {name} must not contain duplicate names")
    return groups


def _pair(term: str, name: str, value: Any) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} {name} must be a numeric (min, max) pair")
    if len(value) != 2:
        raise ValueError(f"{term} {name} must contain two values")
    lower = _real(term, f"{name}[0]", value[0])
    upper = _real(term, f"{name}[1]", value[1])
    if lower > upper:
        raise ValueError(f"{term} {name} lower bound {lower} exceeds upper bound {upper}")
    return lower, upper


def _scalar_schedule(
    term: str,
    name: str,
    value_name: str,
    value: Any,
    *,
    maximum: float | None = None,
) -> tuple[tuple[int, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} {name} must be a sequence of mappings")
    result: list[tuple[int, float]] = []
    for index, stage in enumerate(value):
        if not isinstance(stage, Mapping):
            raise TypeError(f"{term} {name}[{index}] must be a mapping")
        if set(stage) != {"step", value_name}:
            raise ValueError(f"{term} {name}[{index}] must contain only step and {value_name}")
        step = stage["step"]
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)):
            raise TypeError(f"{term} {name}[{index}].step must be an integer")
        step = int(step)
        if step < 0:
            raise ValueError(f"{term} {name}[{index}].step must be nonnegative")
        scalar = _real(
            term,
            f"{name}[{index}].{value_name}",
            stage[value_name],
            minimum=0.0,
        )
        if maximum is not None and scalar > maximum:
            raise ValueError(f"{term} {name}[{index}].{value_name} must be at most {maximum}")
        if result and step <= result[-1][0]:
            raise ValueError(f"{term} {name} steps must be strictly increasing")
        result.append((step, scalar))
    if not result or result[0][0] != 0:
        raise ValueError(f"{term} {name} must start at step 0")
    return tuple(result)


def _amplitude_schedule(term: str, value: Any) -> tuple[tuple[int, float], ...]:
    return _scalar_schedule(term, "gait_amplitude_schedule", "amplitude", value)


def _scheduled_amplitude(schedule: tuple[tuple[int, float], ...], step: int) -> float:
    if step <= schedule[0][0]:
        return schedule[0][1]
    for (start_step, start_value), (end_step, end_value) in zip(
        schedule, schedule[1:], strict=False
    ):
        if step <= end_step:
            fraction = (step - start_step) / (end_step - start_step)
            return start_value + fraction * (end_value - start_value)
    return schedule[-1][1]


def _cyclic_gait_reference(
    phase: np.ndarray,
    amplitude: float,
    thumb_amplitude: float,
    thumb_half_width: float,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    """Build opposing-finger cycles plus a narrow thumb micro-regrasp."""
    reference = np.zeros((phase.shape[0], 16), dtype=dtype)
    if amplitude == 0.0 and thumb_amplitude == 0.0:
        return reference
    if amplitude > 0.0:
        half_width = np.pi / 3.0
        for finger, center in enumerate((np.pi / 3.0, np.pi, 5.0 * np.pi / 3.0)):
            distance = (phase - center + np.pi) % (2.0 * np.pi) - np.pi
            active = np.abs(distance) < half_width
            normalized = np.clip(distance / half_width, -1.0, 1.0)
            release = np.where(active, 0.5 * (1.0 + np.cos(np.pi * normalized)), 0.0)
            sweep = np.where(active, release * np.sin(np.pi * normalized), 0.0)
            offset = 4 * finger
            reference[:, offset] = -0.25 * amplitude * release
            reference[:, offset + 1] = 0.5 * amplitude * sweep
            reference[:, offset + 2] = -amplitude * release
            reference[:, offset + 3] = -0.75 * amplitude * release

    if thumb_amplitude > 0.0:
        # The first 2π/3 boundary occurs after the initial grasp has settled and
        # every opposing-finger release pulse is zero. Keep this window shorter
        # than the contact grace period so the thumb briefly unloads and sweeps
        # without becoming an open grasp.
        center = 2.0 * np.pi / 3.0
        distance = (phase - center + np.pi) % (2.0 * np.pi) - np.pi
        active = np.abs(distance) < thumb_half_width
        normalized = np.clip(distance / thumb_half_width, -1.0, 1.0)
        release = np.where(active, 0.5 * (1.0 + np.cos(np.pi * normalized)), 0.0)
        sweep = np.where(active, release * np.sin(np.pi * normalized), 0.0)
        reference[:, 12] = 0.25 * thumb_amplitude * sweep
        reference[:, 13] = 0.5 * thumb_amplitude * sweep
        reference[:, 14] = -thumb_amplitude * release
        reference[:, 15] = -0.75 * thumb_amplitude * release
    return reference


def _env_ids(env: ManagerBasedRlEnv, env_ids: np.ndarray | slice | None) -> np.ndarray:
    if env_ids is None:
        return np.arange(env.num_envs, dtype=np.int32)
    if isinstance(env_ids, slice):
        return np.arange(env.num_envs, dtype=np.int32)[env_ids]
    return env_ids


def _resolve_grasp_cache(cache_path: str) -> epath.Path:
    path = epath.Path(cache_path)
    if path.is_absolute() or path.exists():
        return path
    return epath.Path(ASSETS_ROOT_PATH / cache_path)


@dataclass(kw_only=True)
class LeapIncrementalPositionActionCfg(ActionTermCfg):
    """Incremental position targets used by the original Leap policy."""

    actuator_names: tuple[str, ...] | list[str]
    action_scale: float
    raw_action_clip: tuple[float, float] | list[float]
    reset_target: tuple[float, ...] | list[float] | None = None
    reset_target_from_joint_state: bool = False
    gait_period: float = 2.0
    thumb_gait_amplitude: float = 0.0
    thumb_gait_duration: float = 0.12
    thumb_policy_target_offset_limit: float | None = None
    thumb_target_return_time_constant: float | None = None
    opposing_finger_policy_target_offset_limit: float | None = None
    opposing_finger_target_return_time_constant: float | None = None
    gait_amplitude_schedule: (
        tuple[dict[str, float | int], ...] | list[dict[str, float | int]] | None
    ) = None
    action_noise_std_schedule: (
        tuple[dict[str, float | int], ...] | list[dict[str, float | int]] | None
    ) = None
    action_delay_probability_schedule: (
        tuple[dict[str, float | int], ...] | list[dict[str, float | int]] | None
    ) = None

    def build(self, env: ManagerBasedRlEnv) -> LeapIncrementalPositionAction:
        return LeapIncrementalPositionAction(self, env)


class LeapIncrementalPositionAction(ActionTerm):
    """Integrate policy deltas and add a bounded cyclic regrasp reference."""

    cfg: LeapIncrementalPositionActionCfg
    _entity: Entity
    _raw_action: np.ndarray
    _clipped_action: np.ndarray
    _target: np.ndarray

    def __init__(self, cfg: LeapIncrementalPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        term = type(self).__name__
        if cfg.clip is not None:
            raise NotImplementedError(
                f"{term} does not support actuator-name clip; use raw_action_clip"
            )
        if isinstance(cfg.actuator_names, (str, bytes)) or not isinstance(
            cfg.actuator_names, (tuple, list)
        ):
            raise TypeError(f"{term} actuator_names must be a sequence of patterns")
        self._joint_ids, target_names = self._entity.find_joints_by_actuator_names(
            cfg.actuator_names
        )
        actuator_ids, actuator_names = self._entity.find_actuators(
            cfg.actuator_names, preserve_order=True
        )
        if len(self._joint_ids) != len(actuator_ids) or target_names != list(
            self._entity.joint_names[index] for index in self._joint_ids
        ):
            raise ValueError(f"{term} actuator-to-joint mapping is incomplete")
        if len(self._joint_ids) != 16:
            raise ValueError(f"{term} requires 16 hand actuators, got {len(self._joint_ids)}")
        if len(set(actuator_names)) != len(actuator_names):
            raise ValueError(f"{term} actuator selector resolved duplicate names")

        self._joint_ids_array = np.asarray(self._joint_ids, dtype=np.intp)
        self._joint_ids_array.setflags(write=False)
        local_actuator_ids = np.asarray(actuator_ids, dtype=np.intp)
        ranges = np.asarray(self._entity.data.actuator_ctrl_range, dtype=get_global_dtype())
        self._ctrl_lower = np.array(ranges[local_actuator_ids, 0], copy=True)
        self._ctrl_upper = np.array(ranges[local_actuator_ids, 1], copy=True)
        if np.any(self._ctrl_lower >= self._ctrl_upper):
            raise ValueError(f"{term} actuator control ranges must have lower < upper")

        self._scale = _real(term, "action_scale", cfg.action_scale, minimum=0.0)
        self._raw_clip = _pair(term, "raw_action_clip", cfg.raw_action_clip)
        self._gait_period = _real(
            term, "gait_period", cfg.gait_period, minimum=0.0, strict_minimum=True
        )
        self._thumb_gait_amplitude = _real(
            term, "thumb_gait_amplitude", cfg.thumb_gait_amplitude, minimum=0.0
        )
        self._thumb_gait_duration = _real(
            term,
            "thumb_gait_duration",
            cfg.thumb_gait_duration,
            minimum=0.0,
            strict_minimum=True,
        )
        if self._thumb_gait_duration > self._gait_period / 3.0:
            raise ValueError(f"{term} thumb_gait_duration must not exceed one third of gait_period")
        self._thumb_gait_half_width = np.pi * self._thumb_gait_duration / self._gait_period
        self._thumb_policy_target_offset_limit = (
            None
            if cfg.thumb_policy_target_offset_limit is None
            else _real(
                term,
                "thumb_policy_target_offset_limit",
                cfg.thumb_policy_target_offset_limit,
                minimum=0.0,
                strict_minimum=True,
            )
        )
        self._thumb_target_return_factor = (
            1.0
            if cfg.thumb_target_return_time_constant is None
            else np.exp(
                -env.step_dt
                / _real(
                    term,
                    "thumb_target_return_time_constant",
                    cfg.thumb_target_return_time_constant,
                    minimum=0.0,
                    strict_minimum=True,
                )
            )
        )
        self._opposing_finger_policy_target_offset_limit = (
            None
            if cfg.opposing_finger_policy_target_offset_limit is None
            else _real(
                term,
                "opposing_finger_policy_target_offset_limit",
                cfg.opposing_finger_policy_target_offset_limit,
                minimum=0.0,
                strict_minimum=True,
            )
        )
        self._opposing_finger_target_return_factor = (
            1.0
            if cfg.opposing_finger_target_return_time_constant is None
            else np.exp(
                -env.step_dt
                / _real(
                    term,
                    "opposing_finger_target_return_time_constant",
                    cfg.opposing_finger_target_return_time_constant,
                    minimum=0.0,
                    strict_minimum=True,
                )
            )
        )
        self._gait_schedule = (
            ((0, 0.0),)
            if cfg.gait_amplitude_schedule is None
            else _amplitude_schedule(term, cfg.gait_amplitude_schedule)
        )
        self._action_noise_std_schedule = (
            ((0, 0.0),)
            if cfg.action_noise_std_schedule is None
            else _scalar_schedule(
                term,
                "action_noise_std_schedule",
                "std",
                cfg.action_noise_std_schedule,
            )
        )
        self._action_delay_probability_schedule = (
            ((0, 0.0),)
            if cfg.action_delay_probability_schedule is None
            else _scalar_schedule(
                term,
                "action_delay_probability_schedule",
                "probability",
                cfg.action_delay_probability_schedule,
                maximum=1.0,
            )
        )
        dtype = get_global_dtype()
        if cfg.reset_target is None:
            reset_target = np.asarray(
                self._entity.data.default_joint_pos[0, self._joint_ids_array], dtype=dtype
            )
        else:
            try:
                reset_target = np.asarray(cfg.reset_target, dtype=dtype)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{term} reset_target must contain numeric values") from exc
            if reset_target.shape != (len(self._joint_ids),):
                raise ValueError(
                    f"{term} reset_target must have shape ({len(self._joint_ids)},), "
                    f"got {reset_target.shape}"
                )
            if not np.isfinite(reset_target).all():
                raise ValueError(f"{term} reset_target must contain only finite values")
        self._reset_target = np.clip(reset_target, self._ctrl_lower, self._ctrl_upper)
        if not isinstance(cfg.reset_target_from_joint_state, (bool, np.bool_)):
            raise TypeError(f"{term} reset_target_from_joint_state must be boolean")
        self._reset_target_from_joint_state = bool(cfg.reset_target_from_joint_state)
        self._raw_action = np.zeros((env.num_envs, len(self._joint_ids)), dtype=dtype)
        self._previous_policy_action = np.zeros_like(self._raw_action)
        self._effective_action = np.zeros_like(self._raw_action)
        self._command_noise = np.zeros_like(self._raw_action)
        self._command_delayed = np.zeros(env.num_envs, dtype=np.bool_)
        self._clipped_action = np.zeros_like(self._raw_action)
        self._episode_anchor = np.broadcast_to(
            self._reset_target, (env.num_envs, len(self._joint_ids))
        ).copy()
        self._policy_target = self._episode_anchor.copy()
        self._gait_reference = np.zeros_like(self._policy_target)
        self._target = self._policy_target.copy()

    @property
    def action_dim(self) -> int:
        return int(self._raw_action.shape[1])

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw_action

    @property
    def target(self) -> np.ndarray:
        return self._target

    @property
    def policy_target(self) -> np.ndarray:
        return self._policy_target

    @property
    def episode_anchor(self) -> np.ndarray:
        """Per-environment grasp about which persistent targets are bounded."""
        return self._episode_anchor

    @property
    def gait_reference(self) -> np.ndarray:
        return self._gait_reference

    @property
    def gait_amplitude(self) -> float:
        return _scheduled_amplitude(self._gait_schedule, int(self._env.common_step_counter))

    @property
    def thumb_gait_amplitude(self) -> float:
        return self._thumb_gait_amplitude

    @property
    def action_noise_std(self) -> float:
        return _scheduled_amplitude(
            self._action_noise_std_schedule, int(self._env.common_step_counter)
        )

    @property
    def action_delay_probability(self) -> float:
        return _scheduled_amplitude(
            self._action_delay_probability_schedule, int(self._env.common_step_counter)
        )

    @property
    def command_noise(self) -> np.ndarray:
        return self._command_noise

    @property
    def command_delayed(self) -> np.ndarray:
        return self._command_delayed

    @property
    def ctrl_lower(self) -> np.ndarray:
        return self._ctrl_lower

    @property
    def ctrl_upper(self) -> np.ndarray:
        return self._ctrl_upper

    @property
    def joint_ids(self) -> np.ndarray:
        return self._joint_ids_array

    def process_actions(self, actions: np.ndarray) -> None:
        if not isinstance(actions, np.ndarray):
            raise TypeError(f"expected np.ndarray actions, got {type(actions).__name__}")
        if actions.shape != self._raw_action.shape:
            raise ValueError(f"expected action shape {self._raw_action.shape}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("received NaN or Inf actions")
        self._raw_action[:] = actions
        delay_probability = self.action_delay_probability
        if delay_probability > 0.0:
            self._command_delayed[:] = self._env.rng.random(self._env.num_envs) < delay_probability
        else:
            self._command_delayed[:] = False
        self._effective_action[:] = actions
        self._effective_action[self._command_delayed] = self._previous_policy_action[
            self._command_delayed
        ]
        self._previous_policy_action[:] = actions
        noise_std = self.action_noise_std
        if noise_std > 0.0:
            self._command_noise[:] = self._env.rng.normal(
                0.0, noise_std, size=self._command_noise.shape
            )
            self._effective_action += self._command_noise
        else:
            self._command_noise[:] = 0.0
        np.clip(
            self._effective_action,
            self._raw_clip[0],
            self._raw_clip[1],
            out=self._clipped_action,
        )
        self._policy_target += self._scale * self._clipped_action
        np.clip(
            self._policy_target,
            self._ctrl_lower,
            self._ctrl_upper,
            out=self._policy_target,
        )
        self._policy_target[:, :12] = self._episode_anchor[:, :12] + (
            self._opposing_finger_target_return_factor
            * (self._policy_target[:, :12] - self._episode_anchor[:, :12])
        )
        opposing_lower = self._ctrl_lower[:12]
        opposing_upper = self._ctrl_upper[:12]
        if self._opposing_finger_policy_target_offset_limit is not None:
            opposing_lower = np.maximum(
                opposing_lower,
                self._episode_anchor[:, :12] - self._opposing_finger_policy_target_offset_limit,
            )
            opposing_upper = np.minimum(
                opposing_upper,
                self._episode_anchor[:, :12] + self._opposing_finger_policy_target_offset_limit,
            )
        np.clip(
            self._policy_target[:, :12],
            opposing_lower,
            opposing_upper,
            out=self._policy_target[:, :12],
        )
        self._policy_target[:, 12:] = self._episode_anchor[:, 12:] + (
            self._thumb_target_return_factor
            * (self._policy_target[:, 12:] - self._episode_anchor[:, 12:])
        )
        thumb_lower = self._ctrl_lower[12:]
        thumb_upper = self._ctrl_upper[12:]
        if self._thumb_policy_target_offset_limit is not None:
            thumb_lower = np.maximum(
                thumb_lower,
                self._episode_anchor[:, 12:] - self._thumb_policy_target_offset_limit,
            )
            thumb_upper = np.minimum(
                thumb_upper,
                self._episode_anchor[:, 12:] + self._thumb_policy_target_offset_limit,
            )
        np.clip(
            self._policy_target[:, 12:],
            thumb_lower,
            thumb_upper,
            out=self._policy_target[:, 12:],
        )
        steps = np.maximum(self._env.episode_length_buf, 0).astype(get_global_dtype())
        phase = steps * self._env.step_dt * (2.0 * np.pi / self._gait_period)
        self._gait_reference[:] = _cyclic_gait_reference(
            phase,
            self.gait_amplitude,
            self.thumb_gait_amplitude,
            self._thumb_gait_half_width,
            dtype=self._target.dtype,
        )
        np.add(self._policy_target, self._gait_reference, out=self._target)
        np.clip(self._target, self._ctrl_lower, self._ctrl_upper, out=self._target)

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self._target, joint_ids=self._joint_ids_array)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = _env_ids(self._env, env_ids)
        self._raw_action[ids] = 0.0
        self._previous_policy_action[ids] = 0.0
        self._effective_action[ids] = 0.0
        self._command_noise[ids] = 0.0
        self._command_delayed[ids] = False
        self._clipped_action[ids] = 0.0
        if self._reset_target_from_joint_state:
            joint_pos = np.asarray(
                self._entity.data.joint_pos[ids][:, self._joint_ids_array],
                dtype=self._policy_target.dtype,
            )
            self._episode_anchor[ids] = np.clip(
                joint_pos,
                self._ctrl_lower,
                self._ctrl_upper,
            )
        else:
            self._episode_anchor[ids] = self._reset_target
        self._policy_target[ids] = self._episode_anchor[ids]
        self._gait_reference[ids] = 0.0
        self._target[ids] = self._episode_anchor[ids]


def _episode_phase(env: _LeapEnv, period: float) -> np.ndarray:
    """Return a deployable per-episode sine/cosine clock."""
    previous_steps = np.maximum(env.episode_length_buf - 1, 0)
    angle = previous_steps.astype(get_global_dtype()) * env.step_dt * (2.0 * np.pi / period)
    return np.stack((np.sin(angle), np.cos(angle)), axis=1).astype(get_global_dtype(), copy=False)


class LeapProprioObservation(ManagerTermBase):
    """Hardware-compatible LEAP actor frame: joint pose, targets, and phase."""

    _ALLOWED_PARAMS = frozenset({"entity_name", "action_name", "joint_noise", "phase_period"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: _LeapEnv):
        super().__init__(env)
        term = type(self).__name__
        unexpected = set(cfg.params) - self._ALLOWED_PARAMS
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        entity_name = _name(term, "entity_name", cfg.params.get("entity_name"))
        action_name = _name(term, "action_name", cfg.params.get("action_name"))
        self._entity = cast("Entity", env.scene[entity_name])
        action = env.action_manager.get_term(action_name)
        if not isinstance(action, LeapIncrementalPositionAction):
            raise TypeError(
                f"{term} action {action_name!r} must be LeapIncrementalPositionAction, "
                f"got {type(action).__name__}"
            )
        self._action = action
        self._joint_noise = _real(term, "joint_noise", cfg.params.get("joint_noise"), minimum=0.0)
        self._phase_period = _real(
            term,
            "phase_period",
            cfg.params.get("phase_period"),
            minimum=0.0,
            strict_minimum=True,
        )
        self._dof_mid = (self._action.ctrl_upper + self._action.ctrl_lower) / 2.0
        self._dof_range = self._action.ctrl_upper - self._action.ctrl_lower
        self._just_reset = np.ones(env.num_envs, dtype=np.bool_)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        self._just_reset[_env_ids(self._env, env_ids)] = True

    def __call__(self, env: _LeapEnv, **params: Any) -> np.ndarray:
        del params
        dof_pos = np.asarray(
            self._entity.data.joint_pos[:, self._action.joint_ids],
            dtype=get_global_dtype(),
        )
        dof_pos_norm = 2.0 * (dof_pos - self._dof_mid) / (self._dof_range + 1.0e-8)
        if self._joint_noise > 0.0:
            active = ~self._just_reset
            if np.any(active):
                dof_pos_norm = dof_pos_norm.copy()
                dof_pos_norm[active] += env.rng.uniform(
                    -self._joint_noise,
                    self._joint_noise,
                    size=(int(np.count_nonzero(active)), self._action.action_dim),
                )
        self._just_reset[:] = False
        return np.concatenate(
            (dof_pos_norm, self._action.target, _episode_phase(env, self._phase_period)),
            axis=1,
            dtype=get_global_dtype(),
        )


class LeapRotationObservation(ManagerTermBase):
    """Privileged state frame shared by the critic, termination, and rewards."""

    _ALLOWED_PARAMS = frozenset(
        {
            "entity_name",
            "action_name",
            "joint_noise",
            "torque_estimate_kp",
            "torque_estimate_kd",
            "phase_period",
        }
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: _LeapEnv):
        super().__init__(env)
        term = type(self).__name__
        unexpected = set(cfg.params) - self._ALLOWED_PARAMS
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        entity_name = _name(term, "entity_name", cfg.params.get("entity_name"))
        action_name = _name(term, "action_name", cfg.params.get("action_name"))
        self._entity = cast("Entity", env.scene[entity_name])
        action = env.action_manager.get_term(action_name)
        if not isinstance(action, LeapIncrementalPositionAction):
            raise TypeError(
                f"{term} action {action_name!r} must be LeapIncrementalPositionAction, "
                f"got {type(action).__name__}"
            )
        self._action = action
        self._joint_noise = _real(term, "joint_noise", cfg.params.get("joint_noise"), minimum=0.0)
        self._torque_kp = _real(
            term, "torque_estimate_kp", cfg.params.get("torque_estimate_kp"), minimum=0.0
        )
        self._torque_kd = _real(
            term, "torque_estimate_kd", cfg.params.get("torque_estimate_kd"), minimum=0.0
        )
        self._phase_period = _real(
            term,
            "phase_period",
            cfg.params.get("phase_period"),
            minimum=0.0,
            strict_minimum=True,
        )

        dtype = get_global_dtype()
        self.dof_pos = np.asarray(
            self._entity.data.joint_pos[:, self._action.joint_ids], dtype=dtype
        ).copy()
        self.dof_vel = np.zeros_like(self.dof_pos)
        self.object_pos = np.asarray(self._entity.data.root_link_pos_w, dtype=dtype).copy()
        self.object_quat = np.asarray(self._entity.data.root_link_quat_w, dtype=dtype).copy()
        self.object_linvel = np.zeros_like(self.object_pos)
        self.object_angvel = np.zeros_like(self.object_pos)
        self.torques = np.zeros_like(self.dof_pos)
        self.init_pose = self.dof_pos.copy()
        self.init_object_pos = self.object_pos.copy()
        self.init_object_quat = self.object_quat.copy()
        self._previous_dof_pos = self.dof_pos.copy()
        self._previous_object_pos = self.object_pos.copy()
        self._previous_object_quat = self.object_quat.copy()
        self._just_reset = np.ones(env.num_envs, dtype=np.bool_)
        self._last_counter = int(env.common_step_counter)

        self._dof_mid = (self._action.ctrl_upper + self._action.ctrl_lower) / 2.0
        self._dof_range = self._action.ctrl_upper - self._action.ctrl_lower

    @property
    def last_counter(self) -> int:
        return self._last_counter

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = _env_ids(self._env, env_ids)
        dof_pos = np.asarray(
            self._entity.data.joint_pos[:, self._action.joint_ids], dtype=get_global_dtype()
        )
        object_pos = np.asarray(self._entity.data.root_link_pos_w, dtype=get_global_dtype())
        object_quat = np.asarray(self._entity.data.root_link_quat_w, dtype=get_global_dtype())
        self.dof_pos[ids] = dof_pos[ids]
        self.dof_vel[ids] = 0.0
        self.object_pos[ids] = object_pos[ids]
        self.object_quat[ids] = object_quat[ids]
        self.object_linvel[ids] = 0.0
        self.object_angvel[ids] = 0.0
        self.torques[ids] = 0.0
        self.init_pose[ids] = dof_pos[ids]
        self.init_object_pos[ids] = object_pos[ids]
        self.init_object_quat[ids] = object_quat[ids]
        self._previous_dof_pos[ids] = dof_pos[ids]
        self._previous_object_pos[ids] = object_pos[ids]
        self._previous_object_quat[ids] = object_quat[ids]
        self._just_reset[ids] = True
        self._last_counter = int(cast("_LeapEnv", self._env).common_step_counter)

    def snapshot(self, env: _LeapEnv) -> LeapRotationObservation:
        counter = int(env.common_step_counter)
        if counter == self._last_counter:
            return self
        if counter != self._last_counter + 1:
            raise RuntimeError(
                f"LeapRotationObservation missed a control-step update: "
                f"last={self._last_counter}, current={counter}"
            )

        dtype = get_global_dtype()
        dof_pos = np.asarray(self._entity.data.joint_pos[:, self._action.joint_ids], dtype=dtype)
        object_pos = np.asarray(self._entity.data.root_link_pos_w, dtype=dtype)
        object_quat = np.asarray(self._entity.data.root_link_quat_w, dtype=dtype)
        np.subtract(dof_pos, self._previous_dof_pos, out=self.dof_vel)
        self.dof_vel /= env.step_dt
        np.subtract(object_pos, self._previous_object_pos, out=self.object_linvel)
        self.object_linvel /= env.step_dt
        self.object_angvel[:] = np_quat_angular_velocity_from_pair(
            object_quat, self._previous_object_quat, env.step_dt
        )
        self.dof_pos[:] = dof_pos
        self.object_pos[:] = object_pos
        self.object_quat[:] = object_quat
        self.torques[:] = self._torque_kp * (self._action.target - self.dof_pos)
        self.torques -= self._torque_kd * self.dof_vel
        np.clip(self.torques, -0.5, 0.5, out=self.torques)
        self._previous_dof_pos[:] = dof_pos
        self._previous_object_pos[:] = object_pos
        self._previous_object_quat[:] = object_quat
        self._last_counter = counter
        return self

    def __call__(self, env: _LeapEnv, **params: Any) -> np.ndarray:
        del params
        self.snapshot(env)
        dof_pos_norm = 2.0 * (self.dof_pos - self._dof_mid) / (self._dof_range + 1.0e-8)
        if self._joint_noise > 0.0:
            active = ~self._just_reset
            if np.any(active):
                dof_pos_norm = dof_pos_norm.copy()
                dof_pos_norm[active] += env.rng.uniform(
                    -self._joint_noise,
                    self._joint_noise,
                    size=(int(np.count_nonzero(active)), self._action.action_dim),
                )
        self._just_reset[:] = False
        return np.concatenate(
            (
                dof_pos_norm,
                self.dof_vel,
                self._action.target,
                _episode_phase(env, self._phase_period),
                self.object_pos,
                self.object_quat,
                self.object_linvel,
                self.object_angvel,
            ),
            axis=1,
            dtype=get_global_dtype(),
        )


class LeapGraspTermination(ManagerTermBase):
    """Cache privileged grasp quality and terminate only geometric escapes.

    Contact loss makes ``valid_grasp`` false but does not itself terminate an
    episode.  This leaves room for a policy to release and replace individual
    fingers while learning an in-hand gait.
    """

    _ALLOWED_PARAMS = frozenset(
        {
            "observation_group",
            "observation_term",
            "opposing_contact_sensor_groups",
            "thumb_contact_sensor_names",
            "palm_contact_sensor_names",
            "maximum_valid_vertical_drop",
            "maximum_valid_lateral_drift",
            "maximum_valid_tilt_deg",
            "termination_vertical_drop",
            "termination_lateral_drift",
            "absolute_minimum_height",
            "minimum_opposing_contacts",
            "desired_opposing_contacts",
            "finger_contact_grace_seconds",
            "palm_support_grace_seconds",
            "palm_support_minimum_drop",
            "vertical_quality_scale",
            "lateral_quality_scale",
            "tilt_quality_scale_deg",
        }
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: _LeapEnv):
        super().__init__(env)
        term = type(self).__name__
        unexpected = set(cfg.params) - self._ALLOWED_PARAMS
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        group = _name(term, "observation_group", cfg.params.get("observation_group"))
        name = _name(term, "observation_term", cfg.params.get("observation_term"))
        observation = env.observation_manager.get_term_cfg(group, name).func
        if not isinstance(observation, LeapRotationObservation):
            raise TypeError(
                f"{term} observation {group}/{name} must be LeapRotationObservation, "
                f"got {type(observation).__name__}"
            )
        self.observation = observation

        opposing_groups = _name_groups(
            term,
            "opposing_contact_sensor_groups",
            cfg.params.get("opposing_contact_sensor_groups"),
        )
        opposing_names = tuple(name for group in opposing_groups for name in group)
        thumb_names = _names(
            term,
            "thumb_contact_sensor_names",
            cfg.params.get("thumb_contact_sensor_names"),
        )
        palm_names = _names(
            term, "palm_contact_sensor_names", cfg.params.get("palm_contact_sensor_names")
        )
        self._opposing_contacts = env.scene.bind_sensor_data(opposing_names)
        self._thumb_contact = env.scene.bind_sensor_data(thumb_names)
        self._palm_contacts = env.scene.bind_sensor_data(palm_names)
        for label, view, count in (
            ("opposing", self._opposing_contacts, len(opposing_names)),
            ("thumb", self._thumb_contact, len(thumb_names)),
            ("palm", self._palm_contacts, len(palm_names)),
        ):
            if view.dimensions != (1,) * count:
                raise ValueError(
                    f"{term} {label} contact sensors must each be scalar, got {view.dimensions}"
                )
        opposing_widths = tuple(len(group) for group in opposing_groups)
        self._opposing_slices = tuple(
            slice(sum(opposing_widths[:index]), sum(opposing_widths[: index + 1]))
            for index in range(len(opposing_widths))
        )

        self._maximum_valid_vertical_drop = _real(
            term,
            "maximum_valid_vertical_drop",
            cfg.params.get("maximum_valid_vertical_drop"),
            minimum=0.0,
            strict_minimum=True,
        )
        self._maximum_valid_lateral_drift = _real(
            term,
            "maximum_valid_lateral_drift",
            cfg.params.get("maximum_valid_lateral_drift"),
            minimum=0.0,
            strict_minimum=True,
        )
        self._maximum_valid_tilt_rad = np.deg2rad(
            _real(
                term,
                "maximum_valid_tilt_deg",
                cfg.params.get("maximum_valid_tilt_deg"),
                minimum=0.0,
                strict_minimum=True,
            )
        )
        self._termination_vertical_drop = _real(
            term,
            "termination_vertical_drop",
            cfg.params.get("termination_vertical_drop"),
            minimum=self._maximum_valid_vertical_drop,
            strict_minimum=True,
        )
        self._termination_lateral_drift = _real(
            term,
            "termination_lateral_drift",
            cfg.params.get("termination_lateral_drift"),
            minimum=self._maximum_valid_lateral_drift,
            strict_minimum=True,
        )
        self._absolute_minimum_height = _real(
            term, "absolute_minimum_height", cfg.params.get("absolute_minimum_height")
        )
        self._minimum_opposing_contacts = _positive_int(
            term,
            "minimum_opposing_contacts",
            cfg.params.get("minimum_opposing_contacts"),
        )
        self._desired_opposing_contacts = _positive_int(
            term,
            "desired_opposing_contacts",
            cfg.params.get("desired_opposing_contacts"),
        )
        if self._minimum_opposing_contacts > len(opposing_groups):
            raise ValueError(f"{term} minimum_opposing_contacts exceeds configured sensors")
        if self._desired_opposing_contacts < self._minimum_opposing_contacts:
            raise ValueError(
                f"{term} desired_opposing_contacts must be at least minimum_opposing_contacts"
            )
        self._finger_contact_grace_seconds = _real(
            term,
            "finger_contact_grace_seconds",
            cfg.params.get("finger_contact_grace_seconds"),
            minimum=0.0,
        )
        self._palm_support_grace_seconds = _real(
            term,
            "palm_support_grace_seconds",
            cfg.params.get("palm_support_grace_seconds"),
            minimum=0.0,
            strict_minimum=True,
        )
        self._palm_support_minimum_drop = _real(
            term,
            "palm_support_minimum_drop",
            cfg.params.get("palm_support_minimum_drop"),
            minimum=0.0,
        )
        self._vertical_quality_scale = _real(
            term,
            "vertical_quality_scale",
            cfg.params.get("vertical_quality_scale"),
            minimum=0.0,
            strict_minimum=True,
        )
        self._lateral_quality_scale = _real(
            term,
            "lateral_quality_scale",
            cfg.params.get("lateral_quality_scale"),
            minimum=0.0,
            strict_minimum=True,
        )
        self._tilt_quality_scale_rad = np.deg2rad(
            _real(
                term,
                "tilt_quality_scale_deg",
                cfg.params.get("tilt_quality_scale_deg"),
                minimum=0.0,
                strict_minimum=True,
            )
        )

        dtype = get_global_dtype()
        self.vertical_drop = np.zeros(env.num_envs, dtype=dtype)
        self.lateral_drift = np.zeros(env.num_envs, dtype=dtype)
        self.tilt_rad = np.zeros(env.num_envs, dtype=dtype)
        self.opposing_contact_count = np.zeros(env.num_envs, dtype=np.int32)
        # Raw contact is retained separately from the grace-filtered deployment
        # signal so rollout audits can distinguish a real fingertip contact from
        # the short contact-loss grace period used by the task.
        self.raw_opposing_contacts = np.zeros((env.num_envs, len(opposing_groups)), dtype=np.bool_)
        self.opposing_contacts = np.zeros((env.num_envs, len(opposing_groups)), dtype=np.bool_)
        self.raw_thumb_contact = np.zeros(env.num_envs, dtype=np.bool_)
        self.thumb_contact = np.zeros(env.num_envs, dtype=np.bool_)
        self._opposing_contact_gap = np.zeros((env.num_envs, len(opposing_groups)), dtype=dtype)
        self._thumb_contact_gap = np.zeros(env.num_envs, dtype=dtype)
        self.palm_contact = np.zeros(env.num_envs, dtype=np.bool_)
        self.palm_contact_duration = np.zeros(env.num_envs, dtype=dtype)
        self.palm_supported = np.zeros(env.num_envs, dtype=np.bool_)
        self.position_ok = np.zeros(env.num_envs, dtype=np.bool_)
        self.tilt_ok = np.zeros(env.num_envs, dtype=np.bool_)
        self.contact_ok = np.zeros(env.num_envs, dtype=np.bool_)
        self.valid_grasp = np.zeros(env.num_envs, dtype=np.bool_)
        self.grasp_quality = np.zeros(env.num_envs, dtype=dtype)
        self.fallen = np.zeros(env.num_envs, dtype=np.bool_)
        # Backwards-compatible name used by the evaluator and existing logs.
        self.dropped = self.fallen
        self.cylinder_axis_w = np.zeros((env.num_envs, 3), dtype=dtype)
        self._last_counter = int(env.common_step_counter)

    @property
    def last_counter(self) -> int:
        return self._last_counter

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = _env_ids(self._env, env_ids)
        for values in (
            self.vertical_drop,
            self.lateral_drift,
            self.tilt_rad,
            self.opposing_contact_count,
            self.raw_opposing_contacts,
            self.opposing_contacts,
            self.raw_thumb_contact,
            self.thumb_contact,
            self._opposing_contact_gap,
            self._thumb_contact_gap,
            self.palm_contact,
            self.palm_contact_duration,
            self.palm_supported,
            self.position_ok,
            self.tilt_ok,
            self.contact_ok,
            self.valid_grasp,
            self.grasp_quality,
            self.fallen,
        ):
            values[ids] = 0
        self._last_counter = int(cast("_LeapEnv", self._env).common_step_counter)

    def __call__(self, env: _LeapEnv, **params: Any) -> np.ndarray:
        del params
        self.observation.snapshot(env)
        delta = self.observation.object_pos - self.observation.init_object_pos
        self.vertical_drop[:] = np.maximum(-delta[:, 2], 0.0)
        self.lateral_drift[:] = np.linalg.norm(delta[:, :2], axis=1)

        local_axis = np.array((0.0, 0.0, 1.0), dtype=get_global_dtype())
        self.cylinder_axis_w[:] = np_quat_apply_batched(self.observation.object_quat, local_axis)
        initial_axis = np_quat_apply_batched(self.observation.init_object_quat, local_axis)
        alignment = np.abs(np.sum(self.cylinder_axis_w * initial_axis, axis=1))
        self.tilt_rad[:] = np.arccos(np.clip(alignment, 0.0, 1.0))

        opposing_sensor_values = self._opposing_contacts.read() > 0.5
        opposing_raw = np.stack(
            [
                np.any(opposing_sensor_values[:, columns], axis=1)
                for columns in self._opposing_slices
            ],
            axis=1,
        )
        self.raw_opposing_contacts[:] = opposing_raw
        self._opposing_contact_gap[:] = np.where(
            opposing_raw, 0.0, self._opposing_contact_gap + env.step_dt
        )
        opposing = self._opposing_contact_gap <= self._finger_contact_grace_seconds
        self.opposing_contacts[:] = opposing
        self.opposing_contact_count[:] = np.count_nonzero(opposing, axis=1)
        thumb_raw = np.any(self._thumb_contact.read() > 0.5, axis=1)
        self.raw_thumb_contact[:] = thumb_raw
        self._thumb_contact_gap[:] = np.where(thumb_raw, 0.0, self._thumb_contact_gap + env.step_dt)
        self.thumb_contact[:] = self._thumb_contact_gap <= self._finger_contact_grace_seconds
        self.palm_contact[:] = np.any(self._palm_contacts.read() > 0.5, axis=1)
        support_candidate = self.palm_contact & (
            self.vertical_drop >= self._palm_support_minimum_drop
        )
        self.palm_contact_duration[:] = np.where(
            support_candidate, self.palm_contact_duration + env.step_dt, 0.0
        )
        self.palm_supported[:] = self.palm_contact_duration >= self._palm_support_grace_seconds

        self.position_ok[:] = (self.vertical_drop <= self._maximum_valid_vertical_drop) & (
            self.lateral_drift <= self._maximum_valid_lateral_drift
        )
        self.tilt_ok[:] = self.tilt_rad <= self._maximum_valid_tilt_rad
        self.contact_ok[:] = self.thumb_contact & (
            self.opposing_contact_count >= self._minimum_opposing_contacts
        )
        self.valid_grasp[:] = (
            self.position_ok & self.tilt_ok & self.contact_ok & ~self.palm_supported
        )

        position_quality = np.exp(
            -np.square(self.vertical_drop / self._vertical_quality_scale)
            - np.square(self.lateral_drift / self._lateral_quality_scale)
        )
        contact_quality = 0.5 * self.thumb_contact + 0.5 * np.clip(
            self.opposing_contact_count / self._desired_opposing_contacts, 0.0, 1.0
        )
        tilt_quality = np.exp(-np.square(self.tilt_rad / self._tilt_quality_scale_rad))
        self.grasp_quality[:] = position_quality * (0.2 + 0.8 * contact_quality) * tilt_quality

        self.fallen[:] = (
            (self.vertical_drop >= self._termination_vertical_drop)
            | (self.lateral_drift >= self._termination_lateral_drift)
            | (self.observation.object_pos[:, 2] < self._absolute_minimum_height)
        )
        self.grasp_quality[self.fallen | self.palm_supported] = 0.0
        self._last_counter = int(env.common_step_counter)
        return self.fallen


class LeapDropTermination(ManagerTermBase):
    """Legacy absolute-height termination retained for old run snapshots."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: _LeapEnv):
        super().__init__(env)
        term = type(self).__name__
        allowed = {"observation_group", "observation_term", "minimum_object_height"}
        unexpected = set(cfg.params) - allowed
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        group = _name(term, "observation_group", cfg.params.get("observation_group"))
        name = _name(term, "observation_term", cfg.params.get("observation_term"))
        observation = env.observation_manager.get_term_cfg(group, name).func
        if not isinstance(observation, LeapRotationObservation):
            raise TypeError(
                f"{term} observation {group}/{name} must be LeapRotationObservation, "
                f"got {type(observation).__name__}"
            )
        self.observation = observation
        self._minimum_height = _real(
            term, "minimum_object_height", cfg.params.get("minimum_object_height")
        )
        self.dropped = np.zeros(env.num_envs, dtype=np.bool_)
        self._last_counter = int(env.common_step_counter)

    @property
    def last_counter(self) -> int:
        return self._last_counter

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        self.dropped[_env_ids(self._env, env_ids)] = False
        self._last_counter = int(cast("_LeapEnv", self._env).common_step_counter)

    def __call__(self, env: _LeapEnv, **params: Any) -> np.ndarray:
        del params
        self.observation.snapshot(env)
        self.dropped[:] = self.observation.object_pos[:, 2] < self._minimum_height
        self._last_counter = int(env.common_step_counter)
        return self.dropped


def _rotation_state(
    env: _LeapEnv,
    state_term_name: str,
) -> tuple[LeapGraspTermination | LeapDropTermination, LeapRotationObservation]:
    name = _name("Leap reward", "state_term_name", state_term_name)
    state = env.termination_manager.get_term_cfg(name).func
    if not isinstance(state, (LeapGraspTermination, LeapDropTermination)):
        raise TypeError(
            f"Leap reward termination term {name!r} must be a LEAP grasp state, "
            f"got {type(state).__name__}"
        )
    if state.last_counter != int(env.common_step_counter):
        raise RuntimeError(
            f"Leap reward state {name!r} was not computed for control step "
            f"{env.common_step_counter}"
        )
    return state, state.observation


class LeapBidirectionalFrontierReward(ManagerTermBase):
    """Discover either direction, then sustain progress along the committed sign.

    The held angle freezes while the strict grasp contract is invalid, so free
    object motion cannot create progress that is claimed after recontact. Dense
    held-speed shaping makes useful motion visible between frontier advances;
    one-time milestones and wrong-way entry penalties cannot be farmed by rocking.
    Old snapshots without the newer parameters retain their original behavior.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: _LeapEnv):
        super().__init__(env)
        term = type(self).__name__
        allowed = {
            "state_term_name",
            "maximum_frontier_speed",
            "frontier_progress_gain_per_rad",
            "maximum_frontier_progress_multiplier",
            "reversal_speed_threshold",
            "direction_lock_threshold",
            "directional_velocity_scale",
            "directional_speed_filter_time_constant",
            "fast_rotation_bonus_scale",
            "opposite_direction_penalty_scale",
            "reversal_event_penalty",
            "stall_speed_threshold",
            "stall_grace_period",
            "stall_penalty_scale",
            "milestone_angles",
            "milestone_bonus",
        }
        unexpected = set(cfg.params) - allowed
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        self._state_term_name = _name(term, "state_term_name", cfg.params.get("state_term_name"))
        self._maximum_frontier_speed = _real(
            term,
            "maximum_frontier_speed",
            cfg.params.get("maximum_frontier_speed"),
            minimum=0.0,
            strict_minimum=True,
        )
        self._frontier_progress_gain_per_rad = _real(
            term,
            "frontier_progress_gain_per_rad",
            cfg.params.get("frontier_progress_gain_per_rad", 0.0),
            minimum=0.0,
        )
        self._maximum_frontier_progress_multiplier = _real(
            term,
            "maximum_frontier_progress_multiplier",
            cfg.params.get("maximum_frontier_progress_multiplier", 1.0),
            minimum=1.0,
        )
        self._reversal_speed_threshold = _real(
            term,
            "reversal_speed_threshold",
            cfg.params.get("reversal_speed_threshold"),
            minimum=0.0,
        )
        lock_threshold = cfg.params.get("direction_lock_threshold")
        self._direction_lock_threshold = (
            None
            if lock_threshold is None
            else _real(
                term,
                "direction_lock_threshold",
                lock_threshold,
                minimum=0.0,
                strict_minimum=True,
            )
        )
        self._opposite_direction_penalty_scale = _real(
            term,
            "opposite_direction_penalty_scale",
            cfg.params.get("opposite_direction_penalty_scale", 0.0),
            minimum=0.0,
        )
        self._directional_velocity_scale = _real(
            term,
            "directional_velocity_scale",
            cfg.params.get("directional_velocity_scale", 0.0),
            minimum=0.0,
        )
        filter_time_constant = _real(
            term,
            "directional_speed_filter_time_constant",
            cfg.params.get("directional_speed_filter_time_constant", env.step_dt),
            minimum=0.0,
            strict_minimum=True,
        )
        self._directional_speed_filter_alpha = 1.0 - np.exp(-env.step_dt / filter_time_constant)
        self._fast_rotation_bonus_scale = _real(
            term,
            "fast_rotation_bonus_scale",
            cfg.params.get("fast_rotation_bonus_scale", 0.0),
            minimum=0.0,
        )
        self._reversal_event_penalty = _real(
            term,
            "reversal_event_penalty",
            cfg.params.get("reversal_event_penalty", 0.0),
            minimum=0.0,
        )
        self._stall_speed_threshold = _real(
            term,
            "stall_speed_threshold",
            cfg.params.get("stall_speed_threshold", 0.0),
            minimum=0.0,
        )
        self._stall_grace_period = _real(
            term,
            "stall_grace_period",
            cfg.params.get("stall_grace_period", 0.0),
            minimum=0.0,
        )
        self._stall_penalty_scale = _real(
            term,
            "stall_penalty_scale",
            cfg.params.get("stall_penalty_scale", 0.0),
            minimum=0.0,
        )
        milestone_angles = cfg.params.get("milestone_angles", ())
        if isinstance(milestone_angles, (str, bytes)) or not isinstance(
            milestone_angles, (tuple, list)
        ):
            raise TypeError(f"{term} milestone_angles must be a sequence of real numbers")
        self._milestone_angles = np.asarray(
            [
                _real(
                    term,
                    f"milestone_angles[{index}]",
                    angle,
                    minimum=0.0,
                    strict_minimum=True,
                )
                for index, angle in enumerate(milestone_angles)
            ],
            dtype=get_global_dtype(),
        )
        if self._milestone_angles.size > 1 and np.any(np.diff(self._milestone_angles) <= 0.0):
            raise ValueError(f"{term} milestone_angles must be strictly increasing")
        self._milestone_bonus_scale = _real(
            term,
            "milestone_bonus",
            cfg.params.get("milestone_bonus", 0.0),
            minimum=0.0,
        )
        dtype = get_global_dtype()
        self.held_angle = np.zeros(env.num_envs, dtype=dtype)
        self.frontier_angle = np.zeros(env.num_envs, dtype=dtype)
        self.frontier_progress = np.zeros(env.num_envs, dtype=dtype)
        self.frontier_speed = np.zeros(env.num_envs, dtype=dtype)
        self.frontier_progress_multiplier = np.ones(env.num_envs, dtype=dtype)
        self.reward = np.zeros(env.num_envs, dtype=dtype)
        self.directional_speed = np.zeros(env.num_envs, dtype=dtype)
        self.filtered_directional_speed = np.zeros(env.num_envs, dtype=dtype)
        self.fast_rotation_bonus = np.zeros(env.num_envs, dtype=dtype)
        self.opposite_direction_speed = np.zeros(env.num_envs, dtype=dtype)
        self.reversal_penalty_event = np.zeros(env.num_envs, dtype=np.bool_)
        self.stalled = np.zeros(env.num_envs, dtype=np.bool_)
        self.committed_time = np.zeros(env.num_envs, dtype=dtype)
        self.milestone_bonus = np.zeros(env.num_envs, dtype=dtype)
        self.milestones_reached = np.zeros(env.num_envs, dtype=np.int32)
        self.committed_direction = np.zeros(env.num_envs, dtype=np.int8)
        self.direction_reversals = np.zeros(env.num_envs, dtype=np.int32)
        self._last_motion_direction = np.zeros(env.num_envs, dtype=np.int8)
        self._visual_object_pos = np.zeros((env.num_envs, 3), dtype=dtype)
        self._visual_axis_w = np.zeros((env.num_envs, 3), dtype=dtype)
        self._last_counter = -1

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = _env_ids(self._env, env_ids)
        for values in (
            self.held_angle,
            self.frontier_angle,
            self.frontier_progress,
            self.frontier_speed,
            self.frontier_progress_multiplier,
            self.reward,
            self.directional_speed,
            self.filtered_directional_speed,
            self.fast_rotation_bonus,
            self.opposite_direction_speed,
            self.reversal_penalty_event,
            self.stalled,
            self.committed_time,
            self.milestone_bonus,
            self.milestones_reached,
            self.committed_direction,
            self.direction_reversals,
            self._last_motion_direction,
        ):
            values[ids] = 0
        self.frontier_progress_multiplier[ids] = 1.0
        self._last_counter = -1

    def __call__(self, env: _LeapEnv, **params: Any) -> np.ndarray:
        del params
        counter = int(env.common_step_counter)
        if self._last_counter == counter:
            return self.reward
        grasp, state = _rotation_state(env, self._state_term_name)
        if not isinstance(grasp, LeapGraspTermination):
            raise TypeError("LeapBidirectionalFrontierReward requires LeapGraspTermination")

        axial_speed = np.sum(state.object_angvel * grasp.cylinder_axis_w, axis=1)
        self._visual_object_pos[:] = state.object_pos
        self._visual_axis_w[:] = grasp.cylinder_axis_w
        held_speed = np.where(grasp.valid_grasp, axial_speed, 0.0)
        moving = np.abs(held_speed) >= self._reversal_speed_threshold
        direction = np.sign(held_speed).astype(np.int8)
        reversed_direction = (
            moving & (self._last_motion_direction != 0) & (direction != self._last_motion_direction)
        )
        self.direction_reversals += reversed_direction
        previous_motion_direction = self._last_motion_direction.copy()
        self._last_motion_direction[moving] = direction[moving]

        self.held_angle += held_speed * env.step_dt
        candidate_frontier = np.abs(self.held_angle)
        if self._direction_lock_threshold is not None:
            lock_tolerance = 8.0 * np.finfo(self.held_angle.dtype).eps
            newly_committed = (self.committed_direction == 0) & (
                np.abs(self.held_angle) >= self._direction_lock_threshold - lock_tolerance
            )
            self.committed_direction[newly_committed] = np.sign(
                self.held_angle[newly_committed]
            ).astype(np.int8)
            committed = self.committed_direction != 0
            candidate_frontier[committed] = (
                self.committed_direction[committed] * self.held_angle[committed]
            )

        self.frontier_progress[:] = np.maximum(candidate_frontier - self.frontier_angle, 0.0)
        np.maximum(self.frontier_angle, candidate_frontier, out=self.frontier_angle)
        np.divide(self.frontier_progress, env.step_dt, out=self.frontier_speed)
        np.minimum(
            self.frontier_speed,
            self._maximum_frontier_speed,
            out=self.frontier_speed,
        )
        self.frontier_progress_multiplier[:] = np.minimum(
            1.0 + self._frontier_progress_gain_per_rad * self.frontier_angle,
            self._maximum_frontier_progress_multiplier,
        )
        committed = self.committed_direction != 0
        self.directional_speed[:] = np.abs(held_speed)
        self.directional_speed[committed] = (
            self.committed_direction[committed] * held_speed[committed]
        )
        np.clip(
            self.directional_speed,
            -self._maximum_frontier_speed,
            self._maximum_frontier_speed,
            out=self.directional_speed,
        )
        self.filtered_directional_speed += self._directional_speed_filter_alpha * (
            self.directional_speed - self.filtered_directional_speed
        )
        self.fast_rotation_bonus.fill(0.0)
        forward_filtered_speed = np.maximum(self.filtered_directional_speed[committed], 0.0)
        self.fast_rotation_bonus[committed] = np.square(forward_filtered_speed) / (
            self._maximum_frontier_speed
        )
        self.committed_time[committed & grasp.valid_grasp] += env.step_dt
        self.stalled[:] = (
            committed
            & grasp.valid_grasp
            & (self.committed_time >= self._stall_grace_period)
            & (self.filtered_directional_speed < self._stall_speed_threshold)
        )
        self.opposite_direction_speed.fill(0.0)
        self.opposite_direction_speed[committed] = np.clip(
            -(self.committed_direction[committed] * held_speed[committed]),
            0.0,
            self._maximum_frontier_speed,
        )
        opposite_direction = -self.committed_direction
        self.reversal_penalty_event[:] = (
            committed
            & moving
            & (direction == opposite_direction)
            & (previous_motion_direction != opposite_direction)
        )

        previous_milestones = self.milestones_reached.copy()
        if self._milestone_angles.size:
            self.milestones_reached[:] = np.sum(
                self.frontier_angle[:, None] >= self._milestone_angles[None, :], axis=1
            )
        else:
            self.milestones_reached.fill(0)
        newly_reached = self.milestones_reached - previous_milestones
        self.milestone_bonus[:] = newly_reached * self._milestone_bonus_scale

        reward_rate = (
            self.frontier_progress_multiplier * self.frontier_speed
            + (self._directional_velocity_scale * self.filtered_directional_speed)
            + (self._fast_rotation_bonus_scale * self.fast_rotation_bonus)
            - (self._opposite_direction_penalty_scale * self.opposite_direction_speed)
            - (self._stall_penalty_scale * self.stalled)
        )
        # The manager scales rewards by dt. Convert discrete event rewards to
        # rates here so their configured magnitude is independent of control dt.
        reward_rate += (
            self.milestone_bonus - self._reversal_event_penalty * self.reversal_penalty_event
        ) / env.step_dt
        np.multiply(reward_rate, grasp.grasp_quality, out=self.reward)
        self._last_counter = counter
        return self.reward

    def playback_debug_overlay_getter(self):
        """Show a virtual surface dot driven by rewarded held rotation.

        Green means fast motion in the committed direction, orange means slow
        motion, red means stalled or wrong-way motion, and yellow means the
        episode has not committed to a direction yet.
        """

        def _get_overlay() -> list[list[DebugPrimitive]] | None:
            if self._last_counter < 0:
                return None
            overlays: list[list[DebugPrimitive]] = []
            for env_idx in range(self.num_envs):
                axis = np.asarray(self._visual_axis_w[env_idx], dtype=np.float64)
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm < 1.0e-9:
                    overlays.append([])
                    continue
                axis /= axis_norm
                reference = (
                    np.array([0.0, 0.0, 1.0])
                    if abs(float(axis[2])) < 0.9
                    else np.array([1.0, 0.0, 0.0])
                )
                radial_x = np.cross(axis, reference)
                radial_x /= np.linalg.norm(radial_x)
                radial_y = np.cross(axis, radial_x)
                angle = float(self.held_angle[env_idx])
                radial = np.cos(angle) * radial_x + np.sin(angle) * radial_y
                marker_pos = self._visual_object_pos[env_idx] + 0.045 * radial

                if self.committed_direction[env_idx] == 0:
                    rgba = (1.0, 0.8, 0.1, 1.0)
                elif self.stalled[env_idx] or self.filtered_directional_speed[env_idx] < 0.0:
                    rgba = (1.0, 0.1, 0.1, 1.0)
                elif self.filtered_directional_speed[env_idx] >= 0.1:
                    rgba = (0.1, 1.0, 0.2, 1.0)
                else:
                    rgba = (1.0, 0.45, 0.05, 1.0)
                overlays.append(
                    [
                        DebugPrimitive(
                            kind="sphere",
                            pos=tuple(float(value) for value in marker_pos),
                            size=(0.006,),
                            rgba=rgba,
                        )
                    ]
                )
            return overlays

        return _get_overlay


class LeapAxialRotateReward(ManagerTermBase):
    """Legacy signed cylinder-axis rotation reward for existing run snapshots."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: _LeapEnv):
        super().__init__(env)
        term = type(self).__name__
        allowed = {
            "state_term_name",
            "clip_min",
            "clip_max",
        }
        unexpected = set(cfg.params) - allowed
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        self._state_term_name = _name(term, "state_term_name", cfg.params.get("state_term_name"))
        self._clip = _pair(
            term,
            "clip",
            (cfg.params.get("clip_min"), cfg.params.get("clip_max")),
        )

    def __call__(self, env: _LeapEnv, **params: Any) -> np.ndarray:
        del params
        grasp, state = _rotation_state(env, self._state_term_name)
        if not isinstance(grasp, LeapGraspTermination):
            raise TypeError("LeapAxialRotateReward requires LeapGraspTermination")
        rotation = np.clip(
            np.sum(state.object_angvel * grasp.cylinder_axis_w, axis=1),
            self._clip[0],
            self._clip[1],
        )
        return np.asarray(rotation * grasp.grasp_quality, dtype=get_global_dtype())


class LeapRotateReward(ManagerTermBase):
    """Legacy world-axis rotation reward retained for old run snapshots."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: _LeapEnv):
        super().__init__(env)
        term = type(self).__name__
        allowed = {
            "state_term_name",
            "rotation_axis",
            "clip_min",
            "clip_max",
            "minimum_object_height",
            "maximum_object_xy_drift",
        }
        unexpected = set(cfg.params) - allowed
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        self._state_term_name = _name(term, "state_term_name", cfg.params.get("state_term_name"))
        axis = np.asarray(cfg.params.get("rotation_axis"), dtype=get_global_dtype())
        if axis.shape != (3,) or not np.isfinite(axis).all():
            raise ValueError(f"{term} rotation_axis must be a finite 3-D vector")
        self._axis = np.asarray(np_normalize_axis(axis), dtype=get_global_dtype())
        self._clip = _pair(term, "clip", (cfg.params.get("clip_min"), cfg.params.get("clip_max")))
        self._minimum_object_height = _real(
            term, "minimum_object_height", cfg.params.get("minimum_object_height")
        )
        self._maximum_object_xy_drift = _real(
            term,
            "maximum_object_xy_drift",
            cfg.params.get("maximum_object_xy_drift"),
            minimum=0.0,
            strict_minimum=True,
        )

    def __call__(self, env: _LeapEnv, **params: Any) -> np.ndarray:
        del params
        termination, state = _rotation_state(env, self._state_term_name)
        rotation = np.clip(state.object_angvel @ self._axis, self._clip[0], self._clip[1])
        retained = _object_retained(
            termination,
            state,
            minimum_object_height=self._minimum_object_height,
            maximum_object_xy_drift=self._maximum_object_xy_drift,
        )
        return np.asarray(rotation * retained, dtype=get_global_dtype())


def _object_retained(
    termination: LeapGraspTermination | LeapDropTermination,
    state: LeapRotationObservation,
    *,
    minimum_object_height: float,
    maximum_object_xy_drift: float,
) -> np.ndarray:
    xy_drift = np.linalg.norm(state.object_pos[:, :2] - state.init_object_pos[:, :2], axis=1)
    return (
        ~termination.dropped
        & (state.object_pos[:, 2] >= minimum_object_height)
        & (xy_drift <= maximum_object_xy_drift)
    )


def object_retained(
    env: _LeapEnv,
    state_term_name: str,
    minimum_object_height: float,
    maximum_object_xy_drift: float,
) -> np.ndarray:
    termination, state = _rotation_state(env, state_term_name)
    return np.asarray(
        _object_retained(
            termination,
            state,
            minimum_object_height=_real(
                "object_retained", "minimum_object_height", minimum_object_height
            ),
            maximum_object_xy_drift=_real(
                "object_retained",
                "maximum_object_xy_drift",
                maximum_object_xy_drift,
                minimum=0.0,
                strict_minimum=True,
            ),
        ),
        dtype=get_global_dtype(),
    )


def object_retention_quality(
    env: _LeapEnv,
    state_term_name: str,
    minimum_object_height: float,
    target_object_height: float,
    lateral_scale: float,
) -> np.ndarray:
    termination, state = _rotation_state(env, state_term_name)
    minimum_height = _real(
        "object_retention_quality", "minimum_object_height", minimum_object_height
    )
    target_height = _real("object_retention_quality", "target_object_height", target_object_height)
    if target_height <= minimum_height:
        raise ValueError("target_object_height must exceed minimum_object_height")
    scale = _real(
        "object_retention_quality",
        "lateral_scale",
        lateral_scale,
        minimum=0.0,
        strict_minimum=True,
    )
    height_quality = np.clip(
        (state.object_pos[:, 2] - minimum_height) / (target_height - minimum_height),
        0.0,
        1.0,
    )
    xy_drift = np.linalg.norm(state.object_pos[:, :2] - state.init_object_pos[:, :2], axis=1)
    lateral_quality = 1.0 / (1.0 + np.square(xy_drift / scale))
    return np.asarray(
        (~termination.dropped) * height_quality * lateral_quality,
        dtype=get_global_dtype(),
    )


def _grasp_state(
    env: _LeapEnv, state_term_name: str
) -> tuple[LeapGraspTermination, LeapRotationObservation]:
    grasp, state = _rotation_state(env, state_term_name)
    if not isinstance(grasp, LeapGraspTermination):
        raise TypeError("contact-aware reward requires LeapGraspTermination")
    return grasp, state


def grasp_quality(env: _LeapEnv, state_term_name: str) -> np.ndarray:
    grasp, _ = _grasp_state(env, state_term_name)
    return np.asarray(grasp.grasp_quality, dtype=get_global_dtype())


def palm_support(env: _LeapEnv, state_term_name: str) -> np.ndarray:
    grasp, _ = _grasp_state(env, state_term_name)
    return np.asarray(grasp.palm_supported, dtype=get_global_dtype())


def thumb_contact(env: _LeapEnv, state_term_name: str) -> np.ndarray:
    grasp, _ = _grasp_state(env, state_term_name)
    return np.asarray(grasp.thumb_contact, dtype=get_global_dtype())


def grasp_state_metric(env: _LeapEnv, state_term_name: str, field: str) -> np.ndarray:
    grasp, _ = _grasp_state(env, state_term_name)
    values = {
        "valid_grasp": grasp.valid_grasp,
        "vertical_drop": grasp.vertical_drop,
        "lateral_drift": grasp.lateral_drift,
        "tilt_deg": np.rad2deg(grasp.tilt_rad),
        "thumb_contact": grasp.thumb_contact,
        "index_contact": grasp.opposing_contacts[:, 0],
        "middle_contact": grasp.opposing_contacts[:, 1],
        "outer_contact": grasp.opposing_contacts[:, 2],
        "opposing_contact_count": grasp.opposing_contact_count,
        "palm_supported": grasp.palm_supported,
    }
    try:
        value = values[field]
    except KeyError:
        raise ValueError(
            f"grasp_state_metric field must be one of {sorted(values)}, got {field!r}"
        ) from None
    return np.asarray(value, dtype=get_global_dtype())


def gait_state_metric(env: _LeapEnv, action_name: str, field: str) -> np.ndarray:
    name = _name("gait_state_metric", "action_name", action_name)
    action = env.action_manager.get_term(name)
    if not isinstance(action, LeapIncrementalPositionAction):
        raise TypeError(
            f"gait_state_metric action {name!r} must be LeapIncrementalPositionAction, "
            f"got {type(action).__name__}"
        )
    values = {
        "amplitude": np.full(env.num_envs, action.gait_amplitude),
        "thumb_amplitude": np.full(env.num_envs, action.thumb_gait_amplitude),
        "reference_l2": np.linalg.norm(action.gait_reference, axis=1),
        "thumb_reference_l2": np.linalg.norm(action.gait_reference[:, 12:], axis=1),
        "thumb_policy_offset_l2": np.linalg.norm(
            action.policy_target[:, 12:] - action.episode_anchor[:, 12:], axis=1
        ),
        "index_finger_policy_offset_l2": np.linalg.norm(
            action.policy_target[:, :4] - action.episode_anchor[:, :4], axis=1
        ),
        "middle_finger_policy_offset_l2": np.linalg.norm(
            action.policy_target[:, 4:8] - action.episode_anchor[:, 4:8], axis=1
        ),
        "outer_finger_policy_offset_l2": np.linalg.norm(
            action.policy_target[:, 8:12] - action.episode_anchor[:, 8:12], axis=1
        ),
        "residual_l2": np.linalg.norm(action.raw_action, axis=1),
        "command_noise_l2": np.linalg.norm(action.command_noise, axis=1),
        "command_delayed": action.command_delayed,
        "action_noise_std": np.full(env.num_envs, action.action_noise_std),
        "action_delay_probability": np.full(env.num_envs, action.action_delay_probability),
    }
    try:
        value = values[field]
    except KeyError:
        raise ValueError(
            f"gait_state_metric field must be one of {sorted(values)}, got {field!r}"
        ) from None
    return np.asarray(value, dtype=get_global_dtype())


def reset_randomization_metric(env: _LeapEnv, event_name: str, field: str) -> np.ndarray:
    """Expose the most recently sampled reset perturbation for each environment."""
    name = _name("reset_randomization_metric", "event_name", event_name)
    reset = env.event_manager.get_term_cfg(name).func
    if not isinstance(reset, LeapHandObjectReset):
        raise TypeError(
            f"reset_randomization_metric event {name!r} must be LeapHandObjectReset, "
            f"got {type(reset).__name__}"
        )
    values = {
        "randomized": reset.randomized,
        "joint_offset_l2": reset.joint_offset_l2,
        "position_offset_l2": reset.position_offset_l2,
        "tilt_offset_deg": reset.tilt_offset_deg,
        "linear_velocity_l2": reset.linear_velocity_l2,
        "angular_velocity_l2": reset.angular_velocity_l2,
    }
    try:
        value = values[field]
    except KeyError:
        raise ValueError(
            f"reset_randomization_metric field must be one of {sorted(values)}, got {field!r}"
        ) from None
    return np.asarray(value, dtype=get_global_dtype())


def frontier_state_metric(env: _LeapEnv, reward_name: str, field: str) -> np.ndarray:
    name = _name("frontier_state_metric", "reward_name", reward_name)
    reward = env.reward_manager.get_term_cfg(name).func
    if not isinstance(reward, LeapBidirectionalFrontierReward):
        raise TypeError(
            f"frontier_state_metric reward {name!r} must be "
            f"LeapBidirectionalFrontierReward, got {type(reward).__name__}"
        )
    # RewardManager deliberately skips zero-weight terms.  Frontier tracking is
    # also used as a diagnostic in HORA-style owners where it must not shape the
    # policy, so let the metrics path advance the stateful tracker.  The term's
    # common-step guard makes this a no-op when a nonzero reward already ran it.
    reward(env)
    values = {
        "held_angle": reward.held_angle,
        "absolute_held_angle": np.abs(reward.held_angle),
        "frontier_angle": reward.frontier_angle,
        "frontier_progress": reward.frontier_progress,
        "frontier_speed": reward.frontier_speed,
        "frontier_progress_multiplier": reward.frontier_progress_multiplier,
        "directional_speed": reward.directional_speed,
        "filtered_directional_speed": reward.filtered_directional_speed,
        "fast_rotation_bonus": reward.fast_rotation_bonus,
        "direction_committed": reward.committed_direction != 0,
        "committed_time": reward.committed_time,
        "opposite_direction_speed": reward.opposite_direction_speed,
        "reversal_penalty_event": reward.reversal_penalty_event,
        "stalled": reward.stalled,
        "milestone_bonus": reward.milestone_bonus,
        "milestones_reached": reward.milestones_reached,
        "direction_reversals": reward.direction_reversals,
    }
    try:
        value = values[field]
    except KeyError:
        raise ValueError(
            f"frontier_state_metric field must be one of {sorted(values)}, got {field!r}"
        ) from None
    return np.asarray(value, dtype=get_global_dtype())


def successful_valid_grasp(
    env: _LeapEnv,
    state_term_name: str,
    time_out_term_name: str,
) -> np.ndarray:
    grasp, _ = _grasp_state(env, state_term_name)
    time_out_name = _name("successful_valid_grasp", "time_out_term_name", time_out_term_name)
    time_out = env.termination_manager.get_term(time_out_name)
    return np.asarray(grasp.valid_grasp * time_out, dtype=get_global_dtype())


def successful_rotating_grasp(
    env: _LeapEnv,
    state_term_name: str,
    time_out_term_name: str,
    rotation_reward_name: str,
    minimum_frontier_angle: float,
) -> np.ndarray:
    """Pay timeout success after enough held rotation with secure finger contact."""
    grasp, _ = _grasp_state(env, state_term_name)
    term = "successful_rotating_grasp"
    time_out = env.termination_manager.get_term(
        _name(term, "time_out_term_name", time_out_term_name)
    )
    reward_name = _name(term, "rotation_reward_name", rotation_reward_name)
    rotation = env.reward_manager.get_term_cfg(reward_name).func
    if not isinstance(rotation, LeapBidirectionalFrontierReward):
        raise TypeError(
            f"{term} reward {reward_name!r} must be LeapBidirectionalFrontierReward, "
            f"got {type(rotation).__name__}"
        )
    if rotation._last_counter != int(env.common_step_counter):
        raise RuntimeError(
            f"{term} rotation reward {reward_name!r} was not computed for control step "
            f"{env.common_step_counter}"
        )
    threshold = _real(
        term,
        "minimum_frontier_angle",
        minimum_frontier_angle,
        minimum=0.0,
        strict_minimum=True,
    )
    return np.asarray(
        ~grasp.fallen * grasp.contact_ok * time_out * (rotation.frontier_angle >= threshold),
        dtype=get_global_dtype(),
    )


def successful_retention(
    env: _LeapEnv,
    state_term_name: str,
    time_out_term_name: str,
    minimum_object_height: float,
    maximum_object_xy_drift: float,
) -> np.ndarray:
    retained = object_retained(
        env,
        state_term_name,
        minimum_object_height,
        maximum_object_xy_drift,
    )
    time_out_name = _name("successful_retention", "time_out_term_name", time_out_term_name)
    return np.asarray(
        retained * env.termination_manager.get_term(time_out_name), dtype=get_global_dtype()
    )


def object_linear_velocity_l1(env: _LeapEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    return np.asarray(np.sum(np.abs(state.object_linvel), axis=1), dtype=get_global_dtype())


def hand_pose_deviation_l2(env: _LeapEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    return np.asarray(
        np.sum(np.square(state.dof_pos - state.init_pose), axis=1), dtype=get_global_dtype()
    )


def estimated_torque_l2(env: _LeapEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    return np.asarray(np.sum(np.square(state.torques), axis=1), dtype=get_global_dtype())


def estimated_work_l2(env: _LeapEnv, state_term_name: str) -> np.ndarray:
    _, state = _rotation_state(env, state_term_name)
    work = np.sum(state.torques * state.dof_vel, axis=1)
    return np.asarray(np.square(work), dtype=get_global_dtype())


def dropped(env: _LeapEnv, state_term_name: str) -> np.ndarray:
    state, _ = _rotation_state(env, state_term_name)
    return np.asarray(state.dropped, dtype=get_global_dtype())


class LeapHandObjectReset(ManagerTermBase):
    """Reset hand joints and the object root without exposing qpos layout."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term = type(self).__name__
        allowed = {
            "entity_name",
            "grasp_cache_path",
            "grasp_cache_sampling",
            "grasp_cache_start_index",
            "joint_noise",
            "object_velocity_noise",
            "object_z_offset",
            "randomization_fraction",
            "object_xy_noise",
            "object_z_noise",
            "object_tilt_noise_deg",
            "object_angular_velocity_noise",
        }
        unexpected = set(cfg.params) - allowed
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        entity_name = _name(term, "entity_name", cfg.params.get("entity_name"))
        self._entity = cast("Entity", env.scene[entity_name])
        if self._entity.num_joints != 16:
            raise ValueError(f"{term} requires 16 hand joints, got {self._entity.num_joints}")
        if self._entity.data.default_root_state.shape != (env.num_envs, 13):
            raise ValueError(f"{term} requires a 13-D floating object root for every environment")
        self._validate_randomization_params(cfg.params)

        cache_value = cfg.params.get("grasp_cache_path")
        self._grasp_cache: np.ndarray | None = None
        self._grasp_cache_sampling = str(cfg.params.get("grasp_cache_sampling", "random"))
        if self._grasp_cache_sampling not in {"random", "sequential"}:
            raise ValueError(
                f"{term} grasp_cache_sampling must be 'random' or 'sequential', "
                f"got {self._grasp_cache_sampling!r}"
            )
        cache_start_index = cfg.params.get("grasp_cache_start_index", 0)
        if isinstance(cache_start_index, (bool, np.bool_)) or not isinstance(
            cache_start_index, (int, np.integer)
        ):
            raise TypeError(f"{term} grasp_cache_start_index must be an integer")
        if int(cache_start_index) < 0:
            raise ValueError(f"{term} grasp_cache_start_index must be nonnegative")
        self._grasp_cache_cursor = int(cache_start_index)
        if cache_value is not None:
            cache_path = _resolve_grasp_cache(_name(term, "grasp_cache_path", cache_value))
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"{term} configured grasp cache does not exist: {cache_path}. "
                    "Set grasp_cache_path to null to use the model home pose, "
                    "or generate a compatible LEAP grasp cache first."
                )
            cache = np.asarray(np.load(cache_path), dtype=np.float64)
            if cache.ndim != 2 or cache.shape[1] != 23 or cache.shape[0] == 0:
                raise ValueError(
                    f"{term} grasp cache {cache_path} must have shape (N, 23), got {cache.shape}"
                )
            if not np.isfinite(cache).all():
                raise ValueError(f"{term} grasp cache {cache_path} contains NaN or Inf")
            self._grasp_cache = cache
            self._grasp_cache_cursor %= cache.shape[0]
        elif self._grasp_cache_sampling != "random" or self._grasp_cache_cursor != 0:
            raise ValueError(
                f"{term} cache sampling controls require a configured grasp_cache_path"
            )

        ranges = np.asarray(self._entity.data.actuator_ctrl_range, dtype=np.float64)
        if ranges.shape != (16, 2):
            raise ValueError(f"{term} actuator control range must have shape (16, 2)")
        self._ctrl_lower = ranges[:, 0]
        self._ctrl_upper = ranges[:, 1]
        dtype = get_global_dtype()
        self.randomized = np.zeros(env.num_envs, dtype=np.bool_)
        self.joint_offset_l2 = np.zeros(env.num_envs, dtype=dtype)
        self.position_offset_l2 = np.zeros(env.num_envs, dtype=dtype)
        self.tilt_offset_deg = np.zeros(env.num_envs, dtype=dtype)
        self.linear_velocity_l2 = np.zeros(env.num_envs, dtype=dtype)
        self.angular_velocity_l2 = np.zeros(env.num_envs, dtype=dtype)
        self.joint_offset = np.zeros((env.num_envs, 16), dtype=dtype)
        self.position_offset = np.zeros((env.num_envs, 3), dtype=dtype)
        self.tilt_axis = np.zeros((env.num_envs, 3), dtype=dtype)
        self.linear_velocity = np.zeros((env.num_envs, 3), dtype=dtype)
        self.angular_velocity = np.zeros((env.num_envs, 3), dtype=dtype)
        self.grasp_cache_index = np.full(env.num_envs, -1, dtype=np.int64)

    @staticmethod
    def _validate_randomization_params(params: Mapping[str, Any]) -> tuple[float, ...]:
        term = "LeapHandObjectReset"
        fraction = _real(
            term, "randomization_fraction", params.get("randomization_fraction"), minimum=0.0
        )
        if fraction > 1.0:
            raise ValueError(f"{term} randomization_fraction must be at most 1.0")
        return (
            fraction,
            _real(term, "joint_noise", params.get("joint_noise"), minimum=0.0),
            _real(term, "object_xy_noise", params.get("object_xy_noise"), minimum=0.0),
            _real(term, "object_z_noise", params.get("object_z_noise"), minimum=0.0),
            np.deg2rad(
                _real(
                    term,
                    "object_tilt_noise_deg",
                    params.get("object_tilt_noise_deg"),
                    minimum=0.0,
                )
            ),
            _real(
                term,
                "object_velocity_noise",
                params.get("object_velocity_noise"),
                minimum=0.0,
            ),
            _real(
                term,
                "object_angular_velocity_noise",
                params.get("object_angular_velocity_noise"),
                minimum=0.0,
            ),
            _real(term, "object_z_offset", params.get("object_z_offset")),
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        **params: Any,
    ) -> None:
        (
            randomization_fraction,
            joint_noise,
            object_xy_noise,
            object_z_noise,
            object_tilt_noise_rad,
            object_velocity_noise,
            object_angular_velocity_noise,
            object_z_offset,
        ) = self._validate_randomization_params(params)
        ids = _env_ids(env, env_ids)
        count = ids.size
        dtype = get_global_dtype()
        root_state = np.array(self._entity.data.default_root_state[ids], copy=True)
        if self._grasp_cache is not None:
            if self._grasp_cache_sampling == "sequential":
                cache_indices = (
                    np.arange(count, dtype=np.int64) + self._grasp_cache_cursor
                ) % self._grasp_cache.shape[0]
                self._grasp_cache_cursor = int(
                    (self._grasp_cache_cursor + count) % self._grasp_cache.shape[0]
                )
            else:
                cache_indices = env.rng.integers(
                    0,
                    self._grasp_cache.shape[0],
                    size=count,
                    dtype=np.int64,
                )
            self.grasp_cache_index[ids] = cache_indices
            rows = self._grasp_cache[cache_indices]
            joint_pos = np.array(rows[:, :16], copy=True)
            root_state[:, :3] = rows[:, 16:19]
            root_state[:, 3:7] = rows[:, 19:23]
        else:
            self.grasp_cache_index[ids] = -1
            joint_pos = np.array(self._entity.data.default_joint_pos[ids], copy=True)
        randomized = env.rng.random(count) < randomization_fraction
        joint_offset = np.zeros_like(joint_pos)
        if joint_noise > 0.0 and np.any(randomized):
            joint_offset[randomized] = env.rng.uniform(
                -joint_noise, joint_noise, size=(int(np.count_nonzero(randomized)), 16)
            )
            joint_pos += joint_offset

        position_offset = np.zeros((count, 3), dtype=root_state.dtype)
        randomized_count = int(np.count_nonzero(randomized))
        if object_xy_noise > 0.0 and randomized_count:
            position_offset[randomized, :2] = env.rng.uniform(
                -object_xy_noise, object_xy_noise, size=(randomized_count, 2)
            )
        if object_z_noise > 0.0 and randomized_count:
            position_offset[randomized, 2] = env.rng.uniform(
                -object_z_noise, object_z_noise, size=randomized_count
            )
        root_state[:, :3] += position_offset
        root_state[:, 2] += object_z_offset

        tilt_angle = np.zeros(count, dtype=root_state.dtype)
        tilt_axis = np.zeros((count, 3), dtype=root_state.dtype)
        if object_tilt_noise_rad > 0.0 and randomized_count:
            tilt_angle[randomized] = object_tilt_noise_rad * np.sqrt(
                env.rng.random(randomized_count)
            )
            azimuth = env.rng.uniform(-np.pi, np.pi, size=randomized_count)
            tilt_axis[randomized] = np.stack(
                (np.cos(azimuth), np.sin(azimuth), np.zeros(randomized_count)), axis=1
            )
            tilt_quat = np_quat_from_angle_axis(tilt_angle[randomized], tilt_axis[randomized])
            root_state[randomized, 3:7] = np_quat_mul(root_state[randomized, 3:7], tilt_quat)
        np.clip(joint_pos, self._ctrl_lower, self._ctrl_upper, out=joint_pos)
        joint_vel = np.zeros_like(joint_pos)
        root_state[:, 7:] = 0.0
        if object_velocity_noise > 0.0 and randomized_count:
            root_state[randomized, 7:10] = env.rng.uniform(
                -object_velocity_noise,
                object_velocity_noise,
                size=(randomized_count, 3),
            )
        if object_angular_velocity_noise > 0.0 and randomized_count:
            root_state[randomized, 10:13] = env.rng.uniform(
                -object_angular_velocity_noise,
                object_angular_velocity_noise,
                size=(randomized_count, 3),
            )
        self.randomized[ids] = randomized
        self.joint_offset_l2[ids] = np.linalg.norm(joint_offset, axis=1)
        self.position_offset_l2[ids] = np.linalg.norm(position_offset, axis=1)
        self.tilt_offset_deg[ids] = np.rad2deg(tilt_angle)
        self.linear_velocity_l2[ids] = np.linalg.norm(root_state[:, 7:10], axis=1)
        self.angular_velocity_l2[ids] = np.linalg.norm(root_state[:, 10:13], axis=1)
        self.joint_offset[ids] = joint_offset
        self.position_offset[ids] = position_offset
        self.tilt_axis[ids] = tilt_axis
        self.linear_velocity[ids] = root_state[:, 7:10]
        self.angular_velocity[ids] = root_state[:, 10:13]
        self._entity.write_joint_state_to_sim(
            np.asarray(joint_pos, dtype=dtype),
            np.asarray(joint_vel, dtype=dtype),
            env_ids=ids,
        )
        self._entity.write_root_link_pose_to_sim(
            np.asarray(root_state[:, :7], dtype=dtype), env_ids=ids
        )
        self._entity.write_root_link_velocity_to_sim(
            np.asarray(root_state[:, 7:], dtype=dtype), env_ids=ids
        )


__all__ = [
    "LeapAxialRotateReward",
    "LeapGraspTermination",
    "LeapDropTermination",
    "LeapHandObjectReset",
    "LeapIncrementalPositionAction",
    "LeapIncrementalPositionActionCfg",
    "LeapProprioObservation",
    "LeapRotateReward",
    "LeapRotationObservation",
    "dropped",
    "estimated_torque_l2",
    "estimated_work_l2",
    "grasp_quality",
    "grasp_state_metric",
    "hand_pose_deviation_l2",
    "object_retained",
    "object_retention_quality",
    "object_linear_velocity_l1",
    "palm_support",
    "reset_randomization_metric",
    "successful_rotating_grasp",
    "successful_retention",
    "successful_valid_grasp",
    "thumb_contact",
]
