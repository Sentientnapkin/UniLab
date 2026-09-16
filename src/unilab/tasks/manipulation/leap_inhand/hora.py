"""HORA-style Stage 1 components for LEAP in-hand rotation.

The gait-assisted policy remains in :mod:`manager_terms`.  This module owns the
separate no-gait action/observation contract and the jointly-trained privileged
extrinsics encoder used by the Stage 1 actor.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from tensordict import TensorDict

from unilab.dtype_config import get_global_dtype
from unilab.managers import ManagerTermBase, ManagerTermBaseCfg

from .manager_terms import (
    LeapGraspTermination,
    LeapIncrementalPositionAction,
    LeapIncrementalPositionActionCfg,
    _env_ids,
    _name,
    _pair,
    _real,
    _rotation_state,
)

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


PROPERTY_NAMES = (
    "cylinder_radius",
    "cylinder_length",
    "object_mass",
    "sliding_friction",
    "torsional_friction",
    "com_x",
    "com_y",
    "com_z",
    "pd_gain_scale",
    "latency_seconds",
)


@dataclass(kw_only=True)
class LeapHoraPositionActionCfg(LeapIncrementalPositionActionCfg):
    """Bounded position offsets around the episode's sampled grasp pose."""

    target_offset_limit: float = 0.25
    latency_steps_range: tuple[int, int] | list[int] = (0, 0)

    def build(self, env: ManagerBasedRlEnv) -> LeapHoraPositionAction:
        return LeapHoraPositionAction(self, env)


class LeapHoraPositionAction(LeapIncrementalPositionAction):
    """No-gait, non-integrating joint targets suitable for HORA Stage 1.

    The policy action is interpreted as a bounded offset from the grasp sampled
    at reset.  A latency in whole 20 Hz policy steps is sampled once per episode,
    making latency a coherent privileged property rather than per-step noise.
    """

    cfg: LeapHoraPositionActionCfg  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, cfg: LeapHoraPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        term = type(self).__name__
        self._target_offset_limit = _real(
            term,
            "target_offset_limit",
            cfg.target_offset_limit,
            minimum=0.0,
            strict_minimum=True,
        )
        values = cfg.latency_steps_range
        if isinstance(values, (str, bytes)) or not isinstance(values, (tuple, list)):
            raise TypeError(f"{term} latency_steps_range must be an integer pair")
        if len(values) != 2 or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in values
        ):
            raise TypeError(f"{term} latency_steps_range must be an integer pair")
        lower, upper = int(values[0]), int(values[1])
        if lower < 0 or lower > upper:
            raise ValueError(f"{term} latency_steps_range must satisfy 0 <= lower <= upper")
        self._latency_steps_range = (lower, upper)
        self._latency_steps = np.zeros(env.num_envs, dtype=np.int64)
        self._action_history = np.zeros(
            (env.num_envs, upper + 1, self.action_dim), dtype=get_global_dtype()
        )

    @property
    def latency_steps(self) -> np.ndarray:
        return self._latency_steps

    @property
    def latency_seconds(self) -> np.ndarray:
        return self._latency_steps.astype(get_global_dtype()) * self._env.step_dt

    def process_actions(self, actions: np.ndarray) -> None:
        if not isinstance(actions, np.ndarray):
            raise TypeError(f"expected np.ndarray actions, got {type(actions).__name__}")
        if actions.shape != self._raw_action.shape:
            raise ValueError(f"expected action shape {self._raw_action.shape}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("received NaN or Inf actions")
        self._raw_action[:] = actions
        self._action_history[:, 1:] = self._action_history[:, :-1]
        self._action_history[:, 0] = actions
        self._effective_action[:] = self._action_history[
            np.arange(self._env.num_envs), self._latency_steps
        ]
        np.clip(
            self._effective_action,
            self._raw_clip[0],
            self._raw_clip[1],
            out=self._clipped_action,
        )
        offset = np.clip(
            self._scale * self._clipped_action,
            -self._target_offset_limit,
            self._target_offset_limit,
        )
        np.add(self._episode_anchor, offset, out=self._policy_target)
        np.clip(self._policy_target, self._ctrl_lower, self._ctrl_upper, out=self._policy_target)
        self._gait_reference.fill(0.0)
        self._target[:] = self._policy_target

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        super().reset(env_ids)
        ids = _env_ids(self._env, env_ids)
        self._action_history[ids] = 0.0
        lower, upper = self._latency_steps_range
        self._latency_steps[ids] = self._env.rng.integers(lower, upper + 1, size=len(ids))


class LeapHoraProprioObservation(ManagerTermBase):
    """One hardware-compatible frame: measured joints and commanded targets."""

    _ALLOWED_PARAMS = frozenset({"entity_name", "action_name", "joint_noise"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term = type(self).__name__
        unexpected = set(cfg.params) - self._ALLOWED_PARAMS
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        entity_name = _name(term, "entity_name", cfg.params.get("entity_name"))
        action_name = _name(term, "action_name", cfg.params.get("action_name"))
        self._entity = cast("Entity", env.scene[entity_name])
        action = env.action_manager.get_term(action_name)
        if not isinstance(action, LeapHoraPositionAction):
            raise TypeError(f"{term} action {action_name!r} must be LeapHoraPositionAction")
        self._action = action
        self._joint_noise = _real(term, "joint_noise", cfg.params.get("joint_noise"), minimum=0.0)
        self._dof_mid = (action.ctrl_upper + action.ctrl_lower) / 2.0
        self._dof_range = action.ctrl_upper - action.ctrl_lower
        self._just_reset = np.ones(env.num_envs, dtype=np.bool_)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        self._just_reset[_env_ids(self._env, env_ids)] = True

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        dof_pos = np.asarray(
            self._entity.data.joint_pos[:, self._action.joint_ids], dtype=get_global_dtype()
        )
        normalized = 2.0 * (dof_pos - self._dof_mid) / (self._dof_range + 1.0e-8)
        active = ~self._just_reset
        if self._joint_noise > 0.0 and np.any(active):
            normalized = normalized.copy()
            normalized[active] += env.rng.uniform(
                -self._joint_noise,
                self._joint_noise,
                size=(int(np.count_nonzero(active)), self._action.action_dim),
            )
        self._just_reset[:] = False
        return np.concatenate((normalized, self._action.target), axis=1).astype(
            get_global_dtype(), copy=False
        )


def normalize_object_properties(
    values: np.ndarray,
    ranges: Mapping[str, Sequence[float]],
) -> np.ndarray:
    """Normalize declared physical properties to ``[-1, 1]`` in canonical order."""
    if values.shape[-1] != len(PROPERTY_NAMES):
        raise ValueError(
            f"property values must end in {len(PROPERTY_NAMES)} fields, got {values.shape}"
        )
    if set(ranges) != set(PROPERTY_NAMES):
        raise ValueError(
            f"property_ranges keys must be exactly {list(PROPERTY_NAMES)}, got {sorted(ranges)}"
        )
    bounds = np.asarray(
        [_pair("LeapObjectProperties", name, ranges[name]) for name in PROPERTY_NAMES]
    )
    widths = bounds[:, 1] - bounds[:, 0]
    if np.any(widths <= 0.0):
        raise ValueError("every property normalization range must have nonzero width")
    if not np.isfinite(values).all():
        raise ValueError("object properties must be finite")
    # Held-out OOD values intentionally extrapolate beyond [-1, 1]; clipping
    # would hide how far a deployment object lies outside the training domain.
    return np.asarray(2.0 * (values - bounds[:, 0]) / widths - 1.0, dtype=get_global_dtype())


class LeapObjectProperties(ManagerTermBase):
    """Task-owned provider for normalized simulator-only object properties."""

    _ALLOWED_PARAMS = frozenset(
        {
            "action_name",
            "mass_event_name",
            "friction_event_name",
            "com_event_name",
            "pd_event_name",
            "geometry_event_name",
            "nominal_values",
            "property_ranges",
        }
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term = type(self).__name__
        unexpected = set(cfg.params) - self._ALLOWED_PARAMS
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        nominal = cfg.params.get("nominal_values")
        ranges = cfg.params.get("property_ranges")
        if not isinstance(nominal, Mapping) or set(nominal) != set(PROPERTY_NAMES):
            raise ValueError(f"{term} nominal_values keys must be exactly {list(PROPERTY_NAMES)}")
        if not isinstance(ranges, Mapping):
            raise TypeError(f"{term} property_ranges must be a mapping")
        self._nominal = np.asarray([nominal[name] for name in PROPERTY_NAMES], dtype=np.float64)
        if not np.isfinite(self._nominal).all():
            raise ValueError(f"{term} nominal_values must be finite")
        self._ranges = dict(ranges)
        normalize_object_properties(self._nominal[None, :], self._ranges)
        self._action_name = _name(term, "action_name", cfg.params.get("action_name"))
        self._event_names = {
            "geometry": _name(term, "geometry_event_name", cfg.params.get("geometry_event_name")),
            "mass": _name(term, "mass_event_name", cfg.params.get("mass_event_name")),
            "friction": _name(term, "friction_event_name", cfg.params.get("friction_event_name")),
            "com": _name(term, "com_event_name", cfg.params.get("com_event_name")),
            "pd": _name(term, "pd_event_name", cfg.params.get("pd_event_name")),
        }

    @property
    def property_names(self) -> tuple[str, ...]:
        return PROPERTY_NAMES

    def raw_values(self, env: ManagerBasedRlEnv) -> np.ndarray:
        values = np.broadcast_to(self._nominal, (env.num_envs, len(PROPERTY_NAMES))).copy()
        geometry_term = env.event_manager.get_term_cfg(self._event_names["geometry"]).func
        radius = getattr(geometry_term, "current_radius", None)
        length = getattr(geometry_term, "current_length", None)
        if radius is not None and length is not None:
            values[:, 0] = np.asarray(radius)
            values[:, 1] = np.asarray(length)
        mass_term = env.event_manager.get_term_cfg(self._event_names["mass"]).func
        mass = getattr(mass_term, "mass", None)
        scales = getattr(mass_term, "scales", None)
        if mass is not None and np.isfinite(mass).all():
            values[:, 2] = np.asarray(mass)
        elif scales is not None:
            values[:, 2] = self._nominal[2] * np.asarray(scales)[:, 0]
        friction = getattr(
            env.event_manager.get_term_cfg(self._event_names["friction"]).func,
            "last_values",
            None,
        )
        if friction is not None and np.isfinite(friction).all():
            values[:, 3:5] = np.asarray(friction)[:, 0, :2]
        offsets = getattr(
            env.event_manager.get_term_cfg(self._event_names["com"]).func,
            "last_offsets",
            None,
        )
        if offsets is not None and np.isfinite(offsets).all():
            values[:, 5:8] = np.asarray(offsets)[:, 0]
        pd_term = env.event_manager.get_term_cfg(self._event_names["pd"]).func
        last_kp = getattr(pd_term, "last_kp", None)
        if last_kp is not None and np.isfinite(last_kp).all():
            values[:, 8] = np.mean(np.asarray(last_kp), axis=1) / self._nominal[8]
        action = env.action_manager.get_term(self._action_name)
        if not isinstance(action, LeapHoraPositionAction):
            raise TypeError("LeapObjectProperties requires LeapHoraPositionAction")
        values[:, 9] = action.latency_seconds
        return values

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        return normalize_object_properties(self.raw_values(env), self._ranges)


def hora_axial_rotation(
    env: ManagerBasedRlEnv,
    state_term_name: str,
    clip_min: float,
    clip_max: float,
    **unused_legacy_params: Any,
) -> np.ndarray:
    """Clipped axial angular velocity, gated only by geometric failure."""
    # Hydra merges mappings inherited from the gait owner's stateful ``rotate``
    # term. They remain inert here and keep the isolated owner a small overlay.
    del unused_legacy_params
    grasp, state = _rotation_state(cast(Any, env), state_term_name)
    if not isinstance(grasp, LeapGraspTermination):
        raise TypeError("hora_axial_rotation requires LeapGraspTermination")
    clip = _pair("hora_axial_rotation", "clip", (clip_min, clip_max))
    axial_velocity = np.sum(state.object_angvel * grasp.cylinder_axis_w, axis=1)
    return np.asarray(
        np.clip(axial_velocity, clip[0], clip[1]) * ~grasp.fallen,
        dtype=get_global_dtype(),
    )


def joint_limit_occupancy(
    env: ManagerBasedRlEnv,
    action_name: str,
    margin_fraction: float = 0.02,
) -> np.ndarray:
    """Fraction of hand joints within a declared fraction of either limit."""
    action = env.action_manager.get_term(action_name)
    if not isinstance(action, LeapIncrementalPositionAction):
        raise TypeError("joint_limit_occupancy requires a LEAP position action")
    margin = _real("joint_limit_occupancy", "margin_fraction", margin_fraction, minimum=0.0)
    if margin > 0.5:
        raise ValueError("joint_limit_occupancy margin_fraction must be at most 0.5")
    positions = np.asarray(action._entity.data.joint_pos[:, action.joint_ids])
    span = action.ctrl_upper - action.ctrl_lower
    near = (positions <= action.ctrl_lower + margin * span) | (
        positions >= action.ctrl_upper - margin * span
    )
    return np.asarray(np.mean(near, axis=1), dtype=get_global_dtype())


class ExtrinsicsEncoder(torch.nn.Module):
    """Small MLP mapping normalized simulator properties to compact extrinsics."""

    def __init__(
        self,
        property_dim: int = len(PROPERTY_NAMES),
        extrinsics_dim: int = 8,
        hidden_dims: Sequence[int] = (32, 32),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        if property_dim <= 0 or extrinsics_dim <= 0:
            raise ValueError("property_dim and extrinsics_dim must be positive")
        self.property_dim = int(property_dim)
        self.extrinsics_dim = int(extrinsics_dim)
        self.network = MLP(property_dim, extrinsics_dim, list(hidden_dims), activation)

    def forward(self, properties: torch.Tensor) -> torch.Tensor:
        if properties.shape[-1] != self.property_dim:
            raise ValueError(
                f"expected {self.property_dim} properties, got shape {tuple(properties.shape)}"
            )
        return self.network(properties)


class HoraActor(MLPModel):
    """RSL-RL actor with an explicit jointly-optimized extrinsics encoder."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        property_dim: int = len(PROPERTY_NAMES),
        extrinsics_dim: int = 8,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (32, 32),
        extrinsics_mode: Literal["exact", "zero", "shuffled", "frozen", "predicted"] = "exact",
    ) -> None:
        self.property_dim = int(property_dim)
        self.extrinsics_dim = int(extrinsics_dim)
        self._hora_input_dim: int | None = None
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            False,
            distribution_cfg,
        )
        if self.obs_dim <= self.property_dim:
            raise ValueError("HoraActor observation must include hardware history and properties")
        self.hardware_dim = self.obs_dim - self.property_dim
        self.obs_normalization = bool(obs_normalization)
        self.hardware_normalizer = (
            EmpiricalNormalization(self.hardware_dim) if obs_normalization else torch.nn.Identity()
        )
        self.extrinsics_encoder = ExtrinsicsEncoder(
            self.property_dim,
            self.extrinsics_dim,
            encoder_hidden_dims,
            activation,
        )
        self.set_extrinsics_mode(extrinsics_mode)
        self._frozen_extrinsics: torch.Tensor | None = None
        self._predicted_extrinsics: torch.Tensor | None = None

    def _get_latent_dim(self) -> int:
        # Called by MLPModel before the HORA-owned modules are constructed.
        return int(self.obs_dim - self.property_dim + self.extrinsics_dim)

    def set_extrinsics_mode(self, mode: str) -> None:
        allowed = {"exact", "zero", "shuffled", "frozen", "predicted"}
        if mode not in allowed:
            raise ValueError(f"extrinsics mode must be one of {sorted(allowed)}, got {mode!r}")
        self.extrinsics_mode = mode

    def set_predicted_extrinsics(self, extrinsics: torch.Tensor | None) -> None:
        if extrinsics is not None and extrinsics.shape[-1] != self.extrinsics_dim:
            raise ValueError(f"predicted extrinsics must end in {self.extrinsics_dim} values")
        self._predicted_extrinsics = extrinsics

    def _split_observation(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        return combined[..., : -self.property_dim], combined[..., -self.property_dim :]

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del masks, hidden_state
        hardware, properties = self._split_observation(obs)
        hardware = self.hardware_normalizer(hardware)
        extrinsics = self.extrinsics_encoder(properties)
        if self.extrinsics_mode == "zero":
            extrinsics = torch.zeros_like(extrinsics)
        elif self.extrinsics_mode == "shuffled" and extrinsics.shape[0] > 1:
            extrinsics = torch.roll(extrinsics, shifts=1, dims=0)
        elif self.extrinsics_mode == "frozen":
            if self._frozen_extrinsics is None or self._frozen_extrinsics.shape != extrinsics.shape:
                self._frozen_extrinsics = extrinsics.detach().clone()
            extrinsics = cast(torch.Tensor, self._frozen_extrinsics)
        elif self.extrinsics_mode == "predicted":
            predicted = self._predicted_extrinsics
            if predicted is None:
                raise RuntimeError("predicted extrinsics mode requires set_predicted_extrinsics()")
            extrinsics = predicted.to(extrinsics)
        return torch.cat((hardware, extrinsics), dim=-1)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        del hidden_state
        if dones is None or bool(torch.all(dones)):
            self._frozen_extrinsics = None

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            hardware, _ = self._split_observation(obs)
            self.hardware_normalizer.update(hardware)  # type: ignore[union-attr]

    def as_jit(self) -> torch.nn.Module:
        return _TorchHoraActor(self)

    def as_onnx(self, verbose: bool) -> torch.nn.Module:
        return _OnnxHoraActor(self, verbose)


class _TorchHoraActor(torch.nn.Module):
    """Deterministic exact-extrinsics deployment wrapper for Stage 1."""

    def __init__(self, model: HoraActor) -> None:
        super().__init__()
        self.property_dim = model.property_dim
        self.hardware_normalizer = copy.deepcopy(model.hardware_normalizer)
        self.extrinsics_encoder = copy.deepcopy(model.extrinsics_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else torch.nn.Identity()
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hardware = self.hardware_normalizer(observation[..., : -self.property_dim])
        properties = observation[..., -self.property_dim :]
        latent = torch.cat((hardware, self.extrinsics_encoder(properties)), dim=-1)
        return self.deterministic_output(self.mlp(latent))

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxHoraActor(_TorchHoraActor):
    """Exact-extrinsics Stage 1 actor with the RSL-RL ONNX interface."""

    is_recurrent: bool = False

    def __init__(self, model: HoraActor, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


__all__ = [
    "ExtrinsicsEncoder",
    "HoraActor",
    "LeapHoraPositionAction",
    "LeapHoraPositionActionCfg",
    "LeapHoraProprioObservation",
    "LeapObjectProperties",
    "PROPERTY_NAMES",
    "hora_axial_rotation",
    "joint_limit_occupancy",
    "normalize_object_properties",
]
