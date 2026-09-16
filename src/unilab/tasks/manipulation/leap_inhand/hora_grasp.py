"""Geometry-aware precision-grasp generation for the HORA-style LEAP task.

The legacy gait baseline intentionally keeps its compact ``(N, 23)`` NumPy
cache.  This module owns the richer HORA cache contract so object geometry and
grasp-family splits cannot be lost when caches are merged or consumed later.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
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
from unilab.utils.rotation import np_quat_from_angle_axis, np_quat_mul

from .grasp_gen import LeapGraspQualityTermination
from .manager_terms import LeapHandObjectReset, _env_ids

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv

    class _HoraGraspEnv(ManagerBasedRlEnv, Protocol):
        @property
        def reset_time_outs(self) -> np.ndarray: ...

        @property
        def reset_terminated(self) -> np.ndarray: ...

        @property
        def extras(self) -> dict[str, Any]: ...


HORA_GRASP_CACHE_SCHEMA_VERSION = 2
HORA_GRASP_SPLITS = ("train", "validation", "held_out")
_SPLIT_TO_ID = {name: index for index, name in enumerate(HORA_GRASP_SPLITS)}
_REPO_ROOT = Path(__file__).parents[5]


def _name(term: str, field: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{term} {field} must be a non-empty string")
    return value


def _positive_int(term: str, field: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{term} {field} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{term} {field} must be positive")
    return result


def _nonnegative(term: str, field: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{term} {field} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{term} {field} must be finite and nonnegative")
    return result


def _pair(term: str, field: str, value: Any) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise TypeError(f"{term} {field} must be a two-element sequence")
    lower = _nonnegative(term, field, value[0])
    upper = _nonnegative(term, field, value[1])
    if lower > upper:
        raise ValueError(f"{term} {field} lower bound must not exceed upper bound")
    return lower, upper


def _asset_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(ASSETS_ROOT_PATH) / path


@dataclass(frozen=True)
class HoraGraspCache:
    """Validated HORA grasp cache with explicit geometry and family topology."""

    joint_pos: np.ndarray
    object_pos: np.ndarray
    object_quat: np.ndarray
    cylinder_radius: np.ndarray
    cylinder_length: np.ndarray
    family_id: np.ndarray
    split: np.ndarray
    validation_trials: np.ndarray
    mass_min: np.ndarray
    mass_max: np.ndarray
    sliding_friction_min: np.ndarray
    sliding_friction_max: np.ndarray
    family_names: tuple[str, ...]

    def __post_init__(self) -> None:
        count = int(np.asarray(self.joint_pos).shape[0])
        shapes = {
            "joint_pos": (count, 16),
            "object_pos": (count, 3),
            "object_quat": (count, 4),
            "cylinder_radius": (count,),
            "cylinder_length": (count,),
            "family_id": (count,),
            "split": (count,),
            "validation_trials": (count,),
            "mass_min": (count,),
            "mass_max": (count,),
            "sliding_friction_min": (count,),
            "sliding_friction_max": (count,),
        }
        for name, expected in shapes.items():
            values = np.asarray(getattr(self, name))
            if values.shape != expected:
                raise ValueError(
                    f"HORA grasp cache {name} must have shape {expected}, got {values.shape}"
                )
        floating = (
            self.joint_pos,
            self.object_pos,
            self.object_quat,
            self.cylinder_radius,
            self.cylinder_length,
            self.mass_min,
            self.mass_max,
            self.sliding_friction_min,
            self.sliding_friction_max,
        )
        if any(not np.isfinite(np.asarray(values)).all() for values in floating):
            raise ValueError("HORA grasp cache contains NaN or Inf")
        if count and np.any(np.linalg.norm(self.object_quat, axis=1) <= 1.0e-8):
            raise ValueError("HORA grasp cache contains a zero-norm quaternion")
        if np.any(self.cylinder_radius <= 0.0) or np.any(self.cylinder_length <= 0.0):
            raise ValueError("HORA grasp cache cylinder dimensions must be positive")
        if not self.family_names or any(not name for name in self.family_names):
            raise ValueError("HORA grasp cache family_names must be non-empty")
        if np.any(self.family_id < 0) or np.any(self.family_id >= len(self.family_names)):
            raise ValueError("HORA grasp cache family_id is out of range")
        if np.any(self.split < 0) or np.any(self.split >= len(HORA_GRASP_SPLITS)):
            raise ValueError("HORA grasp cache split is out of range")
        if np.any(self.validation_trials < 1):
            raise ValueError("HORA grasp cache validation_trials must be positive")
        if np.any(self.mass_min <= 0.0) or np.any(self.mass_min > self.mass_max):
            raise ValueError("HORA grasp cache mass validation bounds are invalid")
        if np.any(self.sliding_friction_min <= 0.0) or np.any(
            self.sliding_friction_min > self.sliding_friction_max
        ):
            raise ValueError("HORA grasp cache friction validation bounds are invalid")
        for family in np.unique(self.family_id):
            family_splits = np.unique(self.split[self.family_id == family])
            if family_splits.size != 1:
                raise ValueError("each HORA grasp family must belong to exactly one split")

    @property
    def size(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def state_rows(self) -> np.ndarray:
        return np.concatenate((self.joint_pos, self.object_pos, self.object_quat), axis=1)

    def select(self, splits: Sequence[str]) -> HoraGraspCache:
        unknown = set(splits) - set(HORA_GRASP_SPLITS)
        if unknown:
            raise ValueError(f"unknown HORA cache splits: {sorted(unknown)}")
        ids = np.asarray([_SPLIT_TO_ID[name] for name in splits], dtype=np.int8)
        mask = np.isin(self.split, ids)
        if not np.any(mask):
            raise ValueError(f"HORA grasp cache has no rows for splits {list(splits)}")
        kwargs = {
            name: np.asarray(getattr(self, name))[mask]
            for name in (
                "joint_pos",
                "object_pos",
                "object_quat",
                "cylinder_radius",
                "cylinder_length",
                "family_id",
                "split",
                "validation_trials",
                "mass_min",
                "mass_max",
                "sliding_friction_min",
                "sliding_friction_max",
            )
        }
        return HoraGraspCache(**kwargs, family_names=self.family_names)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.asarray(HORA_GRASP_CACHE_SCHEMA_VERSION, dtype=np.int16),
            joint_pos=np.asarray(self.joint_pos, dtype=np.float32),
            object_pos=np.asarray(self.object_pos, dtype=np.float32),
            object_quat=np.asarray(self.object_quat, dtype=np.float32),
            cylinder_radius=np.asarray(self.cylinder_radius, dtype=np.float32),
            cylinder_length=np.asarray(self.cylinder_length, dtype=np.float32),
            family_id=np.asarray(self.family_id, dtype=np.int16),
            split=np.asarray(self.split, dtype=np.int8),
            validation_trials=np.asarray(self.validation_trials, dtype=np.int16),
            mass_min=np.asarray(self.mass_min, dtype=np.float32),
            mass_max=np.asarray(self.mass_max, dtype=np.float32),
            sliding_friction_min=np.asarray(self.sliding_friction_min, dtype=np.float32),
            sliding_friction_max=np.asarray(self.sliding_friction_max, dtype=np.float32),
            family_names=np.asarray(self.family_names, dtype=np.str_),
            split_names=np.asarray(HORA_GRASP_SPLITS, dtype=np.str_),
        )

    @classmethod
    def load(cls, path: Path) -> HoraGraspCache:
        with np.load(path, allow_pickle=False) as payload:
            version = int(np.asarray(payload["schema_version"]).item())
            if version != HORA_GRASP_CACHE_SCHEMA_VERSION:
                raise ValueError(
                    f"HORA grasp cache {path} uses schema {version}; expected "
                    f"{HORA_GRASP_CACHE_SCHEMA_VERSION}"
                )
            split_names = tuple(str(value) for value in payload["split_names"].tolist())
            if split_names != HORA_GRASP_SPLITS:
                raise ValueError(f"HORA grasp cache {path} has incompatible split names")
            kwargs = {
                name: np.array(payload[name], copy=True)
                for name in (
                    "joint_pos",
                    "object_pos",
                    "object_quat",
                    "cylinder_radius",
                    "cylinder_length",
                    "family_id",
                    "split",
                    "validation_trials",
                    "mass_min",
                    "mass_max",
                    "sliding_friction_min",
                    "sliding_friction_max",
                )
            }
            family_names = tuple(str(value) for value in payload["family_names"].tolist())
        return cls(**kwargs, family_names=family_names)


class LeapHoraHandObjectReset(LeapHandObjectReset):
    """Sample a geometry-associated cache row or canonical precision family."""

    _EXTRA_PARAMS = frozenset(
        {
            "cache_splits",
            "object_geom_name",
            "canonical_families",
            "cylinder_geometries",
            "cylinder_geometry_override",
            "candidate_joint_noise",
            "candidate_object_xy_noise",
            "candidate_object_z_noise",
            "candidate_object_tilt_noise_deg",
        }
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        term = type(self).__name__
        original = dict(cfg.params)
        base_cfg = copy.deepcopy(cfg)
        base_cfg.params = {
            key: value for key, value in original.items() if key not in self._EXTRA_PARAMS
        }
        cache_value = base_cfg.params.get("grasp_cache_path")
        base_cfg.params["grasp_cache_path"] = None
        base_cfg.params.pop("grasp_cache_sampling", None)
        base_cfg.params.pop("grasp_cache_start_index", None)
        super().__init__(base_cfg, env)

        self._hora_cache: HoraGraspCache | None = None
        self._eligible_cache_indices = np.empty(0, dtype=np.int64)
        if cache_value is not None:
            path = _asset_path(_name(term, "grasp_cache_path", cache_value))
            if not path.is_file():
                raise FileNotFoundError(
                    f"{term} configured HORA grasp cache does not exist: {path}"
                )
            splits = original.get("cache_splits", ("train",))
            if isinstance(splits, (str, bytes)) or not isinstance(splits, Sequence):
                raise TypeError(f"{term} cache_splits must be a sequence")
            self._hora_cache = HoraGraspCache.load(path)
            wanted = np.asarray([_SPLIT_TO_ID[str(name)] for name in splits], dtype=np.int8)
            self._eligible_cache_indices = np.flatnonzero(
                np.isin(self._hora_cache.split, wanted)
            ).astype(np.int64)
            if not self._eligible_cache_indices.size:
                raise ValueError(f"{term} cache has no rows for splits {list(splits)}")

        self._family_names, self._family_split, self._family_rows = self._parse_families(
            original.get("canonical_families")
        )
        self._geometries = self._parse_geometries(original.get("cylinder_geometries"))
        geometry_override = original.get("cylinder_geometry_override")
        self._geometry_override = (
            None
            if geometry_override is None
            else np.asarray(
                _pair(term, "cylinder_geometry_override", geometry_override),
                dtype=np.float64,
            )
        )
        if self._hora_cache is None and not self._family_names:
            raise ValueError(f"{term} requires a HORA cache or canonical_families")
        if self._hora_cache is None and not self._geometries.size:
            raise ValueError(f"{term} requires cylinder_geometries when generating candidates")

        self._candidate_joint_noise = _nonnegative(
            term, "candidate_joint_noise", original.get("candidate_joint_noise", 0.0)
        )
        self._candidate_xy_noise = _nonnegative(
            term, "candidate_object_xy_noise", original.get("candidate_object_xy_noise", 0.0)
        )
        self._candidate_z_noise = _nonnegative(
            term, "candidate_object_z_noise", original.get("candidate_object_z_noise", 0.0)
        )
        self._candidate_tilt_rad = np.deg2rad(
            _nonnegative(
                term,
                "candidate_object_tilt_noise_deg",
                original.get("candidate_object_tilt_noise_deg", 0.0),
            )
        )
        entity = cast("Entity", self._entity)
        geom_name = _name(term, "object_geom_name", original.get("object_geom_name"))
        geom_ids, names = entity.find_geoms(geom_name, preserve_order=True)
        if len(geom_ids) != 1:
            raise ValueError(f"{term} expected one object geom {geom_name!r}, got {names}")
        self._object_geom_ids, defaults = entity.bind_geom_size_write(
            geom_ids, term_name="hora_cylinder_geometry"
        )
        self._default_geom_size = np.asarray(defaults[0], dtype=np.float64)

        dtype = get_global_dtype()
        self.current_radius = np.zeros(env.num_envs, dtype=dtype)
        self.current_length = np.zeros(env.num_envs, dtype=dtype)
        self.current_family_id = np.full(env.num_envs, -1, dtype=np.int16)
        self.current_split = np.full(env.num_envs, -1, dtype=np.int8)
        self.candidate_id = np.full(env.num_envs, -1, dtype=np.int64)
        self.validation_passes = np.zeros(env.num_envs, dtype=np.int16)
        self._repeat_candidate = np.zeros(env.num_envs, dtype=np.bool_)
        self._candidate_rows = np.zeros((env.num_envs, 23), dtype=np.float64)
        self._candidate_rows[:, 19] = 1.0
        self._next_candidate_id = 0
        self._cache_cursor = int(original.get("grasp_cache_start_index", 0))
        self._requested_sampling = str(original.get("grasp_cache_sampling", "random"))
        if self._requested_sampling not in {"random", "sequential"}:
            raise ValueError(f"{term} grasp_cache_sampling must be 'random' or 'sequential'")

    @staticmethod
    def _parse_families(value: Any) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
        if value is None:
            return (), np.empty(0, dtype=np.int8), np.empty((0, 23), dtype=np.float64)
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("LeapHoraHandObjectReset canonical_families must be a sequence")
        names: list[str] = []
        splits: list[int] = []
        rows: list[np.ndarray] = []
        for index, family in enumerate(value):
            if not isinstance(family, Mapping):
                raise TypeError("every canonical HORA grasp family must be a mapping")
            name = _name(
                "LeapHoraHandObjectReset", f"canonical_families[{index}].name", family.get("name")
            )
            split_name = str(family.get("split"))
            if split_name not in _SPLIT_TO_ID:
                raise ValueError(f"canonical family {name!r} has invalid split {split_name!r}")
            joint = np.asarray(family.get("joint_pos"), dtype=np.float64)
            pos = np.asarray(family.get("object_pos"), dtype=np.float64)
            quat = np.asarray(family.get("object_quat"), dtype=np.float64)
            if joint.shape != (16,) or pos.shape != (3,) or quat.shape != (4,):
                raise ValueError(f"canonical family {name!r} has invalid pose dimensions")
            if not np.isfinite(np.concatenate((joint, pos, quat))).all():
                raise ValueError(f"canonical family {name!r} contains NaN or Inf")
            norm = np.linalg.norm(quat)
            if norm <= 1.0e-8:
                raise ValueError(f"canonical family {name!r} has a zero-norm quaternion")
            names.append(name)
            splits.append(_SPLIT_TO_ID[split_name])
            rows.append(np.concatenate((joint, pos, quat / norm)))
        if len(set(names)) != len(names):
            raise ValueError("canonical HORA grasp family names must be unique")
        return tuple(names), np.asarray(splits, dtype=np.int8), np.asarray(rows, dtype=np.float64)

    @staticmethod
    def _parse_geometries(value: Any) -> np.ndarray:
        if value is None:
            return np.empty((0, 2), dtype=np.float64)
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("LeapHoraHandObjectReset cylinder_geometries must be a sequence")
        geometries = np.asarray(value, dtype=np.float64)
        if geometries.ndim != 2 or geometries.shape[1] != 2:
            raise ValueError("cylinder_geometries must have shape (N, 2) as [radius, length]")
        if not np.isfinite(geometries).all() or np.any(geometries <= 0.0):
            raise ValueError("cylinder_geometries must be finite and positive")
        return geometries

    @property
    def family_names(self) -> tuple[str, ...]:
        return self._hora_cache.family_names if self._hora_cache is not None else self._family_names

    def _sample_cache_indices(self, env: ManagerBasedRlEnv, count: int) -> np.ndarray:
        if self._requested_sampling == "random":
            positions = env.rng.integers(0, self._eligible_cache_indices.size, size=count)
        else:
            positions = (np.arange(count) + self._cache_cursor) % self._eligible_cache_indices.size
            self._cache_cursor = int(
                (self._cache_cursor + count) % self._eligible_cache_indices.size
            )
        return self._eligible_cache_indices[positions]

    def _new_candidates(self, env: ManagerBasedRlEnv, ids: np.ndarray) -> None:
        count = ids.size
        self.validation_passes[ids] = 0
        self.candidate_id[ids] = np.arange(
            self._next_candidate_id, self._next_candidate_id + count, dtype=np.int64
        )
        self._next_candidate_id += count
        if self._hora_cache is not None:
            indices = self._sample_cache_indices(env, count)
            cache = self._hora_cache
            self._candidate_rows[ids] = cache.state_rows[indices]
            self.current_radius[ids] = cache.cylinder_radius[indices]
            self.current_length[ids] = cache.cylinder_length[indices]
            self.current_family_id[ids] = cache.family_id[indices]
            self.current_split[ids] = cache.split[indices]
            if self._geometry_override is not None:
                self.current_radius[ids] = self._geometry_override[0]
                self.current_length[ids] = self._geometry_override[1]
            self.grasp_cache_index[ids] = indices
            return

        family = env.rng.integers(0, len(self._family_names), size=count)
        geometry = env.rng.integers(0, self._geometries.shape[0], size=count)
        rows = self._family_rows[family].copy()
        if self._candidate_joint_noise:
            rows[:, :16] += env.rng.uniform(
                -self._candidate_joint_noise, self._candidate_joint_noise, size=(count, 16)
            )
        if self._candidate_xy_noise:
            rows[:, 16:18] += env.rng.uniform(
                -self._candidate_xy_noise, self._candidate_xy_noise, size=(count, 2)
            )
        if self._candidate_z_noise:
            rows[:, 18] += env.rng.uniform(
                -self._candidate_z_noise, self._candidate_z_noise, size=count
            )
        if self._candidate_tilt_rad:
            angles = self._candidate_tilt_rad * np.sqrt(env.rng.random(count))
            azimuth = env.rng.uniform(-np.pi, np.pi, size=count)
            axes = np.stack((np.cos(azimuth), np.sin(azimuth), np.zeros(count)), axis=1)
            rows[:, 19:23] = np_quat_mul(rows[:, 19:23], np_quat_from_angle_axis(angles, axes))
        np.clip(rows[:, :16], self._ctrl_lower, self._ctrl_upper, out=rows[:, :16])
        self._candidate_rows[ids] = rows
        self.current_radius[ids] = self._geometries[geometry, 0]
        self.current_length[ids] = self._geometries[geometry, 1]
        self.current_family_id[ids] = family
        self.current_split[ids] = self._family_split[family]
        if self._geometry_override is not None:
            self.current_radius[ids] = self._geometry_override[0]
            self.current_length[ids] = self._geometry_override[1]
        self.grasp_cache_index[ids] = -1

    def record_validation_results(
        self,
        env_ids: np.ndarray,
        passed: np.ndarray,
        settled_rows: np.ndarray,
        required_trials: int,
    ) -> np.ndarray:
        ids = np.asarray(env_ids, dtype=np.intp)
        result = np.asarray(passed, dtype=np.bool_)
        if result.shape != (ids.size,) or settled_rows.shape != (ids.size, 23):
            raise ValueError("HORA validation result dimensions do not match env_ids")
        first_success = result & (self.validation_passes[ids] == 0)
        if np.any(first_success):
            self._candidate_rows[ids[first_success]] = settled_rows[first_success]
        self.validation_passes[ids] = np.where(result, self.validation_passes[ids] + 1, 0)
        accepted = result & (self.validation_passes[ids] >= required_trials)
        self._repeat_candidate[ids] = result & ~accepted
        return accepted

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        **params: Any,
    ) -> None:
        ids = _env_ids(env, env_ids)
        new_ids = ids[~self._repeat_candidate[ids]]
        if new_ids.size:
            self._new_candidates(env, new_ids)
        self._repeat_candidate[ids] = False

        selected_rows = self._candidate_rows[ids]
        saved_cache = self._grasp_cache
        saved_sampling = self._grasp_cache_sampling
        saved_cursor = self._grasp_cache_cursor
        self._grasp_cache = selected_rows
        self._grasp_cache_sampling = "sequential"
        self._grasp_cache_cursor = 0
        base_params = {key: value for key, value in params.items() if key not in self._EXTRA_PARAMS}
        base_params["grasp_cache_path"] = None
        base_params.pop("grasp_cache_sampling", None)
        base_params.pop("grasp_cache_start_index", None)
        try:
            super().__call__(env, ids, **base_params)
        finally:
            self._grasp_cache = saved_cache
            self._grasp_cache_sampling = saved_sampling
            self._grasp_cache_cursor = saved_cursor

        sizes = np.broadcast_to(self._default_geom_size, (ids.size, 1, 3)).copy()
        sizes[:, 0, 0] = self.current_radius[ids]
        sizes[:, 0, 1] = self.current_length[ids] / 2.0
        cast("Entity", self._entity).write_geom_size_to_sim(
            sizes,
            geom_ids=self._object_geom_ids,
            env_ids=ids,
            term_name="hora_cylinder_geometry",
        )


class LeapHoraCylinderMassInertia(ManagerTermBase):
    """Sample mass and exact solid-cylinder inertia for the active geometry."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term = type(self).__name__
        allowed = {"asset_cfg", "geometry_event_name", "density_range"}
        unexpected = set(cfg.params) - allowed
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        asset_cfg = cfg.params.get("asset_cfg")
        if asset_cfg is None or not hasattr(asset_cfg, "name"):
            raise TypeError(f"{term} asset_cfg must be a SceneEntityCfg")
        self._entity = cast("Entity", env.scene[asset_cfg.name])
        self._body_ids, _ = self._entity.bind_body_mass_write(
            asset_cfg.body_ids, term_name="hora_cylinder_mass_inertia"
        )
        self._geometry_event_name = _name(
            term, "geometry_event_name", cfg.params.get("geometry_event_name")
        )
        self._density_range = _pair(term, "density_range", cfg.params.get("density_range"))

        try:
            import mujoco  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise NotImplementedError(
                f"{term} requires mujoco for cold-path inertia metadata"
            ) from exc
        scene = getattr(getattr(env, "cfg", None), "scene", None)
        model_file = Path(str(getattr(scene, "model_file", "")))
        candidates = (model_file, _REPO_ROOT / model_file)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise ValueError(f"{term} cannot locate scene model file {model_file}")
        model = mujoco.MjModel.from_xml_path(str(path))
        inertia_ids, _ = self._entity.bind_body_inertia_write(
            asset_cfg.body_ids,
            default=np.asarray(model.body_inertia, dtype=np.float64),
            default_mass=np.asarray(model.body_mass, dtype=np.float64),
            term_name="hora_cylinder_mass_inertia",
        )
        if not np.array_equal(self._body_ids, inertia_ids) or self._body_ids.size != 1:
            raise ValueError(f"{term} requires exactly one matching cylinder body")
        self.mass = np.full(env.num_envs, np.nan, dtype=get_global_dtype())
        self.density = np.full(env.num_envs, np.nan, dtype=get_global_dtype())

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        **params: Any,
    ) -> None:
        del params
        ids = _env_ids(env, env_ids)
        geometry = env.event_manager.get_term_cfg(self._geometry_event_name).func
        if not isinstance(geometry, LeapHoraHandObjectReset):
            raise TypeError(f"{type(self).__name__} geometry event must be LeapHoraHandObjectReset")
        density = env.rng.uniform(*self._density_range, size=ids.size)
        radius = np.asarray(geometry.current_radius[ids], dtype=np.float64)
        length = np.asarray(geometry.current_length[ids], dtype=np.float64)
        mass = density * np.pi * np.square(radius) * length
        transverse = mass * (3.0 * np.square(radius) + np.square(length)) / 12.0
        axial = 0.5 * mass * np.square(radius)
        inertia = np.stack((transverse, transverse, axial), axis=1)[:, None, :]
        self._entity.write_body_mass_to_sim(
            mass[:, None], self._body_ids, ids, term_name="hora_cylinder_mass_inertia"
        )
        self._entity.write_body_inertia_to_sim(
            inertia, self._body_ids, ids, term_name="hora_cylinder_mass_inertia"
        )
        self.mass[ids] = mass
        self.density[ids] = density


class LeapHoraGraspRecorder(RecorderTerm):
    """Require repeated stable trials, filter diversity, and write cache schema v2."""

    _ALLOWED_PARAMS = frozenset(
        {
            "reset_event_name",
            "quality_term_name",
            "mass_event_name",
            "friction_event_name",
            "output_path",
            "collection_target",
            "validation_trials",
            "auto_save",
            "joint_limit_margin_fraction",
            "minimum_joint_rms_rad",
            "minimum_position_distance_m",
            "minimum_axis_angle_deg",
            "geometry_tolerance_m",
        }
    )

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        term = type(self).__name__
        unexpected = set(cfg.params) - self._ALLOWED_PARAMS
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        reset_name = _name(term, "reset_event_name", cfg.params.get("reset_event_name"))
        reset = env.event_manager.get_term_cfg(reset_name).func
        if not isinstance(reset, LeapHoraHandObjectReset):
            raise TypeError(f"{term} reset event must be LeapHoraHandObjectReset")
        self._reset = reset
        quality_name = _name(term, "quality_term_name", cfg.params.get("quality_term_name"))
        quality = env.termination_manager.get_term_cfg(quality_name).func
        if not isinstance(quality, LeapGraspQualityTermination):
            raise TypeError(f"{term} quality term must be LeapGraspQualityTermination")
        self._quality = quality
        self._mass_event_name = _name(term, "mass_event_name", cfg.params.get("mass_event_name"))
        self._friction_event_name = _name(
            term, "friction_event_name", cfg.params.get("friction_event_name")
        )
        self._output_path = _asset_path(_name(term, "output_path", cfg.params.get("output_path")))
        if self._output_path.suffix != ".npz":
            raise ValueError(f"{term} output_path must end in .npz")
        self._target = _positive_int(term, "collection_target", cfg.params.get("collection_target"))
        self._validation_trials = _positive_int(
            term, "validation_trials", cfg.params.get("validation_trials")
        )
        self._auto_save = bool(cfg.params.get("auto_save"))
        if not isinstance(cfg.params.get("auto_save"), (bool, np.bool_)):
            raise TypeError(f"{term} auto_save must be boolean")
        self._joint_limit_margin = _nonnegative(
            term,
            "joint_limit_margin_fraction",
            cfg.params.get("joint_limit_margin_fraction"),
        )
        if self._joint_limit_margin > 0.5:
            raise ValueError(f"{term} joint_limit_margin_fraction must be at most 0.5")
        self._joint_rms = _nonnegative(
            term, "minimum_joint_rms_rad", cfg.params.get("minimum_joint_rms_rad")
        )
        self._position_distance = _nonnegative(
            term,
            "minimum_position_distance_m",
            cfg.params.get("minimum_position_distance_m"),
        )
        self._axis_angle = np.deg2rad(
            _nonnegative(term, "minimum_axis_angle_deg", cfg.params.get("minimum_axis_angle_deg"))
        )
        self._geometry_tolerance = _nonnegative(
            term, "geometry_tolerance_m", cfg.params.get("geometry_tolerance_m")
        )
        self._rows: list[np.ndarray] = []
        self._radius: list[float] = []
        self._length: list[float] = []
        self._family: list[int] = []
        self._split: list[int] = []
        self._trials: list[int] = []
        self._mass_min: list[float] = []
        self._mass_max: list[float] = []
        self._friction_min: list[float] = []
        self._friction_max: list[float] = []
        self._trial_mass_min = np.full(env.num_envs, np.inf)
        self._trial_mass_max = np.full(env.num_envs, -np.inf)
        self._trial_friction_min = np.full(env.num_envs, np.inf)
        self._trial_friction_max = np.full(env.num_envs, -np.inf)
        self._rejected_validation = 0
        self._rejected_duplicates = 0
        self._saved = False

    @staticmethod
    def _axes(rows: np.ndarray) -> np.ndarray:
        quat = rows[:, 19:23] / np.linalg.norm(rows[:, 19:23], axis=1, keepdims=True)
        w, x, y, z = quat.T
        return np.stack(
            (2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)),
            axis=1,
        )

    def _is_novel(self, row: np.ndarray, radius: float, length: float) -> bool:
        if not self._rows:
            return True
        rows = np.asarray(self._rows)
        same_geometry = (np.abs(np.asarray(self._radius) - radius) <= self._geometry_tolerance) & (
            np.abs(np.asarray(self._length) - length) <= self._geometry_tolerance
        )
        near = same_geometry.copy()
        if self._joint_rms:
            near &= np.sqrt(np.mean(np.square(rows[:, :16] - row[:16]), axis=1)) < self._joint_rms
        if self._position_distance:
            near &= np.linalg.norm(rows[:, 16:19] - row[16:19], axis=1) < self._position_distance
        if self._axis_angle:
            alignment = np.abs(self._axes(rows) @ self._axes(row[None, :])[0])
            near &= np.arccos(np.clip(alignment, 0.0, 1.0)) < self._axis_angle
        return not bool(np.any(near))

    def _physics_values(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        env = cast("_HoraGraspEnv", self._env)
        mass_term = env.event_manager.get_term_cfg(self._mass_event_name).func
        mass = np.asarray(getattr(mass_term, "mass", None))
        friction_term = env.event_manager.get_term_cfg(self._friction_event_name).func
        friction_values = np.asarray(getattr(friction_term, "last_values", None))
        if mass.shape != (env.num_envs,) or friction_values.shape[0] != env.num_envs:
            raise RuntimeError("HORA grasp validation physics events did not expose sampled values")
        return mass[ids], friction_values[ids, 0, 0]

    def _cache(self) -> HoraGraspCache:
        rows = np.asarray(self._rows, dtype=np.float32).reshape(-1, 23)
        return HoraGraspCache(
            joint_pos=rows[:, :16],
            object_pos=rows[:, 16:19],
            object_quat=rows[:, 19:23],
            cylinder_radius=np.asarray(self._radius, dtype=np.float32),
            cylinder_length=np.asarray(self._length, dtype=np.float32),
            family_id=np.asarray(self._family, dtype=np.int16),
            split=np.asarray(self._split, dtype=np.int8),
            validation_trials=np.asarray(self._trials, dtype=np.int16),
            mass_min=np.asarray(self._mass_min, dtype=np.float32),
            mass_max=np.asarray(self._mass_max, dtype=np.float32),
            sliding_friction_min=np.asarray(self._friction_min, dtype=np.float32),
            sliding_friction_max=np.asarray(self._friction_max, dtype=np.float32),
            family_names=self._reset.family_names,
        )

    def _save(self, *, force: bool = False) -> None:
        if self._saved or not self._rows or (not force and len(self._rows) < self._target):
            return
        self._cache().save(self._output_path)
        self._saved = True

    def _accepted_distribution(self) -> dict[str, dict[str, int]]:
        geometry_counts: dict[str, int] = {}
        for radius, length in zip(self._radius, self._length, strict=True):
            key = f"r={radius:.6g},l={length:.6g}"
            geometry_counts[key] = geometry_counts.get(key, 0) + 1
        family_counts = {name: 0 for name in self._reset.family_names}
        for family_id in self._family:
            name = self._reset.family_names[family_id]
            family_counts[name] += 1
        split_counts = {name: 0 for name in HORA_GRASP_SPLITS}
        for split_id in self._split:
            split_counts[HORA_GRASP_SPLITS[split_id]] += 1
        return {
            "geometry_counts": geometry_counts,
            "family_counts": family_counts,
            "split_counts": split_counts,
        }

    def record_pre_reset(self, env_ids: np.ndarray) -> None:
        env = cast("_HoraGraspEnv", self._env)
        ids = np.asarray(env_ids, dtype=np.intp)
        mass, friction = self._physics_values(ids)
        self._trial_mass_min[ids] = np.minimum(self._trial_mass_min[ids], mass)
        self._trial_mass_max[ids] = np.maximum(self._trial_mass_max[ids], mass)
        self._trial_friction_min[ids] = np.minimum(self._trial_friction_min[ids], friction)
        self._trial_friction_max[ids] = np.maximum(self._trial_friction_max[ids], friction)

        observation = self._quality.grasp.observation
        widths = self._reset._ctrl_upper - self._reset._ctrl_lower
        lower = self._reset._ctrl_lower + self._joint_limit_margin * widths
        upper = self._reset._ctrl_upper - self._joint_limit_margin * widths
        away_from_limits = np.all(
            (observation.dof_pos[ids] > lower) & (observation.dof_pos[ids] < upper), axis=1
        )
        passed = (
            env.reset_time_outs[ids]
            & ~env.reset_terminated[ids]
            & self._quality.stable[ids]
            & ~self._quality.grasp.palm_supported[ids]
            & away_from_limits
        )
        settled = np.concatenate(
            (
                observation.dof_pos[ids],
                observation.object_pos[ids],
                observation.object_quat[ids],
            ),
            axis=1,
        )
        accepted = self._reset.record_validation_results(
            ids, passed, settled, self._validation_trials
        )
        for local, env_id in enumerate(ids):
            if not accepted[local]:
                if not passed[local]:
                    self._rejected_validation += 1
                    self._trial_mass_min[env_id] = np.inf
                    self._trial_mass_max[env_id] = -np.inf
                    self._trial_friction_min[env_id] = np.inf
                    self._trial_friction_max[env_id] = -np.inf
                continue
            row = settled[local].astype(np.float32)
            radius = float(self._reset.current_radius[env_id])
            length = float(self._reset.current_length[env_id])
            if not self._is_novel(row, radius, length):
                self._rejected_duplicates += 1
            elif len(self._rows) < self._target:
                self._rows.append(row)
                self._radius.append(radius)
                self._length.append(length)
                self._family.append(int(self._reset.current_family_id[env_id]))
                self._split.append(int(self._reset.current_split[env_id]))
                self._trials.append(self._validation_trials)
                self._mass_min.append(float(self._trial_mass_min[env_id]))
                self._mass_max.append(float(self._trial_mass_max[env_id]))
                self._friction_min.append(float(self._trial_friction_min[env_id]))
                self._friction_max.append(float(self._trial_friction_max[env_id]))
            self._trial_mass_min[env_id] = np.inf
            self._trial_mass_max[env_id] = -np.inf
            self._trial_friction_min[env_id] = np.inf
            self._trial_friction_max[env_id] = -np.inf

        env.extras.setdefault("log", {}).update(
            {
                "hora_grasp/cache_size": float(len(self._rows)),
                "hora_grasp/rejected_validation": float(self._rejected_validation),
                "hora_grasp/rejected_near_duplicates": float(self._rejected_duplicates),
            }
        )
        if len(self._rows) >= self._target:
            self._save()
            raise RunComplete(
                reason="hora_grasp_collection_target_reached",
                summary={
                    "saved_grasps": len(self._rows),
                    "validation_trials": self._validation_trials,
                    "rejected_validation": self._rejected_validation,
                    "rejected_near_duplicates": self._rejected_duplicates,
                    "output_path": str(self._output_path),
                    **self._accepted_distribution(),
                },
            )

    def close(self) -> None:
        self._save(force=self._auto_save)


registry.register_env_config("LeapInhandHoraGrasp", ManagerBasedRlEnvCfg)
registry.register_env("LeapInhandHoraGrasp", make_manager_based_rl_env, sim_backend="mujoco")


__all__ = [
    "HORA_GRASP_CACHE_SCHEMA_VERSION",
    "HORA_GRASP_SPLITS",
    "HoraGraspCache",
    "LeapHoraCylinderMassInertia",
    "LeapHoraGraspRecorder",
    "LeapHoraHandObjectReset",
]
