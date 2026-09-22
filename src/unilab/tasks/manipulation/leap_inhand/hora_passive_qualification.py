"""Long-horizon passive qualification for schema-v2 HORA grasp caches."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.run_control import RunComplete
from unilab.managers import RecorderTerm, RecorderTermCfg

from .hora_grasp import HoraGraspCache, LeapHoraHandObjectReset
from .manager_terms import LeapGraspTermination

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


_TIER_NAMES = ("invalid", "marginal", "recoverable", "anchor")


def _asset_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(ASSETS_ROOT_PATH) / path


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


def _nonnegative_int(term: str, field: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{term} {field} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{term} {field} must be nonnegative")
    return result


def _fraction(term: str, field: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{term} {field} must be a real number")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{term} {field} must be finite and in [0, 1]")
    return result


def _threshold(term: str, field: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{term} {field} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{term} {field} must be finite and nonnegative")
    return result


def select_contact_balanced_indices(
    *,
    eligible: np.ndarray,
    tier: np.ndarray,
    family_id: np.ndarray,
    raw_thumb_fraction: np.ndarray,
    raw_opposing_fraction: np.ndarray,
    cache_size: int,
    anchor_fraction: float,
    middle_contact_fraction: float,
    minimum_raw_thumb_contact_fraction: float,
    minimum_raw_middle_contact_fraction: float,
    minimum_raw_opposing_contact_fraction: float,
    minimum_raw_opposing_fingers: int,
    seed: int,
    family_targets: Mapping[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a family-balanced cache with an explicit middle-contact quota.

    Sampling with replacement is intentional: a small qualified pool can be
    used for a bounded pilot without pretending it contains more unique grasps
    than it does. The score report preserves source indices and uniqueness.
    """
    eligible = np.asarray(eligible, dtype=np.int64)
    if not eligible.size:
        raise ValueError("contact-balanced selection requires eligible rows")
    if raw_opposing_fraction.shape != (tier.size, 3):
        raise ValueError("raw_opposing_fraction must have shape (cache_size, 3)")
    if minimum_raw_opposing_fingers < 0 or minimum_raw_opposing_fingers > 3:
        raise ValueError("minimum_raw_opposing_fingers must be in [0, 3]")

    stable = np.isin(tier, (2, 3))
    support_count = np.count_nonzero(
        raw_opposing_fraction >= minimum_raw_opposing_contact_fraction,
        axis=1,
    )
    base_mask = (
        stable
        & (raw_thumb_fraction >= minimum_raw_thumb_contact_fraction)
        & (support_count >= minimum_raw_opposing_fingers)
    )
    eligible_mask = np.zeros(tier.size, dtype=np.bool_)
    eligible_mask[eligible] = True
    base_mask &= eligible_mask
    middle_mask = base_mask & (raw_opposing_fraction[:, 1] >= minimum_raw_middle_contact_fraction)

    families = np.unique(family_id[eligible])
    if not families.size:
        raise ValueError("contact-balanced selection found no eligible families")
    eligible_families = {int(family) for family in families}
    if family_targets is None:
        base, remainder = divmod(cache_size, int(families.size))
        resolved_family_targets = {
            int(family): base + int(position < remainder)
            for position, family in enumerate(families)
        }
    else:
        resolved_family_targets = {
            int(family): int(count) for family, count in family_targets.items()
        }
        if set(resolved_family_targets) != eligible_families:
            raise ValueError("family_targets must contain every eligible family exactly once")
        if any(count <= 0 for count in resolved_family_targets.values()):
            raise ValueError("family_targets counts must be positive")
        if sum(resolved_family_targets.values()) != cache_size:
            raise ValueError("family_targets counts must sum to cache_size")
    desired_middle = int(round(cache_size * middle_contact_fraction))
    tier_targets: dict[tuple[int, int], int] = {}
    middle_slot_capacity: dict[tuple[int, int], int] = {}
    middle_unique_capacity: dict[tuple[int, int], int] = {}
    for family in families:
        family = int(family)
        anchor_target = int(round(resolved_family_targets[family] * anchor_fraction))
        tier_targets[family, 3] = anchor_target
        tier_targets[family, 2] = resolved_family_targets[family] - anchor_target
        for tier_id in (3, 2):
            pool_size = int(
                np.count_nonzero(middle_mask & (family_id == family) & (tier == tier_id))
            )
            slot_capacity = tier_targets[family, tier_id] if pool_size else 0
            middle_slot_capacity[family, tier_id] = slot_capacity
            middle_unique_capacity[family, tier_id] = min(slot_capacity, pool_size)
    capacity = sum(middle_slot_capacity.values())
    if desired_middle > capacity:
        raise RuntimeError(
            "middle-contact quota exceeds the total family-balanced capacity of families "
            f"with qualifying rows: requested={desired_middle}, capacity={capacity}"
        )
    if desired_middle and not capacity:
        raise RuntimeError("no qualified raw middle-contact grasps are available")

    # Consume unique capacity first, preferring cells with the largest diverse
    # pool. Only then allocate replacement-backed slots. Total family and tier
    # quotas remain fixed, but the selected cache uses the expanded dataset
    # instead of repeating a handful of legacy middle-contact rows.
    middle_targets = {cell: 0 for cell in middle_slot_capacity}
    remaining = desired_middle
    for capacities in (middle_unique_capacity, middle_slot_capacity):
        for cell in sorted(capacities, key=lambda item: (-capacities[item], item)):
            available = capacities[cell] - middle_targets[cell]
            allocated = min(remaining, max(available, 0))
            middle_targets[cell] += allocated
            remaining -= allocated
            if remaining == 0:
                break
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError("unable to allocate the requested middle-contact quota")

    rng = np.random.default_rng(seed)
    selected_parts: list[np.ndarray] = []
    enriched_parts: list[np.ndarray] = []

    used = np.zeros(tier.size, dtype=np.bool_)

    def sample(mask: np.ndarray, count: int, description: str) -> np.ndarray:
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        pool = np.flatnonzero(mask)
        if not pool.size:
            raise RuntimeError(f"no qualified grasps available for {description}")
        fresh = pool[~used[pool]]
        fresh_count = min(count, fresh.size)
        parts: list[np.ndarray] = []
        if fresh_count:
            chosen = rng.choice(fresh, size=fresh_count, replace=False).astype(np.int64)
            used[chosen] = True
            parts.append(chosen)
        remaining_count = count - fresh_count
        if remaining_count:
            parts.append(rng.choice(pool, size=remaining_count, replace=True).astype(np.int64))
        return np.concatenate(parts)

    for family in families:
        family = int(family)
        anchor_target = tier_targets[family, 3]
        recoverable_target = tier_targets[family, 2]
        middle_anchor_count = middle_targets[family, 3]
        middle_recoverable_count = middle_targets[family, 2]
        family_mask = family_id == family
        middle_anchor = sample(
            family_mask & middle_mask & (tier == 3),
            middle_anchor_count,
            f"family={family} middle-contact anchors",
        )
        middle_recoverable = sample(
            family_mask & middle_mask & (tier == 2),
            middle_recoverable_count,
            f"family={family} middle-contact recoverable grasps",
        )
        anchor_fill = sample(
            family_mask & base_mask & (tier == 3),
            anchor_target - middle_anchor_count,
            f"family={family} anchors",
        )
        recoverable_fill = sample(
            family_mask & base_mask & (tier == 2),
            recoverable_target - middle_recoverable_count,
            f"family={family} recoverable grasps",
        )
        selected_parts.extend((middle_anchor, middle_recoverable, anchor_fill, recoverable_fill))
        enriched_parts.extend(
            (
                np.ones(middle_anchor.size, dtype=np.bool_),
                np.ones(middle_recoverable.size, dtype=np.bool_),
                np.zeros(anchor_fill.size, dtype=np.bool_),
                np.zeros(recoverable_fill.size, dtype=np.bool_),
            )
        )

    selected = np.concatenate(selected_parts)
    enriched = np.concatenate(enriched_parts)
    if selected.size != cache_size or np.count_nonzero(enriched) != desired_middle:
        raise RuntimeError("contact-balanced selection produced inconsistent quotas")
    order = rng.permutation(selected.size)
    return selected[order], enriched[order]


class LeapHoraPassiveQualificationRecorder(RecorderTerm):
    """Evaluate each eligible cache row once under a fixed zero command.

    Rows are classified by survival time, then written as schema-v2 tier caches.
    A separate Stage A cache is sampled with equal family representation and a
    configured anchor/recoverable mixture.
    """

    _ALLOWED_PARAMS = frozenset(
        {
            "reset_event_name",
            "grasp_term_name",
            "input_cache_path",
            "output_prefix",
            "anchor_minimum_seconds",
            "recoverable_minimum_seconds",
            "marginal_minimum_seconds",
            "training_cache_size",
            "training_anchor_fraction",
            "training_middle_contact_fraction",
            "minimum_raw_thumb_contact_fraction",
            "minimum_raw_middle_contact_fraction",
            "minimum_raw_opposing_contact_fraction",
            "minimum_raw_opposing_fingers",
            "sampling_seed",
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
        if not isinstance(reset, LeapHoraHandObjectReset) or reset._hora_cache is None:
            raise TypeError(f"{term} requires a cached LeapHoraHandObjectReset")
        if reset._requested_sampling != "sequential":
            raise ValueError(f"{term} requires sequential grasp-cache sampling")
        self._reset = reset

        grasp_name = _name(term, "grasp_term_name", cfg.params.get("grasp_term_name"))
        grasp = env.termination_manager.get_term_cfg(grasp_name).func
        if not isinstance(grasp, LeapGraspTermination):
            raise TypeError(f"{term} grasp term {grasp_name!r} must be LeapGraspTermination")
        self._grasp = grasp

        self._input_path = _asset_path(
            _name(term, "input_cache_path", cfg.params.get("input_cache_path"))
        )
        self._cache = HoraGraspCache.load(self._input_path)
        reset_cache = reset._hora_cache
        if reset_cache.size != self._cache.size or not np.array_equal(
            reset_cache.state_rows, self._cache.state_rows
        ):
            raise ValueError(f"{term} input cache must match the reset cache exactly")
        self._output_prefix = _asset_path(
            _name(term, "output_prefix", cfg.params.get("output_prefix"))
        )

        self._anchor_seconds = _threshold(
            term, "anchor_minimum_seconds", cfg.params.get("anchor_minimum_seconds")
        )
        self._recoverable_seconds = _threshold(
            term,
            "recoverable_minimum_seconds",
            cfg.params.get("recoverable_minimum_seconds"),
        )
        self._marginal_seconds = _threshold(
            term, "marginal_minimum_seconds", cfg.params.get("marginal_minimum_seconds")
        )
        if not self._marginal_seconds < self._recoverable_seconds < self._anchor_seconds:
            raise ValueError(f"{term} thresholds must satisfy marginal < recoverable < anchor")
        self._training_cache_size = _positive_int(
            term, "training_cache_size", cfg.params.get("training_cache_size")
        )
        self._training_anchor_fraction = _fraction(
            term,
            "training_anchor_fraction",
            cfg.params.get("training_anchor_fraction"),
        )
        self._training_middle_contact_fraction = _fraction(
            term,
            "training_middle_contact_fraction",
            cfg.params.get("training_middle_contact_fraction", 0.0),
        )
        self._minimum_raw_thumb_contact_fraction = _fraction(
            term,
            "minimum_raw_thumb_contact_fraction",
            cfg.params.get("minimum_raw_thumb_contact_fraction", 0.0),
        )
        self._minimum_raw_middle_contact_fraction = _fraction(
            term,
            "minimum_raw_middle_contact_fraction",
            cfg.params.get("minimum_raw_middle_contact_fraction", 0.0),
        )
        self._minimum_raw_opposing_contact_fraction = _fraction(
            term,
            "minimum_raw_opposing_contact_fraction",
            cfg.params.get("minimum_raw_opposing_contact_fraction", 0.0),
        )
        self._minimum_raw_opposing_fingers = _nonnegative_int(
            term,
            "minimum_raw_opposing_fingers",
            cfg.params.get("minimum_raw_opposing_fingers", 0),
        )
        if self._minimum_raw_opposing_fingers > 3:
            raise ValueError(f"{term} minimum_raw_opposing_fingers must be at most 3")
        seed = cfg.params.get("sampling_seed")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
            raise TypeError(f"{term} sampling_seed must be an integer")
        self._sampling_seed = int(seed)

        geometries = np.unique(np.round(reset._assigned_geometries, decimals=8), axis=0)
        cache_geometry = np.stack(
            (self._cache.cylinder_radius, self._cache.cylinder_length), axis=1
        )
        geometry_mask = np.zeros(self._cache.size, dtype=np.bool_)
        for radius, length in geometries:
            geometry_mask |= np.isclose(cache_geometry[:, 0], radius) & np.isclose(
                cache_geometry[:, 1], length
            )
        eligible_mask = np.zeros(self._cache.size, dtype=np.bool_)
        eligible_mask[reset._eligible_cache_indices] = True
        self._eligible = np.flatnonzero(eligible_mask & geometry_mask).astype(np.int64)
        if not self._eligible.size:
            raise ValueError(f"{term} found no cache rows for the configured split and geometry")
        self._eligible_mask = np.zeros(self._cache.size, dtype=np.bool_)
        self._eligible_mask[self._eligible] = True

        cache_size = self._cache.size
        num_envs = env.num_envs
        self._completed = np.zeros(cache_size, dtype=np.bool_)
        self._survival_seconds = np.full(cache_size, np.nan, dtype=np.float64)
        self._valid_fraction = np.full(cache_size, np.nan, dtype=np.float64)
        self._thumb_fraction = np.full(cache_size, np.nan, dtype=np.float64)
        self._opposing_fraction = np.full((cache_size, 3), np.nan, dtype=np.float64)
        self._raw_thumb_fraction = np.full(cache_size, np.nan, dtype=np.float64)
        self._raw_opposing_fraction = np.full((cache_size, 3), np.nan, dtype=np.float64)
        self._palm_fraction = np.full(cache_size, np.nan, dtype=np.float64)
        self._termination_causes = np.zeros((cache_size, 3), dtype=np.bool_)
        self._active_index = np.full(num_envs, -1, dtype=np.int64)
        self._steps = np.zeros(num_envs, dtype=np.int32)
        self._valid_steps = np.zeros(num_envs, dtype=np.int32)
        self._thumb_steps = np.zeros(num_envs, dtype=np.int32)
        self._opposing_steps = np.zeros((num_envs, 3), dtype=np.int32)
        self._raw_thumb_steps = np.zeros(num_envs, dtype=np.int32)
        self._raw_opposing_steps = np.zeros((num_envs, 3), dtype=np.int32)
        self._palm_steps = np.zeros(num_envs, dtype=np.int32)
        self._saved_complete = False
        self._last_reported = 0

    @property
    def complete(self) -> bool:
        return bool(np.all(self._completed[self._eligible]))

    def _accumulate(self, ids: np.ndarray) -> None:
        if not ids.size:
            return
        self._steps[ids] += 1
        self._valid_steps[ids] += self._grasp.valid_grasp[ids]
        self._thumb_steps[ids] += self._grasp.thumb_contact[ids]
        self._opposing_steps[ids] += self._grasp.opposing_contacts[ids]
        self._raw_thumb_steps[ids] += self._grasp.raw_thumb_contact[ids]
        self._raw_opposing_steps[ids] += self._grasp.raw_opposing_contacts[ids]
        self._palm_steps[ids] += self._grasp.palm_supported[ids]

    def record_post_reset(self, env_ids: np.ndarray) -> None:
        ids = np.asarray(env_ids, dtype=np.intp)
        self._steps[ids] = 0
        self._valid_steps[ids] = 0
        self._thumb_steps[ids] = 0
        self._opposing_steps[ids] = 0
        self._raw_thumb_steps[ids] = 0
        self._raw_opposing_steps[ids] = 0
        self._palm_steps[ids] = 0
        self._active_index[ids] = self._reset.grasp_cache_index[ids]

    def record_post_step(self) -> None:
        env = cast("ManagerBasedRlEnv", self._env)
        self._accumulate(np.flatnonzero(~env.reset_buf).astype(np.intp))

    def record_pre_reset(self, env_ids: np.ndarray) -> None:
        env = cast("ManagerBasedRlEnv", self._env)
        ids = np.asarray(env_ids, dtype=np.intp)
        self._accumulate(ids)
        for env_id in ids:
            cache_index = int(self._active_index[env_id])
            if (
                cache_index < 0
                or not self._eligible_mask[cache_index]
                or self._completed[cache_index]
            ):
                continue
            steps = max(int(self._steps[env_id]), 1)
            self._completed[cache_index] = True
            self._survival_seconds[cache_index] = steps * float(env.step_dt)
            self._valid_fraction[cache_index] = self._valid_steps[env_id] / steps
            self._thumb_fraction[cache_index] = self._thumb_steps[env_id] / steps
            self._opposing_fraction[cache_index] = self._opposing_steps[env_id] / steps
            self._raw_thumb_fraction[cache_index] = self._raw_thumb_steps[env_id] / steps
            self._raw_opposing_fraction[cache_index] = self._raw_opposing_steps[env_id] / steps
            self._palm_fraction[cache_index] = self._palm_steps[env_id] / steps
            self._termination_causes[cache_index] = (
                self._grasp.vertical_drop[env_id] >= self._grasp._termination_vertical_drop,
                self._grasp.lateral_drift[env_id] >= self._grasp._termination_lateral_drift,
                self._grasp.observation.object_pos[env_id, 2]
                < self._grasp._absolute_minimum_height,
            )

        completed = int(np.count_nonzero(self._completed[self._eligible]))
        env.extras.setdefault("log", {}).update(
            {
                "hora_passive_qualification/completed": float(completed),
                "hora_passive_qualification/total": float(self._eligible.size),
            }
        )
        if completed == self._eligible.size or completed - self._last_reported >= 250:
            print(
                f"[LeapHoraPassiveQualification] completed {completed}/{self._eligible.size} grasps"
            )
            self._last_reported = completed
        if self.complete:
            summary = self._save(complete=True)
            raise RunComplete(reason="hora_passive_qualification_complete", summary=summary)

    def _tier_ids(self) -> np.ndarray:
        seconds = self._survival_seconds
        tier = np.full(self._cache.size, -1, dtype=np.int8)
        tier[self._completed & (seconds < self._marginal_seconds)] = 0
        tier[
            self._completed
            & (seconds >= self._marginal_seconds)
            & (seconds < self._recoverable_seconds)
        ] = 1
        tier[
            self._completed
            & (seconds >= self._recoverable_seconds)
            & (seconds < self._anchor_seconds)
        ] = 2
        tier[self._completed & (seconds >= self._anchor_seconds)] = 3
        return tier

    def _paths(self) -> dict[str, Path]:
        parent = self._output_prefix.parent
        stem = self._output_prefix.name
        return {
            "scores": parent / f"{stem}_scores.npz",
            **{name: parent / f"{stem}_{name}.npz" for name in _TIER_NAMES},
            "stage_a": parent / f"{stem}_stage_a.npz",
        }

    def _stage_a_indices(self, tier: np.ndarray) -> np.ndarray:
        if (
            getattr(self, "_training_middle_contact_fraction", 0.0) > 0.0
            or getattr(self, "_minimum_raw_thumb_contact_fraction", 0.0) > 0.0
            or getattr(self, "_minimum_raw_opposing_fingers", 0) > 0
        ):
            selected, _ = select_contact_balanced_indices(
                eligible=self._eligible,
                tier=tier,
                family_id=self._cache.family_id,
                raw_thumb_fraction=self._raw_thumb_fraction,
                raw_opposing_fraction=self._raw_opposing_fraction,
                cache_size=self._training_cache_size,
                anchor_fraction=self._training_anchor_fraction,
                middle_contact_fraction=self._training_middle_contact_fraction,
                minimum_raw_thumb_contact_fraction=(self._minimum_raw_thumb_contact_fraction),
                minimum_raw_middle_contact_fraction=(self._minimum_raw_middle_contact_fraction),
                minimum_raw_opposing_contact_fraction=(self._minimum_raw_opposing_contact_fraction),
                minimum_raw_opposing_fingers=self._minimum_raw_opposing_fingers,
                seed=self._sampling_seed,
            )
            return selected
        rng = np.random.default_rng(self._sampling_seed)
        families = np.unique(self._cache.family_id[self._eligible])
        if not families.size:
            raise RuntimeError("cannot build Stage A cache without eligible families")
        base, remainder = divmod(self._training_cache_size, int(families.size))
        selected: list[np.ndarray] = []
        for position, family in enumerate(families):
            family_count = base + int(position < remainder)
            anchor_count = int(round(family_count * self._training_anchor_fraction))
            recoverable_count = family_count - anchor_count
            family_mask = self._cache.family_id == family
            anchor = np.flatnonzero(self._eligible_mask & family_mask & (tier == 3))
            recoverable = np.flatnonzero(self._eligible_mask & family_mask & (tier == 2))
            if anchor_count and not anchor.size:
                raise RuntimeError(
                    f"family {self._cache.family_names[int(family)]!r} has no anchor grasps"
                )
            if recoverable_count and not recoverable.size:
                raise RuntimeError(
                    f"family {self._cache.family_names[int(family)]!r} has no recoverable grasps"
                )
            if anchor_count:
                selected.append(
                    rng.choice(anchor, size=anchor_count, replace=anchor_count > anchor.size)
                )
            if recoverable_count:
                selected.append(
                    rng.choice(
                        recoverable,
                        size=recoverable_count,
                        replace=recoverable_count > recoverable.size,
                    )
                )
        result = np.concatenate(selected).astype(np.int64)
        rng.shuffle(result)
        return result

    def _counts_by_family(self, indices: np.ndarray) -> dict[str, int]:
        return {
            self._cache.family_names[int(family)]: int(
                np.count_nonzero(self._cache.family_id[indices] == family)
            )
            for family in np.unique(self._cache.family_id[indices])
        }

    def _save(self, *, complete: bool) -> dict[str, object]:
        paths = self._paths()
        paths["scores"].parent.mkdir(parents=True, exist_ok=True)
        tier = self._tier_ids()
        stage_a = None
        stage_a_enriched = None
        stage_a_selection_error: str | None = None
        if complete:
            try:
                if (
                    self._training_middle_contact_fraction > 0.0
                    or self._minimum_raw_thumb_contact_fraction > 0.0
                    or self._minimum_raw_opposing_fingers > 0
                ):
                    stage_a, stage_a_enriched = select_contact_balanced_indices(
                        eligible=self._eligible,
                        tier=tier,
                        family_id=self._cache.family_id,
                        raw_thumb_fraction=self._raw_thumb_fraction,
                        raw_opposing_fraction=self._raw_opposing_fraction,
                        cache_size=self._training_cache_size,
                        anchor_fraction=self._training_anchor_fraction,
                        middle_contact_fraction=self._training_middle_contact_fraction,
                        minimum_raw_thumb_contact_fraction=(
                            self._minimum_raw_thumb_contact_fraction
                        ),
                        minimum_raw_middle_contact_fraction=(
                            self._minimum_raw_middle_contact_fraction
                        ),
                        minimum_raw_opposing_contact_fraction=(
                            self._minimum_raw_opposing_contact_fraction
                        ),
                        minimum_raw_opposing_fingers=self._minimum_raw_opposing_fingers,
                        seed=self._sampling_seed,
                    )
                else:
                    stage_a = self._stage_a_indices(tier)
                    stage_a_enriched = np.zeros(stage_a.size, dtype=np.bool_)
            except RuntimeError as error:
                stage_a_selection_error = str(error)

        score_payload: dict[str, Any] = {
            "source_cache_path": np.asarray(str(self._input_path)),
            "cache_index": self._eligible,
            "survival_seconds": self._survival_seconds[self._eligible],
            "tier_id": tier[self._eligible],
            "tier_names": np.asarray(_TIER_NAMES, dtype=np.str_),
            "family_id": self._cache.family_id[self._eligible],
            "mean_valid_grasp_fraction": self._valid_fraction[self._eligible],
            "mean_thumb_contact_fraction": self._thumb_fraction[self._eligible],
            "mean_opposing_contact_fraction": self._opposing_fraction[self._eligible],
            "mean_raw_thumb_contact_fraction": self._raw_thumb_fraction[self._eligible],
            "mean_raw_opposing_contact_fraction": self._raw_opposing_fraction[self._eligible],
            "mean_palm_support_fraction": self._palm_fraction[self._eligible],
            "termination_vertical_escape": self._termination_causes[self._eligible, 0],
            "termination_lateral_escape": self._termination_causes[self._eligible, 1],
            "termination_absolute_height": self._termination_causes[self._eligible, 2],
            "complete": np.asarray(complete),
        }
        if stage_a is not None and stage_a_enriched is not None:
            score_payload["stage_a_source_cache_index"] = stage_a
            score_payload["stage_a_middle_enriched"] = stage_a_enriched
        np.savez_compressed(paths["scores"], **score_payload)
        tier_indices: dict[str, np.ndarray] = {}
        for tier_id, name in enumerate(_TIER_NAMES):
            indices = np.flatnonzero(self._eligible_mask & (tier == tier_id))
            tier_indices[name] = indices
            self._cache.take(indices).save(paths[name])

        stage_a_path: str | None = None
        stage_a_counts: Mapping[str, int] = {}
        stage_a_unique_source_rows = 0
        stage_a_raw_contact: dict[str, float] = {}
        if stage_a is not None:
            self._cache.take(stage_a).save(paths["stage_a"])
            stage_a_path = str(paths["stage_a"])
            stage_a_counts = self._counts_by_family(stage_a)
            stage_a_unique_source_rows = int(np.unique(stage_a).size)
            stage_a_raw_contact = {
                "thumb_mean": float(np.mean(self._raw_thumb_fraction[stage_a])),
                "index_mean": float(np.mean(self._raw_opposing_fraction[stage_a, 0])),
                "middle_mean": float(np.mean(self._raw_opposing_fraction[stage_a, 1])),
                "middle_median": float(np.median(self._raw_opposing_fraction[stage_a, 1])),
                "outer_mean": float(np.mean(self._raw_opposing_fraction[stage_a, 2])),
                "middle_threshold_fraction": float(
                    np.mean(
                        self._raw_opposing_fraction[stage_a, 1]
                        >= self._minimum_raw_middle_contact_fraction
                    )
                ),
            }
        if complete:
            self._saved_complete = True

        summary: dict[str, object] = {
            "eligible_grasps": int(self._eligible.size),
            "evaluated_grasps": int(np.count_nonzero(self._completed[self._eligible])),
            "tier_counts": {name: int(tier_indices[name].size) for name in _TIER_NAMES},
            "tier_family_counts": {
                name: self._counts_by_family(indices) for name, indices in tier_indices.items()
            },
            "score_report": str(paths["scores"]),
            "stage_a_cache": stage_a_path,
            "stage_a_family_counts": dict(stage_a_counts),
            "stage_a_anchor_fraction": self._training_anchor_fraction,
            "stage_a_middle_contact_target_fraction": (self._training_middle_contact_fraction),
            "stage_a_unique_source_rows": stage_a_unique_source_rows,
            "stage_a_raw_contact": stage_a_raw_contact,
            "stage_a_selection_error": stage_a_selection_error,
        }
        print(f"[LeapHoraPassiveQualification] {summary}")
        return summary

    def close(self) -> None:
        if not self._saved_complete and np.any(self._completed[self._eligible]):
            self._save(complete=self.complete)


__all__ = ["LeapHoraPassiveQualificationRecorder", "select_contact_balanced_indices"]
