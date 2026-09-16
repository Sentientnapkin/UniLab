"""Policy-rollout qualification for LEAP stable-grasp caches."""

from __future__ import annotations

import csv
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.run_control import RunComplete
from unilab.managers import RecorderTerm, RecorderTermCfg

from .manager_terms import (
    LeapBidirectionalFrontierReward,
    LeapGraspTermination,
    LeapHandObjectReset,
)

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


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


def _fraction(term: str, field: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{term} {field} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{term} {field} must be finite and between 0 and 1")
    return result


def _nonnegative_real(term: str, field: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{term} {field} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{term} {field} must be finite and nonnegative")
    return result


def _asset_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(ASSETS_ROOT_PATH) / path


class LeapGraspQualificationRecorder(RecorderTerm):
    """Score every cached grasp using terminal and per-step policy outcomes.

    The associated reset term must use sequential cache sampling. Each cache row
    is evaluated repeatedly, then classified as mastered, frontier, or
    unsuitable. A weighted training cache contains mostly frontier rows plus a
    mastered anchor subset, allowing the existing random reset sampler to use
    the desired curriculum without a policy-step sampling dependency.
    """

    _ALLOWED_PARAMS = frozenset(
        {
            "reset_event_name",
            "grasp_term_name",
            "rotation_reward_name",
            "input_cache_path",
            "output_prefix",
            "episodes_per_grasp",
            "mastered_minimum_survival_fraction",
            "mastered_minimum_frontier_angle",
            "mastered_minimum_valid_grasp_fraction",
            "frontier_minimum_survival_fraction",
            "frontier_minimum_frontier_angle",
            "frontier_minimum_valid_grasp_fraction",
            "training_cache_size",
            "training_mastered_fraction",
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
        if not isinstance(reset, LeapHandObjectReset):
            raise TypeError(f"{term} reset event {reset_name!r} must be LeapHandObjectReset")
        if reset._grasp_cache is None or reset._grasp_cache_sampling != "sequential":
            raise ValueError(f"{term} requires a sequential cached LeapHandObjectReset")
        self._reset = reset

        grasp_name = _name(term, "grasp_term_name", cfg.params.get("grasp_term_name"))
        grasp = env.termination_manager.get_term_cfg(grasp_name).func
        if not isinstance(grasp, LeapGraspTermination):
            raise TypeError(f"{term} grasp term {grasp_name!r} must be LeapGraspTermination")
        self._grasp = grasp

        rotation_name = _name(
            term,
            "rotation_reward_name",
            cfg.params.get("rotation_reward_name"),
        )
        rotation = env.reward_manager.get_term_cfg(rotation_name).func
        if not isinstance(rotation, LeapBidirectionalFrontierReward):
            raise TypeError(
                f"{term} rotation reward {rotation_name!r} must be "
                "LeapBidirectionalFrontierReward"
            )
        self._rotation = rotation

        self._input_path = _asset_path(
            _name(term, "input_cache_path", cfg.params.get("input_cache_path"))
        )
        self._cache = np.asarray(np.load(self._input_path), dtype=np.float32)
        if self._cache.ndim != 2 or self._cache.shape[1] != 23 or not self._cache.shape[0]:
            raise ValueError(f"{term} input cache must have shape (N, 23), got {self._cache.shape}")
        if self._cache.shape != reset._grasp_cache.shape or not np.array_equal(
            self._cache,
            np.asarray(reset._grasp_cache, dtype=np.float32),
        ):
            raise ValueError(f"{term} input cache must match the reset event cache exactly")

        self._output_prefix = _asset_path(
            _name(term, "output_prefix", cfg.params.get("output_prefix"))
        )
        self._episodes_per_grasp = _positive_int(
            term,
            "episodes_per_grasp",
            cfg.params.get("episodes_per_grasp"),
        )
        self._mastered_survival = _fraction(
            term,
            "mastered_minimum_survival_fraction",
            cfg.params.get("mastered_minimum_survival_fraction"),
        )
        self._mastered_frontier = _nonnegative_real(
            term,
            "mastered_minimum_frontier_angle",
            cfg.params.get("mastered_minimum_frontier_angle"),
        )
        self._mastered_valid = _fraction(
            term,
            "mastered_minimum_valid_grasp_fraction",
            cfg.params.get("mastered_minimum_valid_grasp_fraction"),
        )
        self._frontier_survival = _fraction(
            term,
            "frontier_minimum_survival_fraction",
            cfg.params.get("frontier_minimum_survival_fraction"),
        )
        self._frontier_angle = _nonnegative_real(
            term,
            "frontier_minimum_frontier_angle",
            cfg.params.get("frontier_minimum_frontier_angle"),
        )
        self._frontier_valid = _fraction(
            term,
            "frontier_minimum_valid_grasp_fraction",
            cfg.params.get("frontier_minimum_valid_grasp_fraction"),
        )
        self._training_cache_size = _positive_int(
            term,
            "training_cache_size",
            cfg.params.get("training_cache_size"),
        )
        self._training_mastered_fraction = _fraction(
            term,
            "training_mastered_fraction",
            cfg.params.get("training_mastered_fraction"),
        )
        sampling_seed = cfg.params.get("sampling_seed")
        if isinstance(sampling_seed, (bool, np.bool_)) or not isinstance(
            sampling_seed, Integral
        ):
            raise TypeError(f"{term} sampling_seed must be an integer")
        self._sampling_seed = int(sampling_seed)

        cache_size = self._cache.shape[0]
        num_envs = env.num_envs
        self._episode_count = np.zeros(cache_size, dtype=np.int32)
        self._survival_sum = np.zeros(cache_size, dtype=np.float64)
        self._frontier_sum = np.zeros(cache_size, dtype=np.float64)
        self._valid_fraction_sum = np.zeros(cache_size, dtype=np.float64)
        self._directional_speed_sum = np.zeros(cache_size, dtype=np.float64)
        self._episode_steps = np.zeros(num_envs, dtype=np.int32)
        self._valid_steps = np.zeros(num_envs, dtype=np.int32)
        self._speed_sum = np.zeros(num_envs, dtype=np.float64)
        self._active_index = np.full(num_envs, -1, dtype=np.int64)
        self._saved_complete = False
        self._last_reported_completed = 0

    @property
    def complete(self) -> bool:
        return bool(np.all(self._episode_count >= self._episodes_per_grasp))

    def _accumulate(self, ids: np.ndarray) -> None:
        if not ids.size:
            return
        self._episode_steps[ids] += 1
        self._valid_steps[ids] += self._grasp.valid_grasp[ids]
        self._speed_sum[ids] += np.maximum(self._rotation.directional_speed[ids], 0.0)

    def record_post_reset(self, env_ids: np.ndarray) -> None:
        ids = np.asarray(env_ids, dtype=np.intp)
        self._episode_steps[ids] = 0
        self._valid_steps[ids] = 0
        self._speed_sum[ids] = 0.0
        self._active_index[ids] = self._reset.grasp_cache_index[ids]

    def record_post_step(self) -> None:
        env = cast("ManagerBasedRlEnv", self._env)
        ids = np.flatnonzero(~env.reset_buf).astype(np.intp)
        self._accumulate(ids)

    def record_pre_reset(self, env_ids: np.ndarray) -> None:
        env = cast("ManagerBasedRlEnv", self._env)
        ids = np.asarray(env_ids, dtype=np.intp)
        self._accumulate(ids)
        for env_id in ids:
            cache_index = int(self._active_index[env_id])
            if cache_index < 0 or self._episode_count[cache_index] >= self._episodes_per_grasp:
                continue
            steps = max(int(self._episode_steps[env_id]), 1)
            self._episode_count[cache_index] += 1
            self._survival_sum[cache_index] += float(
                env.reset_time_outs[env_id] and not env.reset_terminated[env_id]
            )
            self._frontier_sum[cache_index] += float(self._rotation.frontier_angle[env_id])
            self._valid_fraction_sum[cache_index] += self._valid_steps[env_id] / steps
            self._directional_speed_sum[cache_index] += self._speed_sum[env_id] / steps

        evaluated = int(np.count_nonzero(self._episode_count))
        completed = int(np.count_nonzero(self._episode_count >= self._episodes_per_grasp))
        env.extras.setdefault("log", {}).update(
            {
                "grasp_qualification/evaluated": float(evaluated),
                "grasp_qualification/completed": float(completed),
                "grasp_qualification/total": float(self._cache.shape[0]),
            }
        )
        if (
            completed == self._cache.shape[0]
            or completed - self._last_reported_completed >= 250
        ):
            print(
                "[LeapGraspQualification] "
                f"completed {completed}/{self._cache.shape[0]} grasps"
            )
            self._last_reported_completed = completed
        if self.complete:
            summary = self._save(complete=True)
            raise RunComplete(reason="grasp_qualification_complete", summary=summary)

    def _means(self, values: np.ndarray) -> np.ndarray:
        result = np.full(values.shape, np.nan, dtype=np.float64)
        np.divide(values, self._episode_count, out=result, where=self._episode_count > 0)
        return result

    def _paths(self) -> dict[str, Path]:
        parent = self._output_prefix.parent
        stem = self._output_prefix.name
        return {
            name: parent / f"{stem}_{suffix}"
            for name, suffix in {
                "scores": "scores.npz",
                "csv": "scores.csv",
                "mastered": "mastered.npy",
                "frontier": "frontier.npy",
                "unsuitable": "unsuitable.npy",
                "training": "training.npy",
            }.items()
        }

    def _training_rows(self, mastered: np.ndarray, frontier: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self._sampling_seed)
        mastered_count = int(round(self._training_cache_size * self._training_mastered_fraction))
        frontier_count = self._training_cache_size - mastered_count
        if not mastered.size:
            frontier_count = self._training_cache_size
            mastered_count = 0
        if not frontier.size:
            mastered_count = self._training_cache_size
            frontier_count = 0
        if not mastered.size and not frontier.size:
            return np.empty((0, self._cache.shape[1]), dtype=self._cache.dtype)
        pieces: list[np.ndarray] = []
        if mastered_count:
            pieces.append(rng.choice(mastered, size=mastered_count, replace=True))
        if frontier_count:
            pieces.append(rng.choice(frontier, size=frontier_count, replace=True))
        indices = np.concatenate(pieces)
        rng.shuffle(indices)
        return self._cache[indices]

    def _save(self, *, complete: bool) -> dict[str, object]:
        paths = self._paths()
        paths["scores"].parent.mkdir(parents=True, exist_ok=True)
        survival = self._means(self._survival_sum)
        frontier_angle = self._means(self._frontier_sum)
        valid_fraction = self._means(self._valid_fraction_sum)
        directional_speed = self._means(self._directional_speed_sum)
        fully_evaluated = self._episode_count >= self._episodes_per_grasp
        mastered_mask = (
            fully_evaluated
            & (survival >= self._mastered_survival)
            & (frontier_angle >= self._mastered_frontier)
            & (valid_fraction >= self._mastered_valid)
        )
        frontier_mask = (
            fully_evaluated
            & ~mastered_mask
            & (survival >= self._frontier_survival)
            & (frontier_angle >= self._frontier_angle)
            & (valid_fraction >= self._frontier_valid)
        )
        unsuitable_mask = fully_evaluated & ~mastered_mask & ~frontier_mask
        mastered = np.flatnonzero(mastered_mask)
        frontier = np.flatnonzero(frontier_mask)
        unsuitable = np.flatnonzero(unsuitable_mask)

        np.savez(
            paths["scores"],
            cache_index=np.arange(self._cache.shape[0], dtype=np.int64),
            episode_count=self._episode_count,
            survival_fraction=survival,
            mean_frontier_angle=frontier_angle,
            mean_valid_grasp_fraction=valid_fraction,
            mean_directional_speed=directional_speed,
            classification=np.where(mastered_mask, 2, np.where(frontier_mask, 1, 0)),
            complete=np.asarray(complete),
        )
        with paths["csv"].open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "cache_index",
                    "episode_count",
                    "survival_fraction",
                    "mean_frontier_angle",
                    "mean_valid_grasp_fraction",
                    "mean_directional_speed",
                    "classification",
                )
            )
            labels = np.where(mastered_mask, "mastered", np.where(frontier_mask, "frontier", "unsuitable"))
            for index in range(self._cache.shape[0]):
                writer.writerow(
                    (
                        index,
                        int(self._episode_count[index]),
                        survival[index],
                        frontier_angle[index],
                        valid_fraction[index],
                        directional_speed[index],
                        labels[index] if fully_evaluated[index] else "incomplete",
                    )
                )
        np.save(paths["mastered"], self._cache[mastered])
        np.save(paths["frontier"], self._cache[frontier])
        np.save(paths["unsuitable"], self._cache[unsuitable])
        if complete:
            np.save(paths["training"], self._training_rows(mastered, frontier))
            self._saved_complete = True

        summary: dict[str, object] = {
            "evaluated_grasps": int(np.count_nonzero(fully_evaluated)),
            "total_grasps": int(self._cache.shape[0]),
            "episodes_per_grasp": self._episodes_per_grasp,
            "mastered_grasps": int(mastered.size),
            "frontier_grasps": int(frontier.size),
            "unsuitable_grasps": int(unsuitable.size),
            "score_report": str(paths["scores"]),
            "training_cache": str(paths["training"]) if complete else None,
        }
        print(f"[LeapGraspQualification] {summary}")
        return summary

    def close(self) -> None:
        if not self._saved_complete:
            self._save(complete=self.complete)


__all__ = ["LeapGraspQualificationRecorder"]
