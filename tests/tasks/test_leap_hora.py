"""Focused contracts for the isolated HORA-style LEAP task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from tensordict import TensorDict

from unilab.scripts.evaluate_leap_inhand import build_case_overrides
from unilab.tasks.manipulation.leap_inhand.evaluation import (
    load_evaluation_manifest,
    summarize_episode_reports,
)
from unilab.tasks.manipulation.leap_inhand.hora import (
    PROPERTY_NAMES,
    HoraActor,
    normalize_object_properties,
)
from unilab.tasks.manipulation.leap_inhand.hora_grasp import (
    HORA_GRASP_CACHE_SCHEMA_VERSION,
    HoraGraspCache,
    LeapHoraCylinderMassInertia,
    LeapHoraHandObjectReset,
)

_ROOT = Path(__file__).parents[2]
_CONF = _ROOT / "src" / "unilab" / "conf" / "ppo"
_MANIFEST = _ROOT / "experiments" / "leap_inhand" / "evaluation_manifest_v1.json"


def _ppo_cfg(task: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF), version_base="1.3"):
        return compose(config_name="config", overrides=[f"task={task}/mujoco"])


def test_hora_owner_isolated_from_gait_baseline() -> None:
    from unilab.base import registry
    from unilab.training import ensure_registries

    ensure_registries()
    assert registry.list_registered_envs()["LeapInhandHoraStage1"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco"],
    }
    baseline = _ppo_cfg("leap_inhand")
    hora = _ppo_cfg("leap_inhand_hora_stage1")

    assert baseline.training.task_name == "LeapInhandRotation"
    assert baseline.env.actions.hand._target_.endswith("LeapIncrementalPositionActionCfg")
    assert baseline.env.actions.hand.gait_amplitude_schedule[0].amplitude > 0.0

    assert hora.training.task_name == "LeapInhandHoraStage1"
    assert hora.env.actions.hand._target_.endswith("LeapHoraPositionActionCfg")
    assert hora.env.actions.hand.gait_amplitude_schedule[0].amplitude == 0.0
    assert hora.env.actions.hand.reset_target_from_joint_state is True
    assert hora.env.observations.actor.history_length is None
    assert hora.env.observations.actor.terms.hardware.history_length == 3
    assert hora.env.observations.actor.terms.object_properties.history_length == 0
    assert hora.algo.actor.property_dim == len(PROPERTY_NAMES)
    assert hora.algo.actor.extrinsics_dim == 8
    assert hora.algo.algorithm.enable_compile is False
    assert hora.reward.retained.weight == 0.0
    assert hora.reward.thumb_contact.weight == 0.0


def test_hora_grasp_owner_has_geometry_families_and_v2_output() -> None:
    from unilab.base import registry
    from unilab.training import ensure_registries

    ensure_registries()
    assert registry.list_registered_envs()["LeapInhandHoraGrasp"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco"],
    }
    cfg = _ppo_cfg("leap_inhand_hora_grasp")
    reset = cfg.env.events.reset_hand_object
    recorder = cfg.env.recorders.grasp_cache.params
    assert reset.func.endswith("LeapHoraHandObjectReset")
    assert len(reset.params.cylinder_geometries) == 3
    assert len(reset.params.canonical_families) == 5
    assert {family.split for family in reset.params.canonical_families} == {
        "train",
        "validation",
        "held_out",
    }
    assert cfg.env.events.cylinder_mass_inertia.func.endswith("LeapHoraCylinderMassInertia")
    assert recorder.output_path.startswith("caches/hora/")
    assert recorder.output_path.endswith(".npz")
    assert recorder.validation_trials == 3


def test_hora_grasp_cache_v2_round_trip_and_family_split(tmp_path: Path) -> None:
    count = 3
    quat = np.zeros((count, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    cache = HoraGraspCache(
        joint_pos=np.arange(count * 16, dtype=np.float32).reshape(count, 16) / 100.0,
        object_pos=np.zeros((count, 3), dtype=np.float32),
        object_quat=quat,
        cylinder_radius=np.asarray([0.028, 0.032, 0.036], dtype=np.float32),
        cylinder_length=np.asarray([0.060, 0.070, 0.080], dtype=np.float32),
        family_id=np.asarray([0, 0, 1], dtype=np.int16),
        split=np.asarray([0, 0, 2], dtype=np.int8),
        validation_trials=np.full(count, 3, dtype=np.int16),
        mass_min=np.asarray([0.03, 0.04, 0.05], dtype=np.float32),
        mass_max=np.asarray([0.05, 0.06, 0.08], dtype=np.float32),
        sliding_friction_min=np.asarray([0.7, 0.8, 0.9], dtype=np.float32),
        sliding_friction_max=np.asarray([1.1, 1.2, 1.3], dtype=np.float32),
        family_names=("train_family", "held_out_family"),
    )
    path = tmp_path / "hora_v2.npz"
    cache.save(path)
    loaded = HoraGraspCache.load(path)
    assert HORA_GRASP_CACHE_SCHEMA_VERSION == 2
    assert loaded.family_names == cache.family_names
    np.testing.assert_array_equal(loaded.state_rows, cache.state_rows)
    train = loaded.select(["train"])
    assert train.size == 2
    np.testing.assert_array_equal(train.family_id, 0)
    held_out = loaded.select(["held_out"])
    assert held_out.size == 1
    np.testing.assert_array_equal(held_out.family_id, 1)

    invalid = dict(cache.__dict__)
    invalid["split"] = np.asarray([0, 1, 2], dtype=np.int8)
    with pytest.raises(ValueError, match="family must belong to exactly one split"):
        HoraGraspCache(**invalid)


def test_hora_candidate_repeats_until_all_validation_trials_pass() -> None:
    reset = object.__new__(LeapHoraHandObjectReset)
    reset.validation_passes = np.zeros(2, dtype=np.int16)
    reset._repeat_candidate = np.zeros(2, dtype=np.bool_)
    reset._candidate_rows = np.zeros((2, 23), dtype=np.float64)
    reset._candidate_rows[:, 19] = 1.0
    ids = np.asarray([0, 1], dtype=np.int32)
    settled = np.ones((2, 23), dtype=np.float64)
    settled[:, 19:23] = (1.0, 0.0, 0.0, 0.0)

    accepted = reset.record_validation_results(ids, np.asarray([True, False]), settled, 2)
    np.testing.assert_array_equal(accepted, [False, False])
    np.testing.assert_array_equal(reset._repeat_candidate, [True, False])
    np.testing.assert_array_equal(reset.validation_passes, [1, 0])
    np.testing.assert_array_equal(reset._candidate_rows[0], settled[0])

    accepted = reset.record_validation_results(ids, np.asarray([True, True]), settled, 2)
    np.testing.assert_array_equal(accepted, [True, False])
    np.testing.assert_array_equal(reset._repeat_candidate, [False, True])
    np.testing.assert_array_equal(reset.validation_passes, [2, 1])


def test_hora_cylinder_mass_and_inertia_follow_sampled_geometry() -> None:
    from types import SimpleNamespace

    written: dict[str, np.ndarray] = {}
    geometry = object.__new__(LeapHoraHandObjectReset)
    geometry.current_radius = np.asarray([0.03, 0.04], dtype=np.float32)
    geometry.current_length = np.asarray([0.06, 0.08], dtype=np.float32)
    entity = SimpleNamespace(
        write_body_mass_to_sim=lambda values, body_ids, env_ids, term_name: written.update(
            mass=values.copy()
        ),
        write_body_inertia_to_sim=lambda values, body_ids, env_ids, term_name: written.update(
            inertia=values.copy()
        ),
    )
    env = SimpleNamespace(
        num_envs=2,
        rng=np.random.default_rng(7),
        event_manager=SimpleNamespace(
            get_term_cfg=lambda name: SimpleNamespace(func=geometry)
        ),
    )
    term = object.__new__(LeapHoraCylinderMassInertia)
    term._geometry_event_name = "reset_hand_object"
    term._density_range = (200.0, 200.0)
    term._entity = entity
    term._body_ids = np.asarray([0], dtype=np.int32)
    term.mass = np.full(2, np.nan, dtype=np.float32)
    term.density = np.full(2, np.nan, dtype=np.float32)

    term(cast(Any, env), np.asarray([0, 1], dtype=np.int32))

    expected_mass = 200.0 * np.pi * np.square(geometry.current_radius) * geometry.current_length
    np.testing.assert_allclose(written["mass"][:, 0], expected_mass)
    transverse = expected_mass * (
        3.0 * np.square(geometry.current_radius) + np.square(geometry.current_length)
    ) / 12.0
    axial = 0.5 * expected_mass * np.square(geometry.current_radius)
    np.testing.assert_allclose(
        written["inertia"][:, 0, :2],
        np.repeat(transverse[:, None], 2, axis=1),
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(written["inertia"][:, 0, 2], axial, rtol=1.0e-6)


def test_hora_actor_jointly_updates_encoder_and_supports_ablations() -> None:
    batch = 4
    hardware_dim = 3 * 32
    obs = TensorDict(
        {"actor": torch.randn(batch, hardware_dim + len(PROPERTY_NAMES))},
        batch_size=batch,
    )
    actor = HoraActor(
        obs,
        {"actor": ["actor"]},
        "actor",
        16,
        hidden_dims=[32, 16],
        encoder_hidden_dims=[16],
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": 0.1,
            "std_type": "scalar",
        },
    )
    output = actor(obs)
    assert output.shape == (batch, 16)
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in actor.extrinsics_encoder.parameters())

    actor.set_extrinsics_mode("zero")
    assert actor.get_latent(obs).shape == (batch, hardware_dim + 8)
    actor.set_extrinsics_mode("shuffled")
    assert actor.get_latent(obs).shape == (batch, hardware_dim + 8)
    actor.set_extrinsics_mode("frozen")
    first = actor.get_latent(obs)[:, -8:].clone()
    changed = TensorDict({"actor": obs["actor"] + torch.randn_like(obs["actor"])}, batch_size=batch)
    torch.testing.assert_close(actor.get_latent(changed)[:, -8:], first)
    actor.set_extrinsics_mode("predicted")
    predicted = torch.randn(batch, 8)
    actor.set_predicted_extrinsics(predicted)
    torch.testing.assert_close(actor.get_latent(obs)[:, -8:], predicted)

    exported = actor.as_onnx(verbose=False)
    assert exported.input_size == hardware_dim + len(PROPERTY_NAMES)
    assert exported.get_dummy_inputs()[0].shape == (1, exported.input_size)
    assert exported(torch.randn(1, exported.input_size)).shape == (1, 16)


def test_property_normalization_is_declared_ordered_and_bounded() -> None:
    ranges = {name: (-2.0, 2.0) for name in PROPERTY_NAMES}
    values = np.zeros((2, len(PROPERTY_NAMES)), dtype=np.float64)
    np.testing.assert_array_equal(normalize_object_properties(values, ranges), values)
    values[0, 0] = 2.1
    assert normalize_object_properties(values, ranges)[0, 0] == pytest.approx(1.05)
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        normalize_object_properties(values, ranges)


def test_evaluation_manifest_and_report_summary(tmp_path: Path) -> None:
    manifest = load_evaluation_manifest(_MANIFEST)
    assert len(manifest["cases"]) == 15
    assert {case["seed"] for case in manifest["cases"]} == {101, 102, 103, 104, 105}
    assert {case["split"] for case in manifest["cases"]} == {
        "in_distribution",
        "edge_of_range",
        "out_of_distribution",
    }
    case = manifest["cases"][0]
    overrides = build_case_overrides(case, tmp_path / "episode.json", policy_family="hora")
    assert "training.play_steps=400" in overrides
    assert "env.actions.hand.latency_steps_range=[0,0]" in overrides
    assert (
        "env.events.reset_hand_object.params.grasp_cache_path="
        "caches/hora/leap_precision_grasps_v2.npz"
    ) in overrides
    assert any("cylinder_geometry_override" in item for item in overrides)
    assert any("density_range" in item for item in overrides)
    assert any(item.startswith("++env.recorders.evaluation.func=") for item in overrides)

    metric_names = (
        "time_to_fall_seconds",
        "net_committed_axis_rotation",
        "accumulated_rotation",
        "directional_speed",
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
    reports = [
        {"split": split, **{name: float(index) for name in metric_names}}
        for index, split in enumerate(("in_distribution", "edge_of_range"), start=1)
    ]
    summary = summarize_episode_reports(reports)
    assert summary["overall"]["episodes"] == 2
    assert summary["overall"]["metrics"]["time_to_fall_seconds"]["median"] == 1.5

    invalid = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    invalid["cases"][0].pop("seed")
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="fields mismatch"):
        load_evaluation_manifest(invalid_path)


@pytest.mark.slow
def test_hora_mujoco_observation_action_and_ppo_contract() -> None:
    pytest.importorskip("mujoco")
    from rsl_rl.runners import OnPolicyRunner
    from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg

    from unilab.base import registry
    from unilab.scripts.train_rsl_rl import build_ppo_env_cfg_override
    from unilab.tasks.manipulation.leap_inhand.hora import LeapHoraPositionAction
    from unilab.training import algo_config_dict, ensure_registries

    cfg = _ppo_cfg("leap_inhand_hora_stage1")
    cfg.algo.num_envs = 2
    ensure_registries()
    env = registry.make(
        "LeapInhandHoraStage1",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=build_ppo_env_cfg_override(cfg),
    )
    try:
        state = env.init_state()
        assert env.obs_groups_spec == {"obs": 106, "critic": 63}
        assert state.obs["obs"].shape == (2, 106)
        action = env.action_manager.get_term("hand")
        assert isinstance(action, LeapHoraPositionAction)
        np.testing.assert_array_equal(action.gait_reference, 0.0)
        state = env.step(np.zeros((2, 16), dtype=np.float32))
        assert np.isfinite(state.obs["obs"]).all()

        wrapped = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(
            wrapped,
            normalize_ppo_train_cfg(algo_config_dict(cfg)),
            log_dir=None,
            device="cpu",
        )
        assert isinstance(runner.alg.actor, HoraActor)
        assert any(key.startswith("extrinsics_encoder") for key in runner.alg.actor.state_dict())
    finally:
        env.close()
