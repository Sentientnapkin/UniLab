"""Focused contracts for the isolated HORA-style LEAP task."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from tensordict import TensorDict

from unilab.base.config_adapter import BackendAdapter
from unilab.base.run_control import RunComplete
from unilab.scripts import (
    audit_leap_hora_rollout,
    build_leap_hora_contact_cache,
    build_leap_hora_v5_candidates,
    generate_leap_hora_grasps,
    generate_leap_hora_middle_grasps,
    qualify_leap_hora_grasps,
    rebuild_leap_hora_stage_a,
    sweep_leap_hora_checkpoints,
)
from unilab.scripts.evaluate_leap_inhand import build_case_overrides
from unilab.scripts.generate_leap_hora_grasps import (
    collection_plan,
    compose_collection_config,
)
from unilab.scripts.generate_leap_hora_grasps import (
    main as generate_hora_grasps_main,
)
from unilab.scripts.qualify_leap_hora_grasps import (
    compose_qualification_config,
    qualification_plan,
)
from unilab.tasks.manipulation.leap_inhand.evaluation import (
    load_evaluation_manifest,
    rank_hora_checkpoint_evaluations,
    summarize_episode_reports,
    summarize_hora_rollout_audits,
)
from unilab.tasks.manipulation.leap_inhand.hora import (
    PROPERTY_NAMES,
    HoraActor,
    LeapHoraPositionAction,
    normalize_object_properties,
)
from unilab.tasks.manipulation.leap_inhand.hora_grasp import (
    HORA_GRASP_CACHE_SCHEMA_VERSION,
    HoraGraspCache,
    LeapHoraCylinderMassInertia,
    LeapHoraGraspRecorder,
    LeapHoraHandObjectReset,
    make_leap_hora_env,
    merge_hora_grasp_caches,
)
from unilab.tasks.manipulation.leap_inhand.hora_passive_qualification import (
    LeapHoraPassiveQualificationRecorder,
    select_contact_balanced_indices,
)

_ROOT = Path(__file__).parents[2]
_CONF = _ROOT / "src" / "unilab" / "conf" / "ppo"
_MANIFEST = _ROOT / "experiments" / "leap_inhand" / "evaluation_manifest_v1.json"


def _ppo_cfg(task: str, owner: str = "mujoco"):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF), version_base="1.3"):
        return compose(config_name="config", overrides=[f"task={task}/{owner}"])


def _hora_action_for_contract_test(*, latency_steps: tuple[int, ...]) -> LeapHoraPositionAction:
    """Build the numerical action state without constructing a simulator."""
    num_envs = len(latency_steps)
    action_dim = 16
    action = object.__new__(LeapHoraPositionAction)
    action._env = SimpleNamespace(
        num_envs=num_envs,
        common_step_counter=0,
        step_dt=0.05,
        rng=np.random.default_rng(7),
    )
    action._scale = 1.0 / 24.0
    action._raw_clip = (-1.0, 1.0)
    action._ctrl_lower = np.full(action_dim, -0.06, dtype=np.float32)
    action._ctrl_upper = np.full(action_dim, 0.06, dtype=np.float32)
    action._raw_action = np.zeros((num_envs, action_dim), dtype=np.float32)
    action._previous_policy_action = np.zeros_like(action._raw_action)
    action._effective_action = np.zeros_like(action._raw_action)
    action._command_noise = np.zeros_like(action._raw_action)
    action._command_delayed = np.zeros(num_envs, dtype=np.bool_)
    action._clipped_action = np.zeros_like(action._raw_action)
    action._episode_anchor = np.zeros_like(action._raw_action)
    action._policy_target = np.zeros_like(action._raw_action)
    action._gait_reference = np.zeros_like(action._raw_action)
    action._target = np.zeros_like(action._raw_action)
    action._action_noise_std_schedule = ((0, 0.0),)
    action._latency_steps = np.asarray(latency_steps, dtype=np.int64)
    action._latency_steps_range = (min(latency_steps), max(latency_steps))
    action._action_history = np.zeros(
        (num_envs, max(latency_steps) + 1, action_dim), dtype=np.float32
    )
    return action


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
    assert hora.env.actions.hand.action_scale == pytest.approx(1.0 / 24.0)
    assert hora.env.actions.hand.thumb_policy_target_offset_limit is None
    assert hora.env.actions.hand.thumb_target_return_time_constant is None
    assert hora.env.actions.hand.opposing_finger_policy_target_offset_limit is None
    assert hora.env.actions.hand.opposing_finger_target_return_time_constant is None
    assert hora.env.observations.actor.history_length is None
    assert hora.env.observations.actor.terms.hardware.history_length == 3
    assert hora.env.observations.actor.terms.object_properties.history_length == 0
    assert hora.algo.actor.property_dim == len(PROPERTY_NAMES)
    assert hora.algo.actor.extrinsics_dim == 8
    assert hora.algo.algorithm.enable_compile is False
    assert hora.reward.retained.weight == 0.0
    assert hora.reward.thumb_contact.weight == 0.0
    assert hora.reward.frontier.weight == 0.0
    assert hora.reward.rotate.weight == pytest.approx(1.0)
    assert hora.reward.obj_linvel.weight == pytest.approx(-0.3)
    assert hora.reward.pose_diff.weight == pytest.approx(-0.3)
    assert hora.reward.torque.weight == pytest.approx(-0.1)
    assert hora.reward.work.weight == pytest.approx(-2.0)
    assert hora.reward.drop.weight == pytest.approx(0.0)
    assert hora.env.metrics.mean_axial_rotation_velocity.func.endswith(
        "hora_axial_rotation_velocity"
    )


def test_hora_stage1_nominal_and_no_extrinsics_are_matched_profiles() -> None:
    exact = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_nominal")
    control = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_no_extrinsics")

    assert exact.algo.experiment_name == "leap_hora_stage1_nominal_exact"
    assert control.algo.experiment_name == "leap_hora_stage1_nominal_no_extrinsics"
    assert exact.algo.actor.extrinsics_mode == "exact"
    assert control.algo.actor.extrinsics_mode == "zero"
    assert exact.interactive.policy_obs_mode == "actor"
    assert control.interactive.policy_obs_mode == "actor"
    assert exact.env.events.reset_hand_object.params.cylinder_geometry_override == [
        0.032,
        0.070,
    ]
    exact_container = OmegaConf.to_container(exact, resolve=True)
    control_container = OmegaConf.to_container(control, resolve=True)
    assert isinstance(exact_container, dict) and isinstance(control_container, dict)
    exact_container["algo"]["actor"]["extrinsics_mode"] = "zero"
    exact_container["algo"]["experiment_name"] = control_container["algo"]["experiment_name"]
    assert exact_container == control_container
    assert (
        exact.env.observations.actor.terms.object_properties.params.property_ranges.object_mass
        == [
            0.020,
            0.120,
        ]
    )


def test_hora_stabilization_profile_uses_qualified_cache_and_near_passive_actor() -> None:
    cfg = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_stabilize")

    assert cfg.algo.experiment_name == "leap_hora_stage1_stabilize_exact"
    assert cfg.algo.actor.extrinsics_mode == "exact"
    assert cfg.algo.actor.zero_mean_output_init is True
    assert cfg.algo.actor.distribution_cfg.init_std == pytest.approx(0.03)
    assert cfg.env.events.reset_hand_object.params.grasp_cache_path.endswith(
        "leap_precision_grasps_v5_stage_a_weighted.npz"
    )
    assert cfg.reward.retained.weight == pytest.approx(1.0)
    assert cfg.reward.rotate.weight == pytest.approx(0.1)
    assert cfg.reward.pose_diff.weight == pytest.approx(-0.03)
    assert cfg.reward.obj_linvel.weight == pytest.approx(-0.3)
    assert cfg.reward.torque.weight == pytest.approx(-0.1)
    assert cfg.reward.work.weight == pytest.approx(-1.0)
    assert cfg.reward.drop.weight == pytest.approx(-20.0)


def test_hora_rotate_b2_profile_changes_only_rotation_retention_tradeoff() -> None:
    stabilize = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_stabilize")
    rotate = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_rotate_b2")

    assert rotate.algo.experiment_name == "leap_hora_stage1_rotate_b2_exact"
    assert rotate.algo.actor.extrinsics_mode == "exact"
    assert rotate.env.events.reset_hand_object.params.grasp_cache_path.endswith(
        "leap_precision_grasps_v5_stage_a_weighted.npz"
    )
    assert rotate.reward.rotate.weight == pytest.approx(1.0)
    assert rotate.reward.retained.weight == pytest.approx(0.5)
    for term in ("pose_diff", "obj_linvel", "torque", "work", "drop"):
        assert rotate.reward[term].weight == stabilize.reward[term].weight


def test_hora_rotate_b3_restores_literal_hora_pose_penalty() -> None:
    b2 = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_rotate_b2")
    b3 = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_rotate_b3")

    assert b3.algo.experiment_name == "leap_hora_stage1_rotate_b3_exact"
    assert b3.reward.rotate.weight == pytest.approx(1.0)
    assert b3.reward.retained.weight == pytest.approx(0.5)
    assert b3.reward.pose_diff.weight == pytest.approx(-0.3)
    for term in ("obj_linvel", "torque", "work", "drop"):
        assert b3.reward[term].weight == b2.reward[term].weight
    assert (
        b3.env.events.reset_hand_object.params.grasp_cache_path
        == b2.env.events.reset_hand_object.params.grasp_cache_path
    )


def test_hora_rotate_b4_relaxes_only_pose_penalty() -> None:
    b3 = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_rotate_b3")
    b4 = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_rotate_b4")

    assert b4.algo.experiment_name == "leap_hora_stage1_rotate_b4_exact"
    assert b4.reward.pose_diff.weight == pytest.approx(-0.1)
    for term in ("rotate", "retained", "obj_linvel", "torque", "work", "drop"):
        assert b4.reward[term].weight == b3.reward[term].weight
    assert OmegaConf.to_container(b4.env, resolve=True) == OmegaConf.to_container(
        b3.env, resolve=True
    )


def test_hora_rotate_b5_increases_only_rotation_reward() -> None:
    b4 = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_rotate_b4")
    b5 = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_rotate_b5")

    assert b5.algo.experiment_name == "leap_hora_stage1_rotate_b5_exact"
    assert b5.reward.rotate.weight == pytest.approx(2.0)
    for term in ("retained", "pose_diff", "obj_linvel", "torque", "work", "drop"):
        assert b5.reward[term].weight == b4.reward[term].weight
    assert OmegaConf.to_container(b5.env, resolve=True) == OmegaConf.to_container(
        b4.env, resolve=True
    )


def test_hora_rotate_b6_rebalances_retention_and_pose() -> None:
    b5 = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_rotate_b5")
    b6 = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_rotate_b6")

    assert b6.algo.experiment_name == "leap_hora_stage1_rotate_b6_exact"
    assert b6.reward.rotate.weight == pytest.approx(2.0)
    assert b6.reward.retained.weight == pytest.approx(0.1)
    assert b6.reward.pose_diff.weight == pytest.approx(-0.15)
    for term in ("obj_linvel", "torque", "work", "drop"):
        assert b6.reward[term].weight == b5.reward[term].weight
    assert OmegaConf.to_container(b6.env, resolve=True) == OmegaConf.to_container(
        b5.env, resolve=True
    )


def test_hora_rotate_b7_increases_motion_incentive() -> None:
    b6 = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_rotate_b6")
    b7 = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_rotate_b7")

    assert b7.algo.experiment_name == "leap_hora_stage1_rotate_b7_exact"
    assert b7.reward.rotate.weight == pytest.approx(3.0)
    assert b7.reward.retained.weight == pytest.approx(0.1)
    assert b7.reward.pose_diff.weight == pytest.approx(-0.1)
    for term in ("obj_linvel", "torque", "work", "drop"):
        assert b7.reward[term].weight == b6.reward[term].weight
    assert OmegaConf.to_container(b7.env, resolve=True) == OmegaConf.to_container(
        b6.env, resolve=True
    )


def test_hora_rotate_b8_uses_paper_reward_balance() -> None:
    stabilize = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_stabilize")
    b8 = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_rotate_b8")

    assert b8.algo.experiment_name == "leap_hora_stage1_rotate_b8_exact"
    assert b8.reward.rotate.weight == pytest.approx(1.0)
    assert b8.reward.pose_diff.weight == pytest.approx(-0.3)
    assert b8.reward.obj_linvel.weight == pytest.approx(-0.3)
    assert b8.reward.torque.weight == pytest.approx(-0.1)
    assert b8.reward.work.weight == pytest.approx(-2.0)
    assert b8.reward.retained.weight == pytest.approx(0.0)
    assert b8.reward.frontier.weight == pytest.approx(0.0)
    assert b8.reward.drop.weight == pytest.approx(0.0)
    assert OmegaConf.to_container(b8.env, resolve=True) == OmegaConf.to_container(
        stabilize.env, resolve=True
    )


def test_hora_rotate_b8_restart_is_actor_only_warm_start() -> None:
    b8 = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_rotate_b8")
    restart = _ppo_cfg("leap_inhand_hora_stage1", "mujoco_rotate_b8_restart")

    assert restart.algo.experiment_name == "leap_hora_stage1_rotate_b8_restart_exact"
    assert restart.training.resume_mode == "actor_only"
    assert OmegaConf.to_container(restart.env, resolve=True) == OmegaConf.to_container(
        b8.env, resolve=True
    )
    assert OmegaConf.to_container(restart.reward, resolve=True) == OmegaConf.to_container(
        b8.reward, resolve=True
    )


def test_hora_action_accumulates_persists_clips_and_delays() -> None:
    action = _hora_action_for_contract_test(latency_steps=(0, 1))
    command = np.full((2, 16), 0.6, dtype=np.float32)

    action.process_actions(command)
    np.testing.assert_allclose(action.policy_target[0], 0.6 / 24.0)
    np.testing.assert_allclose(action.policy_target[1], 0.0)
    np.testing.assert_array_equal(action.command_delayed, [False, True])

    action.process_actions(np.zeros_like(command))
    np.testing.assert_allclose(action.policy_target[0], 0.6 / 24.0)
    np.testing.assert_allclose(action.policy_target[1], 0.6 / 24.0)
    np.testing.assert_allclose(action.latency_seconds, [0.0, 0.05])

    action.process_actions(np.ones_like(command))
    action.process_actions(np.ones_like(command))
    np.testing.assert_allclose(action.policy_target[0], 0.06)
    assert np.all(action.policy_target <= action.ctrl_upper)
    assert np.all(action.policy_target >= action.ctrl_lower)
    np.testing.assert_array_equal(action.gait_reference, 0.0)


def test_hora_action_rejects_invalid_values() -> None:
    action = _hora_action_for_contract_test(latency_steps=(0,))

    with pytest.raises(TypeError, match="expected np.ndarray"):
        action.process_actions(cast(Any, [[0.0] * 16]))
    with pytest.raises(ValueError, match="expected action shape"):
        action.process_actions(np.zeros((1, 15), dtype=np.float32))
    invalid = np.zeros((1, 16), dtype=np.float32)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        action.process_actions(invalid)


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
    assert tuple(recorder.self_collision_sensor_names) == (
        "if_mf_self_contact",
        "if_rf_self_contact",
        "mf_rf_self_contact",
        "th_if_self_contact",
        "th_mf_self_contact",
        "th_rf_self_contact",
    )


def test_hora_grasp_self_collision_rejection_reads_named_sensor_view() -> None:
    recorder = object.__new__(LeapHoraGraspRecorder)
    recorder._env = SimpleNamespace(num_envs=3)
    recorder._self_collision_sensor_names = ("if_mf", "th_if")
    recorder._self_collision_threshold = 0.0
    recorder._self_collision_view = SimpleNamespace(
        read=lambda: np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    )

    collision = recorder._self_collision(np.asarray([0, 2], dtype=np.intp))

    np.testing.assert_array_equal(collision, [False, True])


def test_hora_grasp_collection_command_composes_without_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "hora_smoke.npz"
    cfg = compose_collection_config(num_envs=4, target=8, output_path=output, seed=7)
    plan = collection_plan(cfg, max_control_steps=1000)
    assert plan.num_envs == 4
    assert plan.target == 8
    assert plan.output_path == output.resolve()
    assert plan.validation_trials == 3
    assert plan.geometries == ((0.028, 0.06), (0.032, 0.07), (0.036, 0.08))
    assert {split for _, split in plan.families} == {"train", "validation", "held_out"}

    result = generate_hora_grasps_main(
        [
            "--describe-only",
            "--num-envs",
            "4",
            "--target",
            "8",
            "--seed",
            "7",
            "--max-control-steps",
            "1000",
            "--output",
            str(output),
        ]
    )
    assert result == 0
    described = json.loads(capsys.readouterr().out)["collection_plan"]
    assert described["schema_version"] == 2
    assert described["collection_target"] == 8


def test_hora_passive_qualification_command_targets_medium_train_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = _ROOT / "src/unilab/assets/caches/hora/leap_precision_grasps_v2.npz"
    output_prefix = tmp_path / "qualified"
    cfg = compose_qualification_config(
        num_envs=32,
        seed=9,
        input_path=input_path,
        output_prefix=output_prefix,
        anchor_minimum_seconds=16.0,
        recoverable_minimum_seconds=1.0,
        marginal_minimum_seconds=0.5,
        stage_a_size=3000,
        stage_a_anchor_fraction=0.8,
    )
    plan = qualification_plan(cfg, max_control_steps=30000)

    assert plan.num_envs == 32
    assert plan.eligible_grasps == 2736
    assert sum(plan.family_counts.values()) == 2736
    assert plan.stage_a_size == 3000
    assert plan.stage_a_anchor_fraction == pytest.approx(0.8)
    result = qualify_leap_hora_grasps.main(
        [
            "--describe-only",
            "--num-envs",
            "32",
            "--seed",
            "9",
            "--max-control-steps",
            "30000",
            "--input",
            str(input_path),
            "--output-prefix",
            str(output_prefix),
        ]
    )
    assert result == 0
    described = json.loads(capsys.readouterr().out)["qualification_plan"]
    assert described["eligible_grasps"] == 2736
    assert described["tier_thresholds_seconds"] == {
        "anchor": 16.0,
        "recoverable": 1.0,
        "marginal": 0.5,
    }


def test_contact_balanced_cache_selection_preserves_marginal_quotas() -> None:
    tier = np.asarray([3, 2, 3, 2, 3, 2], dtype=np.int8)
    family = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int16)
    raw_thumb = np.ones(6, dtype=np.float64)
    raw_opposing = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    selected, enriched = select_contact_balanced_indices(
        eligible=np.arange(6),
        tier=tier,
        family_id=family,
        raw_thumb_fraction=raw_thumb,
        raw_opposing_fraction=raw_opposing,
        cache_size=12,
        anchor_fraction=0.75,
        middle_contact_fraction=0.5,
        minimum_raw_thumb_contact_fraction=0.5,
        minimum_raw_middle_contact_fraction=0.5,
        minimum_raw_opposing_contact_fraction=0.5,
        minimum_raw_opposing_fingers=2,
        seed=7,
    )

    assert selected.size == 12
    assert np.count_nonzero(enriched) == 6
    np.testing.assert_array_equal(np.bincount(family[selected], minlength=3), [4, 4, 4])
    assert np.count_nonzero(tier[selected] == 3) == 9
    assert np.all(raw_opposing[selected[enriched], 1] >= 0.5)
    assert np.unique(selected).size < selected.size


def test_contact_balanced_selection_prefers_diverse_middle_pool() -> None:
    family = np.repeat(np.arange(3, dtype=np.int16), [2, 6, 2])
    tier = np.full(family.size, 3, dtype=np.int8)
    raw_opposing = np.ones((family.size, 3), dtype=np.float64)
    raw_opposing[:, 1] = 0.0
    raw_opposing[0, 1] = 1.0
    raw_opposing[2:8, 1] = 1.0

    selected, enriched = select_contact_balanced_indices(
        eligible=np.arange(family.size),
        tier=tier,
        family_id=family,
        raw_thumb_fraction=np.ones(family.size, dtype=np.float64),
        raw_opposing_fraction=raw_opposing,
        cache_size=12,
        anchor_fraction=1.0,
        middle_contact_fraction=0.5,
        minimum_raw_thumb_contact_fraction=0.5,
        minimum_raw_middle_contact_fraction=0.5,
        minimum_raw_opposing_contact_fraction=0.5,
        minimum_raw_opposing_fingers=2,
        seed=7,
    )

    enriched_indices = selected[enriched]
    assert np.count_nonzero(family[enriched_indices] == 1) == 4
    assert np.count_nonzero(family[enriched_indices] == 0) == 2
    assert np.unique(enriched_indices).size == 5


def test_contact_cache_builder_describes_raw_contact_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = _ROOT / "src/unilab/assets/caches/hora/leap_precision_grasps_v2.npz"
    result = build_leap_hora_contact_cache.main(
        [
            "--describe-only",
            "--max-control-steps",
            "30000",
            "--input",
            str(input_path),
            "--output-prefix",
            str(tmp_path / "contact_pilot"),
        ]
    )
    assert result == 0
    described = json.loads(capsys.readouterr().out)
    assert described["qualification_plan"]["stage_a_size"] == 1000
    assert described["contact_selection"] == {
        "anchor_fraction": 0.8,
        "cache_size": 1000,
        "middle_contact_fraction": 0.4,
        "minimum_raw_middle_contact_fraction": 0.5,
        "minimum_raw_opposing_contact_fraction": 0.5,
        "minimum_raw_opposing_fingers": 2,
        "minimum_raw_thumb_contact_fraction": 0.5,
    }


def test_offline_stage_a_rebuild_honors_explicit_family_targets(tmp_path: Path) -> None:
    names = ("balanced", "index", "outer", "middle")
    count = 8
    cache = HoraGraspCache(
        joint_pos=np.arange(count * 16, dtype=np.float32).reshape(count, 16) / 100.0,
        object_pos=np.zeros((count, 3), dtype=np.float32),
        object_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]] * count, dtype=np.float32),
        cylinder_radius=np.full(count, 0.032, dtype=np.float32),
        cylinder_length=np.full(count, 0.070, dtype=np.float32),
        family_id=np.repeat(np.arange(4, dtype=np.int16), 2),
        split=np.zeros(count, dtype=np.int8),
        validation_trials=np.full(count, 3, dtype=np.int16),
        mass_min=np.full(count, 0.03, dtype=np.float32),
        mass_max=np.full(count, 0.08, dtype=np.float32),
        sliding_friction_min=np.full(count, 0.7, dtype=np.float32),
        sliding_friction_max=np.full(count, 1.4, dtype=np.float32),
        family_names=names,
    )
    input_path = tmp_path / "candidates.npz"
    score_path = tmp_path / "scores.npz"
    output_path = tmp_path / "stage_a.npz"
    cache.save(input_path)
    raw_opposing = np.ones((count, 3), dtype=np.float64)
    raw_opposing[:6, 1] = 0.0
    np.savez_compressed(
        score_path,
        complete=np.asarray(True),
        cache_index=np.arange(count),
        family_id=cache.family_id,
        tier_id=np.tile(np.asarray([3, 2], dtype=np.int8), 4),
        tier_names=np.asarray(("invalid", "marginal", "recoverable", "anchor")),
        mean_raw_thumb_contact_fraction=np.ones(count),
        mean_raw_opposing_contact_fraction=raw_opposing,
    )

    argv = [
        "--input",
        str(input_path),
        "--scores",
        str(score_path),
        "--output",
        str(output_path),
        "--anchor-fraction",
        "0.5",
        "--middle-contact-fraction",
        "0.25",
    ]
    for name in names:
        argv.extend(("--family-target", f"{name}=2"))
    assert rebuild_leap_hora_stage_a.main(argv) == 0

    rebuilt = HoraGraspCache.load(output_path)
    np.testing.assert_array_equal(np.bincount(rebuilt.family_id, minlength=4), [2, 2, 2, 2])
    report = json.loads(output_path.with_suffix(".json").read_text())
    assert report["unique_source_rows"] == 8
    assert report["unique_middle_enriched_source_rows"] == 2
    assert report["tier_counts"] == {
        "anchor": 4,
        "invalid": 0,
        "marginal": 0,
        "recoverable": 4,
    }


def test_middle_contact_seed_selection_is_diverse_and_covers_source_families() -> None:
    count = 6
    quat = np.zeros((count, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    joints = np.zeros((count, 16), dtype=np.float32)
    for index in range(count):
        joints[index, index] = 0.1 * (index + 1)
    source = HoraGraspCache(
        joint_pos=joints,
        object_pos=np.zeros((count, 3), dtype=np.float32),
        object_quat=quat,
        cylinder_radius=np.full(count, 0.032, dtype=np.float32),
        cylinder_length=np.full(count, 0.070, dtype=np.float32),
        family_id=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int16),
        split=np.zeros(count, dtype=np.int8),
        validation_trials=np.full(count, 3, dtype=np.int16),
        mass_min=np.full(count, 0.03, dtype=np.float32),
        mass_max=np.full(count, 0.07, dtype=np.float32),
        sliding_friction_min=np.full(count, 0.7, dtype=np.float32),
        sliding_friction_max=np.full(count, 1.3, dtype=np.float32),
        family_names=("first", "second"),
    )
    scores = {
        "cache_index": np.arange(count),
        "tier_id": np.full(count, 3, dtype=np.int8),
        "mean_raw_thumb_contact_fraction": np.full(count, 0.9),
        "mean_raw_opposing_contact_fraction": np.asarray(
            [[0.9, 0.6 + 0.05 * index, 0.9] for index in range(count)]
        ),
    }

    selected = generate_leap_hora_middle_grasps.select_middle_contact_seed_indices(
        source,
        scores,
        seed_count=4,
        minimum_raw_contact=0.5,
    )

    assert selected.size == 4
    assert np.unique(selected).size == 4
    assert set(source.family_id[selected]) == {0, 1}


def test_hora_factory_assigns_compiled_geometry_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SimpleNamespace(
        events={
            "reset_hand_object": SimpleNamespace(
                params={
                    "object_geom_name": "object_visual",
                    "cylinder_geometries": ((0.028, 0.060), (0.032, 0.070), (0.036, 0.080)),
                }
            )
        }
    )
    captured: dict[str, Any] = {}

    def fake_make(cfg: Any, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "env"

    monkeypatch.setattr(
        "unilab.tasks.manipulation.leap_inhand.hora_grasp.make_manager_based_rl_env",
        fake_make,
    )

    result = make_leap_hora_env(cast(Any, cfg), num_envs=5, backend_type="mujoco")

    assert result == "env"
    plan = captured["init_randomization_plan"]
    np.testing.assert_array_equal(plan.model_assignments, [0, 1, 2, 0, 1])
    sizes = [variant.geom_size_overrides[0].size for variant in plan.model_variants]
    assert sizes == [(0.028, 0.03, 0.0), (0.032, 0.035, 0.0), (0.036, 0.04, 0.0)]


def test_hora_grasp_collection_loop_uses_zero_actions_and_run_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = compose_collection_config(
        num_envs=2,
        target=3,
        output_path=tmp_path / "hora_smoke.npz",
        seed=11,
    )

    class FakeAdapter:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

        def build_task_env_cfg_override(self) -> dict[str, Any]:
            return {"seed": 11}

    class FakeEnv:
        action_space = SimpleNamespace(shape=(16,))

        def __init__(self) -> None:
            self.actions: list[np.ndarray] = []
            self.closed = False

        def init_state(self) -> None:
            return None

        def step(self, actions: np.ndarray) -> None:
            self.actions.append(actions.copy())
            if len(self.actions) == 2:
                raise RunComplete(
                    reason="hora_grasp_collection_target_reached",
                    summary={"saved_grasps": 3, "rejected_near_duplicates": 1},
                )

        def close(self) -> None:
            self.closed = True

    fake_env = FakeEnv()
    monkeypatch.setattr(generate_leap_hora_grasps, "BackendAdapter", FakeAdapter)
    monkeypatch.setattr(generate_leap_hora_grasps, "create_env", lambda *args, **kwargs: fake_env)

    summary = generate_leap_hora_grasps.run_collection(cfg, max_control_steps=10)

    assert summary["status"] == "complete"
    assert summary["saved_grasps"] == 3
    assert summary["control_steps"] == 2
    assert fake_env.closed
    assert len(fake_env.actions) == 2
    np.testing.assert_array_equal(fake_env.actions[0], np.zeros((2, 16), dtype=np.float32))


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
    reordered = loaded.take([2, 0, 0])
    assert reordered.size == 3
    np.testing.assert_array_equal(reordered.family_id, [1, 0, 0])
    np.testing.assert_array_equal(reordered.joint_pos[1], reordered.joint_pos[2])

    invalid = dict(cache.__dict__)
    invalid["split"] = np.asarray([0, 1, 2], dtype=np.int8)
    with pytest.raises(ValueError, match="family must belong to exactly one split"):
        HoraGraspCache(**invalid)


def test_hora_v5_merge_collapses_seed_families_and_preserves_metadata(tmp_path: Path) -> None:
    def cache(
        *, names: tuple[str, ...], family_id: list[int], joint_offsets: list[float]
    ) -> HoraGraspCache:
        count = len(family_id)
        quat = np.zeros((count, 4), dtype=np.float32)
        quat[:, 0] = 1.0
        joints = np.zeros((count, 16), dtype=np.float32)
        joints[:, 0] = joint_offsets
        return HoraGraspCache(
            joint_pos=joints,
            object_pos=np.zeros((count, 3), dtype=np.float32),
            object_quat=quat,
            cylinder_radius=np.full(count, 0.032, dtype=np.float32),
            cylinder_length=np.full(count, 0.070, dtype=np.float32),
            family_id=np.asarray(family_id, dtype=np.int16),
            split=np.zeros(count, dtype=np.int8),
            validation_trials=np.arange(3, 3 + count, dtype=np.int16),
            mass_min=np.full(count, 0.03, dtype=np.float32),
            mass_max=np.full(count, 0.08, dtype=np.float32),
            sliding_friction_min=np.full(count, 0.7, dtype=np.float32),
            sliding_friction_max=np.full(count, 1.4, dtype=np.float32),
            family_names=names,
        )

    general = cache(names=("general",), family_id=[0, 0], joint_offsets=[0.1, 0.2])
    middle = cache(
        names=("seed_00", "seed_01"),
        family_id=[0, 1, 1],
        joint_offsets=[0.2, 0.3, 0.4],
    )
    general_path = tmp_path / "general.npz"
    middle_path = tmp_path / "middle.npz"
    output_path = tmp_path / "v5.npz"
    general.save(general_path)
    middle.save(middle_path)

    result = build_leap_hora_v5_candidates.main(
        [
            "--general-cache",
            str(general_path),
            "--middle-cache",
            str(middle_path),
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    merged = HoraGraspCache.load(output_path)
    assert merged.family_names == ("general", "middle_opposition_expansion")
    assert merged.size == 4
    np.testing.assert_array_equal(merged.family_id, [0, 0, 1, 1])
    np.testing.assert_array_equal(merged.validation_trials, [3, 4, 4, 5])
    report = json.loads(output_path.with_suffix(".json").read_text())["merged"]
    assert report["rows"] == 4
    assert report["family_counts"] == {
        "general": 2,
        "middle_opposition_expansion": 2,
    }


def test_hora_cache_merge_rejects_family_split_conflicts() -> None:
    def singleton(name: str, split: int) -> HoraGraspCache:
        return HoraGraspCache(
            joint_pos=np.zeros((1, 16), dtype=np.float32),
            object_pos=np.zeros((1, 3), dtype=np.float32),
            object_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            cylinder_radius=np.asarray([0.032], dtype=np.float32),
            cylinder_length=np.asarray([0.070], dtype=np.float32),
            family_id=np.asarray([0], dtype=np.int16),
            split=np.asarray([split], dtype=np.int8),
            validation_trials=np.asarray([3], dtype=np.int16),
            mass_min=np.asarray([0.03], dtype=np.float32),
            mass_max=np.asarray([0.08], dtype=np.float32),
            sliding_friction_min=np.asarray([0.7], dtype=np.float32),
            sliding_friction_max=np.asarray([1.4], dtype=np.float32),
            family_names=(name,),
        )

    with pytest.raises(ValueError, match="span multiple splits"):
        merge_hora_grasp_caches(
            [singleton("train", 0), singleton("held", 2)],
            family_name_maps=[{"train": "shared"}, {"held": "shared"}],
        )


def test_hora_stage_a_sampling_balances_families_and_tiers() -> None:
    count = 12
    quat = np.zeros((count, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    cache = HoraGraspCache(
        joint_pos=np.zeros((count, 16), dtype=np.float32),
        object_pos=np.zeros((count, 3), dtype=np.float32),
        object_quat=quat,
        cylinder_radius=np.full(count, 0.032, dtype=np.float32),
        cylinder_length=np.full(count, 0.070, dtype=np.float32),
        family_id=np.repeat(np.asarray([0, 1], dtype=np.int16), 6),
        split=np.zeros(count, dtype=np.int8),
        validation_trials=np.full(count, 3, dtype=np.int16),
        mass_min=np.full(count, 0.03, dtype=np.float32),
        mass_max=np.full(count, 0.07, dtype=np.float32),
        sliding_friction_min=np.full(count, 0.7, dtype=np.float32),
        sliding_friction_max=np.full(count, 1.3, dtype=np.float32),
        family_names=("first", "second"),
    )
    recorder = object.__new__(LeapHoraPassiveQualificationRecorder)
    recorder._cache = cache
    recorder._eligible = np.arange(count, dtype=np.int64)
    recorder._eligible_mask = np.ones(count, dtype=np.bool_)
    recorder._training_cache_size = 10
    recorder._training_anchor_fraction = 0.8
    recorder._sampling_seed = 7
    tier = np.asarray([3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 2, 2], dtype=np.int8)

    selected = recorder._stage_a_indices(tier)

    assert selected.size == 10
    np.testing.assert_array_equal(np.bincount(cache.family_id[selected], minlength=2), [5, 5])
    assert np.count_nonzero(tier[selected] == 3) == 8
    assert np.count_nonzero(tier[selected] == 2) == 2


def test_hora_qualification_preserves_scores_when_stage_a_is_infeasible(
    tmp_path: Path,
) -> None:
    count = 2
    cache = HoraGraspCache(
        joint_pos=np.zeros((count, 16), dtype=np.float32),
        object_pos=np.zeros((count, 3), dtype=np.float32),
        object_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]] * count, dtype=np.float32),
        cylinder_radius=np.full(count, 0.032, dtype=np.float32),
        cylinder_length=np.full(count, 0.070, dtype=np.float32),
        family_id=np.zeros(count, dtype=np.int16),
        split=np.zeros(count, dtype=np.int8),
        validation_trials=np.full(count, 3, dtype=np.int16),
        mass_min=np.full(count, 0.03, dtype=np.float32),
        mass_max=np.full(count, 0.08, dtype=np.float32),
        sliding_friction_min=np.full(count, 0.7, dtype=np.float32),
        sliding_friction_max=np.full(count, 1.4, dtype=np.float32),
        family_names=("only_anchor",),
    )
    recorder = object.__new__(LeapHoraPassiveQualificationRecorder)
    recorder._cache = cache
    recorder._input_path = tmp_path / "input.npz"
    recorder._output_prefix = tmp_path / "qualified"
    recorder._eligible = np.arange(count, dtype=np.int64)
    recorder._eligible_mask = np.ones(count, dtype=np.bool_)
    recorder._completed = np.ones(count, dtype=np.bool_)
    recorder._survival_seconds = np.full(count, 20.0)
    recorder._valid_fraction = np.ones(count)
    recorder._thumb_fraction = np.ones(count)
    recorder._opposing_fraction = np.ones((count, 3))
    recorder._raw_thumb_fraction = np.ones(count)
    recorder._raw_opposing_fraction = np.ones((count, 3))
    recorder._palm_fraction = np.zeros(count)
    recorder._termination_causes = np.zeros((count, 3), dtype=np.bool_)
    recorder._anchor_seconds = 16.0
    recorder._recoverable_seconds = 1.0
    recorder._marginal_seconds = 0.5
    recorder._training_cache_size = 2
    recorder._training_anchor_fraction = 0.5
    recorder._training_middle_contact_fraction = 0.5
    recorder._minimum_raw_thumb_contact_fraction = 0.5
    recorder._minimum_raw_middle_contact_fraction = 0.5
    recorder._minimum_raw_opposing_contact_fraction = 0.5
    recorder._minimum_raw_opposing_fingers = 2
    recorder._sampling_seed = 7
    recorder._saved_complete = False

    summary = recorder._save(complete=True)

    assert summary["stage_a_cache"] is None
    assert "recoverable" in str(summary["stage_a_selection_error"])
    assert (tmp_path / "qualified_scores.npz").is_file()
    assert (tmp_path / "qualified_anchor.npz").is_file()
    assert recorder._saved_complete


def test_hora_grasp_completion_distribution_reports_every_axis() -> None:
    recorder = object.__new__(LeapHoraGraspRecorder)
    recorder._radius = [0.028, 0.028, 0.036]
    recorder._length = [0.060, 0.060, 0.080]
    recorder._family = [0, 1, 2]
    recorder._split = [0, 0, 2]
    recorder._reset = SimpleNamespace(family_names=("first", "second", "held"))

    summary = recorder._accepted_distribution()

    assert summary["geometry_counts"] == {"r=0.028,l=0.06": 2, "r=0.036,l=0.08": 1}
    assert summary["family_counts"] == {"first": 1, "second": 1, "held": 1}
    assert summary["split_counts"] == {"train": 2, "validation": 0, "held_out": 1}


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


def test_hora_reset_preserves_authoritative_cache_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unilab.tasks.manipulation.leap_inhand.manager_terms import LeapHandObjectReset

    reset = object.__new__(LeapHoraHandObjectReset)
    reset._repeat_candidate = np.zeros(2, dtype=np.bool_)
    reset._candidate_rows = np.zeros((2, 23), dtype=np.float64)
    reset._candidate_rows[:, 19] = 1.0
    reset._grasp_cache = None
    reset._grasp_cache_sampling = "random"
    reset._grasp_cache_cursor = 11
    reset.grasp_cache_index = np.full(2, -1, dtype=np.int64)

    def new_candidates(env: Any, ids: np.ndarray) -> None:
        del env
        reset.grasp_cache_index[ids] = np.asarray([17, 23], dtype=np.int64)

    def base_reset(self: Any, env: Any, ids: np.ndarray, **params: Any) -> None:
        del env, params
        # Reproduce the base reset's indexing of the temporary N-row cache.
        self.grasp_cache_index[ids] = np.arange(ids.size, dtype=np.int64)

    reset._new_candidates = new_candidates
    monkeypatch.setattr(LeapHandObjectReset, "__call__", base_reset)

    reset(SimpleNamespace(num_envs=2), np.asarray([0, 1], dtype=np.int32))

    np.testing.assert_array_equal(reset.grasp_cache_index, [17, 23])
    assert reset._grasp_cache is None
    assert reset._grasp_cache_sampling == "random"
    assert reset._grasp_cache_cursor == 11


def test_hora_cylinder_mass_and_inertia_follow_sampled_geometry() -> None:
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
        event_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=geometry)),
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
    transverse = (
        expected_mass
        * (3.0 * np.square(geometry.current_radius) + np.square(geometry.current_length))
        / 12.0
    )
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
    zero_latent = actor.get_latent(obs)
    assert zero_latent.shape == (batch, hardware_dim + 8)
    changed_properties = obs["actor"].clone()
    changed_properties[:, -len(PROPERTY_NAMES) :] += 100.0
    changed_obs = TensorDict({"actor": changed_properties}, batch_size=batch)
    torch.testing.assert_close(
        actor.get_latent(changed_obs)[:, -8:],
        torch.zeros(batch, 8),
    )
    zero_export = actor.as_onnx(verbose=False)
    first_export = zero_export(obs["actor"])
    second_export = zero_export(changed_properties)
    torch.testing.assert_close(first_export, second_export)
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


def test_hora_actor_can_initialize_at_exact_hold_command() -> None:
    batch = 3
    obs = TensorDict(
        {"actor": torch.randn(batch, 3 * 32 + len(PROPERTY_NAMES))},
        batch_size=batch,
    )
    actor = HoraActor(
        obs,
        {"actor": ["actor"]},
        "actor",
        16,
        hidden_dims=[32, 16],
        encoder_hidden_dims=[16],
        zero_mean_output_init=True,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": 0.03,
            "std_type": "scalar",
        },
    )

    torch.testing.assert_close(actor(obs), torch.zeros(batch, 16))
    linear_layers = [
        module for module in actor.mlp.modules() if isinstance(module, torch.nn.Linear)
    ]
    torch.testing.assert_close(linear_layers[-1].weight, torch.zeros_like(linear_layers[-1].weight))


def test_property_normalization_is_declared_ordered_and_bounded() -> None:
    ranges = {name: (-2.0, 2.0) for name in PROPERTY_NAMES}
    values = np.zeros((2, len(PROPERTY_NAMES)), dtype=np.float64)
    np.testing.assert_array_equal(normalize_object_properties(values, ranges), values)
    values[0, 0] = 2.1
    assert normalize_object_properties(values, ranges)[0, 0] == pytest.approx(1.05)
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        normalize_object_properties(values, ranges)


def test_frontier_metric_advances_zero_weight_stateful_tracker() -> None:
    from unilab.tasks.manipulation.leap_inhand.manager_terms import (
        LeapBidirectionalFrontierReward,
        frontier_state_metric,
    )

    class TrackingFrontier(LeapBidirectionalFrontierReward):
        def __call__(self, env, **params):
            del params
            self.calls += 1
            self.held_angle += 0.25 * env.step_dt
            self.frontier_angle[:] = np.maximum(self.frontier_angle, np.abs(self.held_angle))
            return np.zeros(env.num_envs, dtype=np.float32)

    tracker = object.__new__(TrackingFrontier)
    tracker.calls = 0
    tracker.held_angle = np.zeros(2, dtype=np.float32)
    tracker.frontier_angle = np.zeros(2, dtype=np.float32)
    tracker.frontier_progress = np.zeros(2, dtype=np.float32)
    tracker.frontier_speed = np.zeros(2, dtype=np.float32)
    tracker.frontier_progress_multiplier = np.ones(2, dtype=np.float32)
    tracker.directional_speed = np.zeros(2, dtype=np.float32)
    tracker.filtered_directional_speed = np.zeros(2, dtype=np.float32)
    tracker.fast_rotation_bonus = np.zeros(2, dtype=np.float32)
    tracker.committed_direction = np.zeros(2, dtype=np.int8)
    tracker.committed_time = np.zeros(2, dtype=np.float32)
    tracker.opposite_direction_speed = np.zeros(2, dtype=np.float32)
    tracker.reversal_penalty_event = np.zeros(2, dtype=np.bool_)
    tracker.stalled = np.zeros(2, dtype=np.bool_)
    tracker.milestone_bonus = np.zeros(2, dtype=np.float32)
    tracker.milestones_reached = np.zeros(2, dtype=np.int32)
    tracker.direction_reversals = np.zeros(2, dtype=np.int32)
    term_cfg = SimpleNamespace(func=tracker, weight=0.0)
    env = SimpleNamespace(
        num_envs=2,
        step_dt=0.05,
        reward_manager=SimpleNamespace(get_term_cfg=lambda name: term_cfg),
    )

    value = frontier_state_metric(cast(Any, env), "frontier", "held_angle")

    assert tracker.calls == 1
    np.testing.assert_allclose(value, 0.0125)
    np.testing.assert_allclose(tracker.frontier_angle, 0.0125)


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
    assert "interactive.action_mode=policy" in overrides
    assert (
        "env.events.reset_hand_object.params.grasp_cache_path="
        "caches/hora/leap_precision_grasps_v2.npz"
    ) in overrides
    assert any(
        item.startswith("++env.events.reset_hand_object.params.cylinder_geometry_override=")
        for item in overrides
    )
    assert any("density_range" in item for item in overrides)
    assert any(item.startswith("++env.recorders.evaluation.func=") for item in overrides)

    zero_overrides = build_case_overrides(
        case,
        tmp_path / "passive.json",
        policy_family="hora",
        action_mode="zero",
    )
    assert "interactive.action_mode=zero" in zero_overrides
    assert "env.actions.hand.latency_steps_range=[0,0]" in zero_overrides
    assert "env.actions.hand.action_noise_std_schedule=[{step:0,std:0}]" in zero_overrides
    assert "env.observations.actor.terms.hardware.params.joint_noise=0" in zero_overrides

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


def test_hora_checkpoint_sweep_parsing_and_stage_ranking(tmp_path: Path) -> None:
    assert sweep_leap_hora_checkpoints.parse_checkpoint_spec("25:75:25,99") == [25, 50, 75, 99]
    with pytest.raises(ValueError, match="invalid checkpoint range"):
        sweep_leap_hora_checkpoints.parse_checkpoint_spec("75:25:25")

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

    def report(*, middle: float, valid: float, net: float) -> dict[str, Any]:
        rows = []
        for split in ("in_distribution", "edge_of_range", "out_of_distribution"):
            row = {name: 0.0 for name in metric_names}
            row.update(
                {
                    "split": split,
                    "time_to_fall_seconds": 20.0,
                    "valid_grasp_fraction": valid,
                    "middle_contact_fraction": middle,
                    "thumb_contact_fraction": 1.0,
                    "index_contact_fraction": 1.0,
                    "outer_contact_fraction": 1.0,
                    "net_committed_axis_rotation": net,
                    "directional_speed": net / 20.0,
                    "direction_reversals": 2.0,
                }
            )
            rows.append(row)
        return summarize_episode_reports(rows)

    ranked = rank_hora_checkpoint_evaluations(
        [
            (100, tmp_path / "100.json", report(middle=0.1, valid=0.95, net=0.0)),
            (200, tmp_path / "200.json", report(middle=0.7, valid=0.85, net=0.1)),
        ],
        stage="stabilize",
    )
    assert ranked["recommended_checkpoint"] == 200
    assert ranked["recommended_passed"] is True
    assert ranked["rankings"][1]["gate_failures"] == ["middle_contact_fraction=0.1 < 0.4"]

    rotating = rank_hora_checkpoint_evaluations(
        [
            (100, tmp_path / "100.json", report(middle=0.7, valid=0.9, net=0.1)),
            (200, tmp_path / "200.json", report(middle=0.7, valid=0.8, net=1.0)),
        ],
        stage="rotate",
    )
    assert rotating["recommended_checkpoint"] == 200


def test_hora_rollout_audit_overrides_and_summary(tmp_path: Path) -> None:
    case = load_evaluation_manifest(_MANIFEST)["cases"][0]
    output = tmp_path / "audit.json"
    overrides = audit_leap_hora_rollout.build_audit_overrides(case, output)
    assert any("LeapHoraRolloutAuditRecorder" in item for item in overrides)
    assert f"++env.recorders.hora_rollout_audit.params.output_path={output}" in overrides

    finger_row = {
        "initial_raw_contact": True,
        "raw_contact_fraction": 0.5,
        "graced_contact_fraction": 0.75,
        "policy_action_rms": 0.1,
        "target_step_rms_rad": 0.01,
        "target_travel_rad": 0.2,
        "target_span_rad": 0.05,
        "measured_step_rms_rad": 0.008,
        "measured_travel_rad": 0.15,
        "measured_span_rad": 0.04,
        "tracking_error_rms_rad": 0.02,
        "estimated_torque_rms": 0.03,
        "estimated_torque_saturation_fraction": 0.0,
        "joint_limit_occupancy": 0.0,
        "raw_contact_transitions": 2,
    }
    reports = [
        {"fingers": {name: dict(finger_row) for name in ("index", "middle", "outer", "thumb")}},
        {"fingers": {name: dict(finger_row) for name in ("index", "middle", "outer", "thumb")}},
    ]
    summary = summarize_hora_rollout_audits(reports)
    assert summary["episodes"] == 2
    assert summary["fingers"]["middle"]["episodes_with_initial_raw_contact"] == 2
    assert summary["fingers"]["middle"]["metrics"]["raw_contact_fraction"]["median"] == 0.5


@pytest.mark.slow
def test_hora_mujoco_compiles_geometry_variants() -> None:
    mujoco = pytest.importorskip("mujoco")
    from unilab.base import registry
    from unilab.training import ensure_registries

    cfg = _ppo_cfg("leap_inhand_hora_grasp")
    ensure_registries()
    env = registry.make(
        "LeapInhandHoraGrasp",
        num_envs=3,
        sim_backend="mujoco",
        env_cfg_override=BackendAdapter(
            cfg, root_dir=_ROOT, algo_name="ppo"
        ).build_task_env_cfg_override(),
    )
    try:
        sizes = []
        for env_index in range(3):
            model = env.get_playback_model(env_index)
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_visual")
            sizes.append(model.geom_size[geom_id, :2])
        np.testing.assert_allclose(sizes, [[0.028, 0.030], [0.032, 0.035], [0.036, 0.040]])
    finally:
        env.close()


@pytest.mark.slow
def test_hora_mujoco_observation_action_and_ppo_contract() -> None:
    pytest.importorskip("mujoco")
    from rsl_rl.runners import OnPolicyRunner
    from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg

    from unilab.base import registry
    from unilab.scripts.train_rsl_rl import build_ppo_env_cfg_override
    from unilab.tasks.manipulation.leap_inhand.hora import LeapHoraPositionAction
    from unilab.tasks.manipulation.leap_inhand.manager_terms import (
        LeapBidirectionalFrontierReward,
    )
    from unilab.training import algo_config_dict, ensure_registries

    cfg = _ppo_cfg("leap_inhand_hora_stage1")
    cfg.algo.num_envs = 2
    cfg.env.actions.hand.latency_steps_range = [1, 1]
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
        np.testing.assert_array_equal(action.latency_steps, 1)
        np.testing.assert_allclose(action.latency_seconds, 0.05)
        np.testing.assert_allclose(state.obs["obs"][:, -1], 1.0, atol=1.0e-6)
        np.testing.assert_array_equal(action.gait_reference, 0.0)

        initial_target = action.policy_target.copy()
        command = np.full((2, 16), 0.12, dtype=np.float32)
        action.process_actions(command)
        np.testing.assert_allclose(action.policy_target, initial_target)
        action.process_actions(np.zeros_like(command))
        expected_target = np.clip(
            initial_target + 0.12 / 24.0,
            action.ctrl_lower,
            action.ctrl_upper,
        )
        np.testing.assert_allclose(action.policy_target, expected_target, atol=1.0e-7)
        action.reset(np.asarray([0], dtype=np.intp))
        np.testing.assert_array_equal(action._action_history[0], 0.0)
        assert action.latency_steps[0] == 1

        state = env.step(np.zeros((2, 16), dtype=np.float32))
        assert np.isfinite(state.obs["obs"]).all()
        frontier_cfg = env.reward_manager.get_term_cfg("frontier")
        assert frontier_cfg.weight == 0.0
        assert isinstance(frontier_cfg.func, LeapBidirectionalFrontierReward)
        assert frontier_cfg.func._last_counter == env.common_step_counter
        assert "mean_axial_rotation_velocity" in env.metrics_manager.active_terms

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
