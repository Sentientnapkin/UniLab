"""Tests for env config completeness and env instantiation.

The whole module is marked slow: Hydra composition and ``registry.make()``
over every owner are CPU-bound and too slow for the single-core CI runner.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unilab.base.registry import ensure_registries

# CPU-bound on the single-core CI runner; kept in the slow lane (make test-slow).
pytestmark = pytest.mark.slow


def _require_mujoco_runtime() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco_uni.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco_uni.batch_env not available (platform/libstdc++ issue)")


def _allegro_manager_override(
    backend: str = "mujoco",
    *,
    config_root: str = "ppo",
    task: str = "allegro_inhand",
) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir

    from unilab.base.config_adapter import BackendAdapter

    repo_root = Path(__file__).parents[2]
    with initialize_config_dir(
        config_dir=str(repo_root / "src" / "unilab" / "conf" / config_root), version_base="1.3"
    ):
        cfg = compose("config", overrides=[f"task={task}/{backend}"])
    return BackendAdapter(
        cfg, root_dir=repo_root, algo_name=config_root
    ).build_task_env_cfg_override()


def _g1_manager_override(
    task: str = "g1_walk_flat", backend: str = "mujoco", config_group: str = "ppo"
) -> dict[str, Any]:
    from hydra import compose, initialize_config_dir

    from unilab.base.config_adapter import BackendAdapter

    repo_root = Path(__file__).parents[2]
    if task == "g1_walk_rough":
        # There is no ppo g1_walk_rough owner; use the SAC owner instead.
        with initialize_config_dir(
            config_dir=str(repo_root / "src" / "unilab" / "conf" / "sac"), version_base="1.3"
        ):
            cfg = compose("config", overrides=[f"task={task}/mujoco"])
        return BackendAdapter(
            cfg, root_dir=repo_root, algo_name="sac"
        ).build_task_env_cfg_override()
    with initialize_config_dir(
        config_dir=str(repo_root / "src" / "unilab" / "conf" / config_group), version_base="1.3"
    ):
        cfg = compose("config", overrides=[f"task={task}/{backend}"])
    return BackendAdapter(
        cfg, root_dir=repo_root, algo_name=config_group
    ).build_task_env_cfg_override()


def _motion_manager_override(
    task: str,
    backend: str,
    *,
    config_root: str = "ppo",
) -> tuple[str, dict[str, Any]]:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from unilab.base.config_adapter import BackendAdapter

    repo_root = Path(__file__).parents[2]
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(repo_root / "src" / "unilab" / "conf" / config_root), version_base="1.3"
    ):
        overrides = [f"task={task}/{backend}"]
        cfg = compose("config", overrides=overrides)
    return str(cfg.training.task_name), BackendAdapter(
        cfg,
        root_dir=repo_root,
        algo_name=config_root,
    ).build_task_env_cfg_override()


# ---------------------------------------------------------------------------
# Config attribute completeness (no env.step(), no MuJoCo sim)
# ---------------------------------------------------------------------------


def test_registry_bootstrap_and_config_imports_do_not_require_mujoco():
    repo_root = Path(__file__).parents[2]
    script = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "mujoco" or name.startswith("mujoco."):
                raise ImportError("mujoco blocked by test")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = blocked_import

        from unilab.base import registry
        from unilab.base.backend_factory import create_backend
        from unilab.base.registry import ensure_registries

        ensure_registries()
        assert callable(create_backend)
        assert registry.contains("G1MotionTracking")
        assert registry.contains("G1MotionTrackingDeploy")
        assert registry.contains("X2WallFlipTracking")
        assert registry.contains("AllegroInhandRotation")
        metadata = registry.list_registered_envs()
        assert metadata["G1MotionTracking"]["config_factory"] == "ManagerBasedRlEnvCfg"
        assert metadata["X2WallFlipTracking"]["config_factory"] == "ManagerBasedRlEnvCfg"
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_g1_walk_tasks_register_to_manager_based_env():
    from unilab.base import registry

    ensure_registries()
    metadata = registry.list_registered_envs()
    assert metadata["G1WalkFlat"]["config_factory"] == "ManagerBasedRlEnvCfg"
    assert metadata["G1WalkFlat"]["available_backends"] == [
        "mujoco",
        "mjwarp",
        "motrix",
        "isaacgym",
        "genesis",
        "isaacsim",
        "newton",
    ]
    assert metadata["G1WalkRough"]["available_backends"] == ["mujoco", "motrix"]


def test_g1_walk_flat_isaacgym_owner_composes_and_materializes():
    """The isaacgym owner composes and materializes a plain manager env cfg."""
    from unilab.base import registry
    from unilab.base.config_materialization import apply_cfg_overrides
    from unilab.envs import ManagerBasedRlEnvCfg

    ensure_registries()
    override = _g1_manager_override("g1_walk_flat", backend="isaacgym")
    env_cfg = registry.materialize_env_config("G1WalkFlat")
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()

    assert env_cfg.scene is not None
    # The subprocess backend consumes the self-contained MJCF scene directly;
    # scene fragments and generated terrain stay unset.
    assert env_cfg.scene.model_file.endswith("robots/g1/scene_flat.xml")
    assert env_cfg.scene.fragment_files == []
    assert env_cfg.scene.terrain is None
    assert env_cfg.scene.default_keyframe_name == "stand"
    assert env_cfg.isaacgym_device_id == 0
    # Effort-mode dofs carry no PD gains, so the owner disables kp/kd
    # randomization like the mjwarp/motrix owners.
    assert env_cfg.events["pd_gains"] is None


@pytest.mark.parametrize("config_group", ("ppo", "sac"))
def test_g1_walk_flat_genesis_owner_composes_and_materializes(config_group):
    """The genesis owners compose and materialize a plain manager env cfg."""
    from unilab.base import registry
    from unilab.base.config_materialization import apply_cfg_overrides
    from unilab.envs import ManagerBasedRlEnvCfg

    ensure_registries()
    override = _g1_manager_override("g1_walk_flat", backend="genesis", config_group=config_group)
    env_cfg = registry.materialize_env_config("G1WalkFlat")
    assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()

    assert env_cfg.scene is not None
    # The in-process backend consumes the self-contained MJCF scene directly;
    # scene fragments and generated terrain stay unset.
    assert env_cfg.scene.model_file.endswith("robots/g1/scene_flat.xml")
    assert env_cfg.scene.fragment_files == []
    assert env_cfg.scene.terrain is None
    assert env_cfg.scene.default_keyframe_name == "stand"
    assert env_cfg.genesis_device_id == 0
    # The owner re-declares the MJCF <option integrator="implicitfast"> that
    # Genesis drops at import; the other global options stay at Genesis
    # defaults (None).
    assert env_cfg.genesis_integrator == "implicitfast"
    assert env_cfg.genesis_constraint_solver is None
    assert env_cfg.genesis_friction_cone is None
    assert env_cfg.genesis_solver_iterations is None
    # kp/kd reset randomization stays enabled: the backend declares the
    # measured RESET_TERM_KP/KD DR terms.
    assert env_cfg.events["pd_gains"] is not None


def test_g1_walk_flat_assets_define_contact_sensors_for_gait_rewards():
    repo_root = Path(__file__).parents[2]
    scene_text = (
        repo_root / "src" / "unilab" / "assets" / "robots" / "g1" / "scene_flat.xml"
    ).read_text()
    model_text = (repo_root / "src" / "unilab" / "assets" / "robots" / "g1" / "g1.xml").read_text()

    for name in (
        "left_foot_contact_0",
        "left_foot_contact_1",
        "left_foot_contact_2",
        "left_foot_contact_3",
        "right_foot_contact_0",
        "right_foot_contact_1",
        "right_foot_contact_2",
        "right_foot_contact_3",
    ):
        assert name in scene_text

    for name in (
        "pelvis_local_linvel",
        "pelvis_gyro",
        "pelvis_acceleration",
        "pelvis_upvector",
        "torso_gyro",
        "torso_acceleration",
        "torso_upvector",
    ):
        assert name in model_text

    for name in (
        "left_foot_contact_0_geom",
        "left_foot_contact_1_geom",
        "left_foot_contact_2_geom",
        "left_foot_contact_3_geom",
        "right_foot_contact_0_geom",
        "right_foot_contact_1_geom",
        "right_foot_contact_2_geom",
        "right_foot_contact_3_geom",
    ):
        assert name in model_text


def test_g1_sphere_hand_assets_align_with_current_g1_sensor_names():
    repo_root = Path(__file__).parents[2]
    model_text = (
        repo_root / "src" / "unilab" / "assets" / "robots" / "g1" / "g1_sphere_hand.xml"
    ).read_text()

    for name in (
        "pelvis_local_linvel",
        "pelvis_gyro",
        "pelvis_acceleration",
        "pelvis_upvector",
        "torso_gyro",
        "torso_acceleration",
        "torso_upvector",
        "left_foot_quat",
        "right_foot_quat",
    ):
        assert name in model_text


def test_g1_box_tracking_scene_compiles_with_pelvis_imu_sensor_names():
    mujoco = pytest.importorskip("mujoco")

    repo_root = Path(__file__).parents[2]
    scene_xml = (
        repo_root / "src" / "unilab" / "assets" / "robots" / "g1" / "scene_flat_with_largebox.xml"
    )

    model = mujoco.MjModel.from_xml_path(str(scene_xml))

    for sensor_name in (
        "pelvis_local_linvel",
        "pelvis_gyro",
        "pelvis_acceleration",
        "pelvis_upvector",
        "torso_gyro",
        "torso_acceleration",
        "torso_upvector",
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name) >= 0


def test_g1_box_tracking_scene_uses_sphere_hand_and_box_tracking_mesh():
    repo_root = Path(__file__).parents[2]
    scene_text = (
        repo_root / "src" / "unilab" / "assets" / "robots" / "g1" / "scene_flat_with_largebox.xml"
    ).read_text()

    for snippet in (
        '<include file="g1_sphere_hand.xml"/>',
        'mesh name="largebox_mesh" file="box_tracking/largebox.obj"',
        '<freejoint name="largebox_joint"/>',
        '<geom name="largebox" type="mesh" mesh="largebox_mesh"',
        "0.0 0.5 0.85",
    ):
        assert snippet in scene_text

    for name in (
        "left_foot_contact_0",
        "left_foot_contact_1",
        "left_foot_contact_2",
        "left_foot_contact_3",
        "right_foot_contact_0",
        "right_foot_contact_1",
        "right_foot_contact_2",
        "right_foot_contact_3",
    ):
        assert name in scene_text


def test_inhand_rotation_and_grasp_registries_are_manager_only():
    from unilab.base import registry
    from unilab.base.config_materialization import apply_cfg_overrides
    from unilab.envs import ManagerBasedRlEnvCfg

    ensure_registries()
    metadata = registry.list_registered_envs()
    assert metadata["AllegroInhandRotation"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco", "motrix", "drake"],
    }
    assert metadata["AllegroInhandRotationGrasp"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco", "motrix"],
    }
    assert metadata["LeapInhandRotationGrasp"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco"],
    }

    cfg = registry.materialize_env_config("AllegroInhandRotation")
    assert isinstance(cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(cfg, _allegro_manager_override())
    assert cfg.policy_observation_group == "policy"
    assert cfg.critic_observation_group is None
    assert cfg.observations["policy"].history_length == 3
    assert list(cfg.actions) == ["hand"]
    assert list(cfg.terminations) == ["dropped", "time_out"]
    assert list(cfg.rewards) == [
        "rotate",
        "obj_linvel",
        "pose_diff",
        "torque",
        "work",
        "drop",
    ]
    assert not hasattr(cfg, "reward_config")

    grasp_cfg = registry.materialize_env_config("AllegroInhandRotationGrasp")
    assert isinstance(grasp_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(
        grasp_cfg,
        _allegro_manager_override(task="allegro_inhand_grasp"),
    )
    assert grasp_cfg.observations["policy"].history_length == 3
    assert grasp_cfg.actions["hand"].action_scale == 0.0
    assert list(grasp_cfg.terminations) == ["dropped", "time_out", "invalid_grasp"]
    assert list(grasp_cfg.metrics) == [
        "fingertips_close",
        "enough_contacts",
        "ball_held",
        "valid",
    ]
    assert list(grasp_cfg.recorders) == ["grasp_cache"]

    leap_grasp_cfg = registry.materialize_env_config("LeapInhandRotationGrasp")
    assert isinstance(leap_grasp_cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(
        leap_grasp_cfg,
        _allegro_manager_override(task="leap_inhand_grasp"),
    )
    assert leap_grasp_cfg.actions["hand"].action_scale == 0.0
    assert list(leap_grasp_cfg.terminations) == ["dropped", "time_out", "invalid_grasp"]
    assert list(leap_grasp_cfg.recorders) == ["grasp_cache"]


def test_allegro_manager_configured_missing_grasp_cache_fails_closed(tmp_path: Path):
    from unilab.managers import EventTermCfg
    from unilab.tasks.manipulation.allegro_inhand.manager_terms import AllegroHandBallReset

    entity = SimpleNamespace(
        num_joints=16,
        data=SimpleNamespace(
            default_root_state=np.zeros((2, 13), dtype=np.float32),
            actuator_ctrl_range=np.tile([-1.0, 1.0], (16, 1)),
        ),
    )
    env = SimpleNamespace(num_envs=2, scene={"robot": entity})
    cfg = EventTermCfg(
        func=AllegroHandBallReset,
        mode="reset",
        params={
            "entity_name": "robot",
            "grasp_cache_path": str(tmp_path / "missing.npy"),
            "joint_noise": 0.0,
            "ball_velocity_noise": 0.0,
            "ball_z_offset": 0.0,
        },
    )

    with pytest.raises(FileNotFoundError, match="configured grasp cache does not exist"):
        AllegroHandBallReset(cfg, cast(Any, env))


def test_leap_reset_randomization_is_bounded_and_preserves_fixed_starts():
    from unilab.managers import EventTermCfg
    from unilab.tasks.manipulation.leap_inhand.manager_terms import LeapHandObjectReset

    num_envs = 8
    written: dict[str, np.ndarray] = {}
    default_root_state = np.zeros((num_envs, 13), dtype=np.float32)
    default_root_state[:, 2] = 0.2
    default_root_state[:, 3] = 1.0
    default_joint_pos = np.zeros((num_envs, 16), dtype=np.float32)
    entity = SimpleNamespace(
        num_joints=16,
        data=SimpleNamespace(
            default_root_state=default_root_state,
            default_joint_pos=default_joint_pos,
            actuator_ctrl_range=np.tile([-1.0, 1.0], (16, 1)),
        ),
        write_joint_state_to_sim=lambda pos, vel, env_ids: written.update(
            joint_pos=pos.copy(), joint_vel=vel.copy()
        ),
        write_root_link_pose_to_sim=lambda pose, env_ids: written.update(root_pose=pose.copy()),
        write_root_link_velocity_to_sim=lambda velocity, env_ids: written.update(
            root_velocity=velocity.copy()
        ),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        rng=np.random.default_rng(7),
        scene={"robot": entity},
    )
    params = {
        "entity_name": "robot",
        "grasp_cache_path": None,
        "joint_noise": 0.02,
        "object_velocity_noise": 0.015,
        "object_z_offset": 0.0,
        "randomization_fraction": 1.0,
        "object_xy_noise": 0.002,
        "object_z_noise": 0.001,
        "object_tilt_noise_deg": 4.0,
        "object_angular_velocity_noise": 0.15,
    }
    cfg = EventTermCfg(func=LeapHandObjectReset, mode="reset", params=params)
    reset = LeapHandObjectReset(cfg, cast(Any, env))
    ids = np.arange(num_envs, dtype=np.int32)
    reset(cast(Any, env), ids, **params)

    assert np.all(reset.randomized)
    assert np.max(np.abs(written["joint_pos"])) <= 0.02
    assert np.max(np.abs(written["root_pose"][:, :2])) <= 0.002
    assert np.max(np.abs(written["root_pose"][:, 2] - 0.2)) <= 0.001
    assert np.max(reset.tilt_offset_deg) <= 4.0
    assert np.max(np.abs(written["root_velocity"][:, :3])) <= 0.015
    assert np.max(np.abs(written["root_velocity"][:, 3:])) <= 0.15

    fixed_params = dict(params, randomization_fraction=0.0)
    reset(cast(Any, env), ids, **fixed_params)
    assert not np.any(reset.randomized)
    np.testing.assert_array_equal(written["joint_pos"], default_joint_pos)
    np.testing.assert_array_equal(written["root_pose"], default_root_state[:, :7])
    np.testing.assert_array_equal(written["root_velocity"], default_root_state[:, 7:])


def test_leap_grasp_quality_waits_for_settling_and_rejects_motion():
    from unilab.managers import TerminationTermCfg
    from unilab.tasks.manipulation.leap_inhand.grasp_gen import (
        LeapGraspQualityTermination,
    )
    from unilab.tasks.manipulation.leap_inhand.manager_terms import LeapGraspTermination

    grasp = object.__new__(LeapGraspTermination)
    grasp._last_counter = 5
    grasp.valid_grasp = np.ones(3, dtype=np.bool_)
    grasp.observation = SimpleNamespace(
        object_linvel=np.array(
            [[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
        ),
        object_angvel=np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32
        ),
    )
    env = SimpleNamespace(
        num_envs=3,
        common_step_counter=5,
        step_dt=0.05,
        episode_length_buf=np.array([5, 10, 10]),
        termination_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=grasp)),
    )
    cfg = TerminationTermCfg(
        func=LeapGraspQualityTermination,
        params={
            "state_term_name": "dropped",
            "minimum_validation_seconds": 0.5,
            "maximum_linear_speed": 0.02,
            "maximum_angular_speed": 0.2,
            "enabled": True,
        },
    )

    quality = LeapGraspQualityTermination(cfg, cast(Any, env))

    np.testing.assert_array_equal(quality(cast(Any, env)), [False, True, True])
    np.testing.assert_array_equal(quality.validation_started, [False, True, True])
    np.testing.assert_array_equal(quality.stable, [True, False, False])
    np.testing.assert_allclose(quality.linear_speed, [0.0, 0.03, 0.0])
    np.testing.assert_allclose(quality.angular_speed, [0.0, 0.0, 0.3])


def test_leap_grasp_recorder_seeds_cache_and_rejects_near_duplicates(tmp_path: Path) -> None:
    from unilab.base.run_control import RunComplete
    from unilab.managers import RecorderTermCfg
    from unilab.tasks.manipulation.leap_inhand.grasp_gen import (
        LeapGraspQualityTermination,
        LeapGraspRecorder,
    )

    seed = np.zeros((1, 23), dtype=np.float32)
    seed[0, 16:19] = (0.0, 0.0, 0.25)
    seed[0, 19] = 1.0
    seed_path = tmp_path / "seed.npy"
    output_path = tmp_path / "diverse.npy"
    np.save(seed_path, seed)

    rows = np.repeat(seed, 4, axis=0)
    rows[0, :16] = 0.001  # near in every enabled criterion: reject
    rows[1, :16] = 0.05  # novel joint posture
    rows[2, 16] = 0.005  # novel object position
    angle = np.deg2rad(10.0)
    rows[3, 19:23] = (np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0)

    grasp = SimpleNamespace(
        observation=SimpleNamespace(
            dof_pos=rows[:, :16],
            object_pos=rows[:, 16:19],
            object_quat=rows[:, 19:23],
        )
    )
    quality = object.__new__(LeapGraspQualityTermination)
    quality._grasp = grasp
    quality.stable = np.ones(4, dtype=np.bool_)
    quality._last_counter = 7
    env = SimpleNamespace(
        num_envs=4,
        common_step_counter=7,
        reset_time_outs=np.ones(4, dtype=np.bool_),
        reset_terminated=np.zeros(4, dtype=np.bool_),
        extras={"log": {}},
        termination_manager=SimpleNamespace(
            get_term_cfg=lambda name: SimpleNamespace(func=quality)
        ),
    )
    cfg = RecorderTermCfg(
        func=LeapGraspRecorder,
        params={
            "quality_term_name": "invalid_grasp",
            "initial_cache_path": str(seed_path),
            "output_path": str(output_path),
            "collection_target": 4,
            "auto_save": True,
            "minimum_joint_rms_rad": 0.01,
            "minimum_position_distance_m": 0.001,
            "minimum_axis_angle_deg": 2.0,
        },
    )
    recorder = LeapGraspRecorder(cfg, cast(Any, env))

    with pytest.raises(RunComplete) as caught:
        recorder.record_pre_reset(np.arange(4, dtype=np.int32))

    saved = np.load(output_path)
    np.testing.assert_array_equal(saved[0], seed[0])
    np.testing.assert_array_equal(saved[1:], rows[1:])
    assert recorder.total_saved_grasps == 4
    assert caught.value.summary["seed_grasps"] == 1
    assert caught.value.summary["generated_grasps"] == 3
    assert caught.value.summary["rejected_near_duplicates"] == 1
    assert env.extras["log"]["grasp/rejected_near_duplicates"] == 1.0


def _allegro_grasp_term_fixture() -> tuple[Any, Any, np.ndarray]:
    from unilab.managers import TerminationTermCfg
    from unilab.tasks.manipulation.allegro_inhand.grasp_gen import (
        AllegroGraspQualityTermination,
    )
    from unilab.tasks.manipulation.allegro_inhand.manager_terms import (
        AllegroRotationObservation,
    )

    num_envs = 3
    states = np.arange(num_envs * 23, dtype=np.float32).reshape(num_envs, 23)
    observation = object.__new__(AllegroRotationObservation)
    observation.dof_pos = states[:, :16]
    observation.ball_pos = np.array(
        [[0.0, 0.0, 0.2], [0.0, 0.0, 0.2], [0.0, 0.0, 0.1]], dtype=np.float32
    )
    observation.ball_quat = states[:, 19:23]
    observation._last_counter = 1

    body_pos = np.repeat(observation.ball_pos[:, None, :], 4, axis=1)
    body_pos[:, :, 0] += 0.05
    body_pos[1, 0, 0] += 0.2
    contacts = np.array(
        [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    entity = SimpleNamespace(
        data=SimpleNamespace(body_link_pos_w=body_pos),
        find_bodies=lambda names, preserve_order: (list(range(4)), list(names)),
    )

    class _Scene(dict):
        def bind_sensor_data(self, names):
            assert tuple(names) == ("ff_contact", "mf_contact", "rf_contact", "th_contact")
            return SimpleNamespace(dimensions=(1, 1, 1, 1), read=lambda: contacts)

    env = SimpleNamespace(
        num_envs=num_envs,
        common_step_counter=1,
        scene=_Scene(robot=entity),
        observation_manager=SimpleNamespace(
            get_term_cfg=lambda group, name: SimpleNamespace(func=observation)
        ),
        action_manager=SimpleNamespace(),
        reset_time_outs=np.ones(num_envs, dtype=np.bool_),
        reset_terminated=np.zeros(num_envs, dtype=np.bool_),
        extras={"log": {}},
    )
    cfg = TerminationTermCfg(
        func=AllegroGraspQualityTermination,
        params={
            "entity_name": "robot",
            "observation_group": "policy",
            "observation_term": "rotation",
            "fingertip_body_names": ["ff_tip", "mf_tip", "rf_tip", "th_tip"],
            "contact_sensor_names": [
                "ff_contact",
                "mf_contact",
                "rf_contact",
                "th_contact",
            ],
            "max_fingertip_distance": 0.1,
            "minimum_contacts": 2,
            "minimum_ball_height": 0.125,
            "enabled": True,
        },
    )
    term = AllegroGraspQualityTermination(cfg, cast(Any, env))
    env.termination_manager = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=term))
    return env, term, states


def test_allegro_grasp_quality_term_matches_legacy_conditions():
    from unilab.managers import MetricsTermCfg
    from unilab.tasks.manipulation.allegro_inhand.grasp_gen import AllegroGraspQualityMetric

    env, term, _ = _allegro_grasp_term_fixture()

    np.testing.assert_array_equal(term(cast(Any, env)), [False, True, True])
    np.testing.assert_array_equal(term.fingertips_close, [True, False, True])
    np.testing.assert_array_equal(term.enough_contacts, [True, True, False])
    np.testing.assert_array_equal(term.ball_held, [True, True, False])
    metric = AllegroGraspQualityMetric(
        MetricsTermCfg(
            func=AllegroGraspQualityMetric,
            params={"quality_term_name": "invalid_grasp", "condition": "valid"},
        ),
        cast(Any, env),
    )
    np.testing.assert_array_equal(metric(cast(Any, env)), [1.0, 0.0, 0.0])


def test_allegro_grasp_recorder_saves_target_and_raises_run_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unilab.base.run_control import RunComplete
    from unilab.managers import RecorderTermCfg
    from unilab.tasks.manipulation.allegro_inhand import grasp_gen

    env, term, states = _allegro_grasp_term_fixture()
    term(cast(Any, env))
    cache_path = tmp_path / "allegro.npy"
    cfg = RecorderTermCfg(
        func=grasp_gen.AllegroGraspRecorder,
        params={
            "quality_term_name": "invalid_grasp",
            "output_path": str(cache_path),
            "collection_target": 2,
            "auto_save": True,
        },
    )
    recorder = grasp_gen.AllegroGraspRecorder(cfg, cast(Any, env))
    save_calls: list[Path] = []
    real_save = grasp_gen.np.save

    def save_once(path: str | Path, values: np.ndarray) -> None:
        save_calls.append(Path(path))
        real_save(path, values)

    monkeypatch.setattr(grasp_gen.np, "save", save_once)
    with pytest.raises(RunComplete) as caught:
        recorder.record_pre_reset(np.arange(3, dtype=np.int32))

    expected = np.concatenate(
        (states[:, :16], term.observation.ball_pos, states[:, 19:23]), axis=1, dtype=np.float32
    )
    np.testing.assert_array_equal(np.load(cache_path), expected[:2])
    assert save_calls == [cache_path]
    assert env.extras["log"] == {
        "grasp_cache/saved": 1.0,
        "grasp_cache/num_states": 2.0,
        "grasp/target_reached": 1.0,
    }
    assert dict(caught.value.summary) == {
        "collected_grasps": 3,
        "saved_grasps": 2,
        "grasp_collection_target": 2,
    }

    recorder.close()
    assert save_calls == [cache_path]


def test_allegro_grasp_recorder_close_autosaves_and_io_failure_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unilab.managers import RecorderTermCfg
    from unilab.tasks.manipulation.allegro_inhand import grasp_gen

    env, term, _ = _allegro_grasp_term_fixture()
    term(cast(Any, env))
    cache_path = tmp_path / "allegro.npy"
    cfg = RecorderTermCfg(
        func=grasp_gen.AllegroGraspRecorder,
        params={
            "quality_term_name": "invalid_grasp",
            "output_path": str(cache_path),
            "collection_target": 3,
            "auto_save": True,
        },
    )
    recorder = grasp_gen.AllegroGraspRecorder(cfg, cast(Any, env))
    env.reset_terminated[1] = True
    recorder.record_pre_reset(np.array([0, 1], dtype=np.int32))
    assert recorder.total_saved_grasps == 1
    assert not cache_path.exists()
    recorder.close()
    assert np.load(cache_path).shape == (1, 23)

    failed_path = tmp_path / "failed.npy"
    failed_cfg = RecorderTermCfg(
        func=grasp_gen.AllegroGraspRecorder,
        params={
            "quality_term_name": "invalid_grasp",
            "output_path": str(failed_path),
            "collection_target": 1,
            "auto_save": True,
        },
    )
    failed = grasp_gen.AllegroGraspRecorder(failed_cfg, cast(Any, env))
    sentinel = OSError("disk full")
    monkeypatch.setattr(
        grasp_gen.np, "save", lambda *_args, **_kwargs: (_ for _ in ()).throw(sentinel)
    )
    with pytest.raises(OSError) as caught:
        failed.record_pre_reset(np.array([0], dtype=np.int32))
    assert caught.value is sentinel
    assert failed.cache_saved is False


# ---------------------------------------------------------------------------
# Fast env/backend smoke tests
# ---------------------------------------------------------------------------

# Environments that don't need special config overrides
_STANDARD_ENVS = [
    "G1WalkFlat",
    "G1WalkRough",
    "AllegroInhandRotation",
    "AllegroInhandRotationGrasp",
    "LeapInhandRotation",
    "LeapInhandRotationGrasp",
]


@pytest.mark.parametrize("env_name", _STANDARD_ENVS)
def test_env_reset_and_step(env_name: str):
    """Every registered env must be constructible, resetable, and steppable.

    Verifies:
    - observation/action spaces are valid
    - init_state + reset produces dict obs with correct keys and shapes
    - step with zero actions produces dict obs, scalar reward, bool done
    """
    _require_mujoco_runtime()
    ensure_registries()
    from unilab.base import registry

    # Provide config overrides for envs that require them via Hydra
    env_cfg_override = None
    if env_name == "G1WalkFlat":
        env_cfg_override = _g1_manager_override("g1_walk_flat")
    elif env_name == "G1WalkRough":
        env_cfg_override = _g1_manager_override("g1_walk_rough")
    elif env_name == "AllegroInhandRotation":
        env_cfg_override = _allegro_manager_override()
    elif env_name == "AllegroInhandRotationGrasp":
        env_cfg_override = _allegro_manager_override(task="allegro_inhand_grasp")
    elif env_name == "LeapInhandRotation":
        env_cfg_override = _allegro_manager_override(task="leap_inhand")
    elif env_name == "LeapInhandRotationGrasp":
        env_cfg_override = _allegro_manager_override(task="leap_inhand_grasp")

    env = cast(
        Any,
        registry.make(
            env_name, num_envs=2, sim_backend="mujoco", env_cfg_override=env_cfg_override
        ),
    )
    try:
        # 1. Spaces
        obs_space = env.observation_space
        act_space = env.action_space
        assert obs_space.shape is not None and obs_space.shape[0] > 0
        assert act_space.shape is not None and act_space.shape[0] > 0

        # obs_groups_spec must sum to observation_space total dim
        spec = env.obs_groups_spec
        assert isinstance(spec, dict)
        assert sum(spec.values()) == obs_space.shape[0]

        # 2. Reset
        state = env.init_state()
        assert isinstance(state.obs, dict)
        for key, dim in spec.items():
            assert key in state.obs, f"obs missing group '{key}'"
            assert state.obs[key].shape == (2, dim), (
                f"obs['{key}'] shape mismatch: {state.obs[key].shape} != (2, {dim})"
            )

        # 3. Step with zero actions
        actions = np.zeros((2, act_space.shape[0]))
        state = env.step(actions)
        assert isinstance(state.obs, dict)
        for key, dim in spec.items():
            assert state.obs[key].shape == (2, dim)
        assert state.reward.shape == (2,)
        assert state.terminated.shape == (2,)
        assert state.truncated.shape == (2,)
    finally:
        env.close()


def test_leap_recurrent_actor_uses_only_deployable_observations():
    """LEAP actor is 3x proprio/target/phase while object state stays privileged."""
    _require_mujoco_runtime()
    ensure_registries()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv
    from unilab.tasks.manipulation.leap_inhand.manager_terms import (
        LeapBidirectionalFrontierReward,
        LeapGraspTermination,
        LeapIncrementalPositionAction,
        LeapProprioObservation,
        LeapRotationObservation,
    )

    override = _allegro_manager_override(task="leap_inhand")
    override["observations"]["actor"]["terms"]["proprio"]["params"]["joint_noise"] = 0.0
    override["observations"]["critic"]["terms"]["rotation"]["params"]["joint_noise"] = 0.0
    override["events"]["reset_hand_object"]["params"]["randomization_fraction"] = 0.0
    env = registry.make(
        "LeapInhandRotation",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        initial = env.init_state()
        assert env.obs_groups_spec == {"obs": 102, "critic": 63}
        assert initial.obs["obs"].shape == (2, 102)
        assert initial.obs["critic"].shape == (2, 63)

        actor_term = env.observation_manager.get_term_cfg("actor", "proprio").func
        critic_term = env.observation_manager.get_term_cfg("critic", "rotation").func
        action_term = env.action_manager.get_term("hand")
        assert isinstance(actor_term, LeapProprioObservation)
        assert isinstance(critic_term, LeapRotationObservation)
        assert isinstance(action_term, LeapIncrementalPositionAction)

        actor_history = initial.obs["obs"].reshape(2, 3, 34)
        np.testing.assert_array_equal(actor_history[:, 0], actor_history[:, 1])
        np.testing.assert_array_equal(actor_history[:, 1], actor_history[:, 2])
        expected_phase = np.broadcast_to((0.0, 1.0), actor_history[:, :, -2:].shape)
        np.testing.assert_allclose(actor_history[:, :, -2:], expected_phase, atol=1.0e-7)

        state = env.step(np.zeros((2, 16), dtype=np.float32))
        actor_frame = actor_term(env)
        critic_frame = critic_term(env)
        assert actor_frame.shape == (2, 34)
        assert critic_frame.shape == (2, 63)
        np.testing.assert_allclose(state.obs["obs"][:, -34:], actor_frame, atol=1.0e-7)

        actor_before_object_change = actor_frame.copy()
        critic_before_object_change = critic_frame.copy()
        critic_term.object_pos[:, 0] += 0.1
        np.testing.assert_array_equal(actor_term(env), actor_before_object_change)
        critic_after_object_change = critic_term(env)
        np.testing.assert_allclose(
            critic_after_object_change[:, 50],
            critic_before_object_change[:, 50] + 0.1,
            atol=1.0e-7,
        )

        state = env.step(np.zeros((2, 16), dtype=np.float32))
        expected_angle = env.step_dt * np.pi
        expected_phase = np.broadcast_to((np.sin(expected_angle), np.cos(expected_angle)), (2, 2))
        np.testing.assert_allclose(
            state.obs["obs"][:, -2:],
            expected_phase,
            atol=1.0e-7,
        )

        grasp_term = env.termination_manager.get_term_cfg("dropped").func
        assert isinstance(grasp_term, LeapGraspTermination)
        assert np.all(grasp_term.valid_grasp)
        assert not np.any(grasp_term.fallen)
        assert np.all(grasp_term.thumb_contact)
        assert np.all(grasp_term.opposing_contact_count >= 2)
        assert not np.any(grasp_term.palm_supported)

        critic_term.object_angvel[:] = 0.25 * grasp_term.cylinder_axis_w
        rotate_cfg = env.reward_manager.get_term_cfg("rotate")
        rotate_term = rotate_cfg.func
        assert isinstance(rotate_term, LeapBidirectionalFrontierReward)
        retained_cfg = env.reward_manager.get_term_cfg("retained")
        palm_cfg = env.reward_manager.get_term_cfg("palm_support")
        completed_cfg = env.reward_manager.get_term_cfg("completed")

        def assert_shaped_rotation_reward() -> np.ndarray:
            actual = rotate_term(env, **rotate_cfg.params)
            expected_rate = (
                rotate_term.frontier_progress_multiplier * rotate_term.frontier_speed
                + rotate_term.filtered_directional_speed
                + rotate_term.fast_rotation_bonus
                - rotate_term.opposite_direction_speed
                - 0.05 * rotate_term.stalled
                + (rotate_term.milestone_bonus - 0.05 * rotate_term.reversal_penalty_event)
                / env.step_dt
            )
            np.testing.assert_allclose(
                actual, expected_rate * grasp_term.grasp_quality, atol=1.0e-7
            )
            return actual

        rotate_term.reset()
        assert_shaped_rotation_reward()
        np.testing.assert_allclose(
            rotate_term.frontier_angle,
            0.25 * env.step_dt,
            rtol=2.0e-7,
            atol=3.0e-9,
        )
        assert np.all(rotate_term.frontier_progress_multiplier > 1.0)
        assert np.all(rotate_term.filtered_directional_speed > 0.0)
        assert np.all(rotate_term.filtered_directional_speed < rotate_term.directional_speed)
        np.testing.assert_array_equal(rotate_term.fast_rotation_bonus, np.zeros(2))
        overlay_getter = env.get_playback_debug_overlays()
        assert overlay_getter is not None
        overlays = overlay_getter()
        assert overlays is not None
        assert overlays[0][0].kind == "sphere"
        assert overlays[0][0].rgba == pytest.approx((1.0, 0.8, 0.1, 1.0))

        # Reversing over already visited angles earns nothing; only exceeding
        # the absolute frontier in either direction creates new progress.
        critic_term.object_angvel[:] *= -1.0
        for _ in range(3):
            env.common_step_counter += 1
            grasp_term._last_counter = env.common_step_counter
            assert_shaped_rotation_reward()
        np.testing.assert_array_equal(rotate_term.direction_reversals, np.ones(2))

        # Either sign can establish the initial frontier. Once +0.10 rad is
        # reached, the direction locks and opposite held speed is penalized.
        rotate_term.reset()
        critic_term.object_angvel[:] = 0.5 * grasp_term.cylinder_axis_w
        for _ in range(4):
            env.common_step_counter += 1
            grasp_term._last_counter = env.common_step_counter
            assert_shaped_rotation_reward()
        np.testing.assert_array_equal(rotate_term.committed_direction, np.ones(2))
        np.testing.assert_allclose(rotate_term.frontier_angle, 0.1, atol=1.0e-7)
        assert np.all(rotate_term.fast_rotation_bonus > 0.0)
        overlays = overlay_getter()
        assert overlays is not None
        assert overlays[0][0].rgba == pytest.approx((0.1, 1.0, 0.2, 1.0))

        critic_term.object_angvel[:] = -0.25 * grasp_term.cylinder_axis_w
        env.common_step_counter += 1
        grasp_term._last_counter = env.common_step_counter
        assert_shaped_rotation_reward()
        np.testing.assert_allclose(rotate_term.opposite_direction_speed, 0.25)
        np.testing.assert_array_equal(rotate_term.reversal_penalty_event, np.ones(2, dtype=bool))

        # Continuing in the already-penalized wrong direction pays the dense
        # signed/opposite-speed cost, but not another discrete entry penalty.
        env.common_step_counter += 1
        grasp_term._last_counter = env.common_step_counter
        assert_shaped_rotation_reward()
        np.testing.assert_array_equal(rotate_term.reversal_penalty_event, np.zeros(2, dtype=bool))

        # A frontier crossing pays its milestone exactly once. The configured
        # 0.25 bonus is converted to a rate because the manager later applies dt.
        critic_term.object_angvel[:] = 0.5 * grasp_term.cylinder_axis_w
        for _ in range(7):
            env.common_step_counter += 1
            grasp_term._last_counter = env.common_step_counter
            assert_shaped_rotation_reward()
        np.testing.assert_array_equal(rotate_term.milestones_reached, np.ones(2, dtype=np.int32))
        np.testing.assert_allclose(rotate_term.milestone_bonus, np.full(2, 0.25))
        env.common_step_counter += 1
        grasp_term._last_counter = env.common_step_counter
        assert_shaped_rotation_reward()
        np.testing.assert_array_equal(rotate_term.milestone_bonus, np.zeros(2))

        rotate_term.reset()
        critic_term.object_angvel[:] = -0.5 * grasp_term.cylinder_axis_w
        for _ in range(4):
            env.common_step_counter += 1
            grasp_term._last_counter = env.common_step_counter
            assert_shaped_rotation_reward()
        np.testing.assert_array_equal(rotate_term.committed_direction, -np.ones(2))

        critic_term.object_angvel[:] = 0.25 * grasp_term.cylinder_axis_w
        env.common_step_counter += 1
        grasp_term._last_counter = env.common_step_counter
        assert_shaped_rotation_reward()
        np.testing.assert_allclose(
            retained_cfg.func(env, **retained_cfg.params), grasp_term.grasp_quality, atol=1.0e-7
        )
        np.testing.assert_array_equal(palm_cfg.func(env, **palm_cfg.params), np.zeros(2))

        # Once the post-commit grace period has elapsed, holding still is no
        # longer locally optimal even though the grasp remains valid.
        critic_term.object_angvel.fill(0.0)
        rotate_term.committed_time[:] = 0.5
        rotate_term.filtered_directional_speed.fill(0.0)
        env.common_step_counter += 1
        grasp_term._last_counter = env.common_step_counter
        np.testing.assert_allclose(
            rotate_term(env, **rotate_cfg.params),
            -0.05 * grasp_term.grasp_quality,
            atol=1.0e-7,
        )
        np.testing.assert_array_equal(rotate_term.stalled, np.ones(2, dtype=bool))

        grasp_term.valid_grasp[:] = False
        grasp_term.palm_supported[:] = True
        grasp_term.grasp_quality[:] = 0.0
        held_angle = rotate_term.held_angle.copy()
        env.common_step_counter += 1
        grasp_term._last_counter = env.common_step_counter
        np.testing.assert_array_equal(rotate_term(env, **rotate_cfg.params), np.zeros(2))
        np.testing.assert_array_equal(rotate_term.held_angle, held_angle)
        np.testing.assert_array_equal(retained_cfg.func(env, **retained_cfg.params), np.zeros(2))
        np.testing.assert_array_equal(palm_cfg.func(env, **palm_cfg.params), np.ones(2))
        env.termination_manager.get_term("time_out")[:] = True
        np.testing.assert_array_equal(completed_cfg.func(env, **completed_cfg.params), np.zeros(2))
        rotate_term.frontier_angle[:] = 1.0
        grasp_term.contact_ok[:] = False
        np.testing.assert_array_equal(completed_cfg.func(env, **completed_cfg.params), np.zeros(2))
        grasp_term.contact_ok[:] = True
        np.testing.assert_allclose(
            completed_cfg.func(env, **completed_cfg.params), np.ones(2), atol=1.0e-7
        )
        grasp_term.fallen[:] = True
        np.testing.assert_array_equal(completed_cfg.func(env, **completed_cfg.params), np.zeros(2))

        expected_gait_schedule = ((0, 0.075), (800, 0.075), (5600, 0.075))
        for step, expected_amplitude in expected_gait_schedule:
            env.common_step_counter = step
            assert action_term.gait_amplitude == pytest.approx(expected_amplitude)
        expected_robustness_schedule = (
            (0, 0.0, 0.0),
            (1600, 0.0, 0.0),
            (3200, 0.25, 0.005),
            (4800, 0.5, 0.01),
        )
        for step, delay_probability, noise_std in expected_robustness_schedule:
            env.common_step_counter = step
            assert action_term.action_delay_probability == pytest.approx(delay_probability)
            assert action_term.action_noise_std == pytest.approx(noise_std)
        env.common_step_counter = 0
        assert rotate_cfg.weight == pytest.approx(8.0)
        assert retained_cfg.weight == pytest.approx(1.0)
        assert action_term.thumb_gait_amplitude == pytest.approx(0.0125)

        action_term.reset()
        env.episode_length_buf[:] = 0
        action_term.process_actions(np.zeros((2, 16), dtype=np.float32))
        np.testing.assert_allclose(action_term.gait_reference[:, :12], np.zeros((2, 12)))
        np.testing.assert_allclose(action_term.gait_reference[:, 12:], np.zeros((2, 4)))

        # The thumb pulse starts only after the reset grasp has settled, near
        # the first boundary between opposing-finger release windows.
        action_term.reset()
        env.episode_length_buf[:] = 13
        action_term.process_actions(np.zeros((2, 16), dtype=np.float32))
        assert np.all(action_term.gait_reference[:, 14] < 0.0)
        assert np.max(np.abs(action_term.gait_reference[:, 12:])) <= 0.0125

        # Learned incremental targets cannot park the thumb at its joint limit,
        # and zero actions pull the persistent target back toward the grasp.
        action_term.reset()
        thumb_actions = np.zeros((2, 16), dtype=np.float32)
        thumb_actions[:, 12:] = 1.0
        for _ in range(20):
            action_term.process_actions(thumb_actions)
        thumb_offset = action_term.policy_target[:, 12:] - action_term._reset_target[12:]
        assert np.max(np.abs(thumb_offset)) <= 0.20 + 1.0e-6
        previous_offset = np.linalg.norm(thumb_offset, axis=1)
        action_term.process_actions(np.zeros((2, 16), dtype=np.float32))
        assert np.all(
            np.linalg.norm(
                action_term.policy_target[:, 12:] - action_term._reset_target[12:], axis=1
            )
            < previous_offset
        )

        # No opposing finger can accumulate an indefinitely open policy target;
        # each group relaxes back toward the stable reset grasp.
        for finger_slice in (slice(0, 4), slice(4, 8), slice(8, 12)):
            action_term.reset()
            finger_actions = np.zeros((2, 16), dtype=np.float32)
            finger_actions[:, finger_slice] = 1.0
            for _ in range(20):
                action_term.process_actions(finger_actions)
            finger_offset = (
                action_term.policy_target[:, finger_slice] - action_term._reset_target[finger_slice]
            )
            assert np.max(np.abs(finger_offset)) <= 0.25 + 1.0e-6
            previous_offset = np.linalg.norm(finger_offset, axis=1)
            action_term.process_actions(np.zeros((2, 16), dtype=np.float32))
            assert np.all(
                np.linalg.norm(
                    action_term.policy_target[:, finger_slice]
                    - action_term._reset_target[finger_slice],
                    axis=1,
                )
                < previous_offset
            )

        action_term.reset()
        env.episode_length_buf[:] = 7
        action_term.process_actions(np.zeros((2, 16), dtype=np.float32))
        assert np.any(action_term.gait_reference[:, :12] != 0.0)
        np.testing.assert_array_equal(action_term.gait_reference[:, 12:], np.zeros((2, 4)))
        np.testing.assert_allclose(
            action_term.target,
            np.clip(
                action_term.policy_target + action_term.gait_reference,
                action_term.ctrl_lower,
                action_term.ctrl_upper,
            ),
            atol=1.0e-7,
        )

        # The model-3400 continuation keeps its proven widest reset tier while
        # leaving a fixed-start regression subset.
        reset_cfg = env.event_manager.get_term_cfg("reset_hand_object")
        assert reset_cfg.params["randomization_fraction"] == pytest.approx(0.90)
        expected_reset_stages = ((0, 0.90, 0.004, 8.0),)
        for step, fraction, xy_noise, tilt_noise_deg in expected_reset_stages:
            env.common_step_counter = step
            env.curriculum_manager.compute()
            assert reset_cfg.params["randomization_fraction"] == pytest.approx(fraction)
            assert reset_cfg.params["object_xy_noise"] == pytest.approx(xy_noise)
            assert reset_cfg.params["object_tilt_noise_deg"] == pytest.approx(tilt_noise_deg)
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.obs["critic"]).all()
    finally:
        env.close()


def test_leap_action_can_anchor_targets_to_randomized_reset_joint_state():
    """A cache-aware action starts and relaxes around each environment's reset pose."""
    _require_mujoco_runtime()
    ensure_registries()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv
    from unilab.tasks.manipulation.leap_inhand.manager_terms import (
        LeapIncrementalPositionAction,
        gait_state_metric,
    )

    override = _allegro_manager_override(task="leap_inhand")
    action_cfg = override["actions"]["hand"]
    action_cfg["reset_target_from_joint_state"] = True
    action_cfg["action_delay_probability_schedule"] = [{"step": 0, "probability": 0.0}]
    action_cfg["action_noise_std_schedule"] = [{"step": 0, "std": 0.0}]
    reset_cfg = override["events"]["reset_hand_object"]["params"]
    reset_cfg.update(
        {
            "grasp_cache_path": None,
            "randomization_fraction": 1.0,
            "joint_noise": 0.05,
            "object_xy_noise": 0.0,
            "object_z_noise": 0.0,
            "object_tilt_noise_deg": 0.0,
            "object_velocity_noise": 0.0,
            "object_angular_velocity_noise": 0.0,
        }
    )
    override["curriculum"]["start_pose_randomization"] = None
    env = registry.make(
        "LeapInhandRotation",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        env.init_state()
        action = env.action_manager.get_term("hand")
        assert isinstance(action, LeapIncrementalPositionAction)
        reset_joint_pos = env.scene["robot"].data.joint_pos[:, action.joint_ids]
        np.testing.assert_allclose(action.episode_anchor, reset_joint_pos, atol=1.0e-7)
        np.testing.assert_allclose(action.policy_target, reset_joint_pos, atol=1.0e-7)
        assert np.any(np.abs(action.episode_anchor - action._reset_target) > 1.0e-4)

        action.process_actions(np.zeros((2, 16), dtype=np.float32))
        np.testing.assert_allclose(action.policy_target, action.episode_anchor, atol=1.0e-7)
        for field in (
            "thumb_policy_offset_l2",
            "index_finger_policy_offset_l2",
            "middle_finger_policy_offset_l2",
            "outer_finger_policy_offset_l2",
        ):
            np.testing.assert_allclose(
                gait_state_metric(env, action_name="hand", field=field),
                np.zeros(2),
                atol=1.0e-7,
            )

        positive_actions = np.ones((2, 16), dtype=np.float32)
        for _ in range(20):
            action.process_actions(positive_actions)
        assert np.max(np.abs(action.policy_target[:, :12] - action.episode_anchor[:, :12])) <= (
            0.25 + 1.0e-6
        )
        assert np.max(np.abs(action.policy_target[:, 12:] - action.episode_anchor[:, 12:])) <= (
            0.20 + 1.0e-6
        )
    finally:
        env.close()


def test_leap_reset_can_assign_cache_rows_sequentially(tmp_path: Path):
    """Qualification mode exposes deterministic cache-row identities."""
    _require_mujoco_runtime()
    ensure_registries()
    from unilab.base import registry
    from unilab.tasks.manipulation.leap_inhand.manager_terms import LeapHandObjectReset

    cache = np.zeros((3, 23), dtype=np.float32)
    cache[:, 19] = 1.0
    cache[:, 0] = (0.1, 0.2, 0.3)
    cache_path = tmp_path / "sequential.npy"
    np.save(cache_path, cache)

    override = _allegro_manager_override(task="leap_inhand")
    override["curriculum"]["start_pose_randomization"] = None
    reset_params = override["events"]["reset_hand_object"]["params"]
    reset_params.update(
        {
            "grasp_cache_path": str(cache_path),
            "grasp_cache_sampling": "sequential",
            "grasp_cache_start_index": 1,
            "randomization_fraction": 0.0,
        }
    )
    env = registry.make(
        "LeapInhandRotation",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=override,
    )
    try:
        env.init_state()
        reset = env.event_manager.get_term_cfg("reset_hand_object").func
        assert isinstance(reset, LeapHandObjectReset)
        np.testing.assert_array_equal(reset.grasp_cache_index, (1, 2))
        env.reset(env_ids=np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(reset.grasp_cache_index, (0, 2))
    finally:
        env.close()


def test_leap_grasp_qualification_writes_policy_ranked_caches(tmp_path: Path):
    from types import SimpleNamespace

    from unilab.base.run_control import RunComplete
    from unilab.managers import RecorderTermCfg
    from unilab.tasks.manipulation.leap_inhand.grasp_qualification import (
        LeapGraspQualificationRecorder,
    )
    from unilab.tasks.manipulation.leap_inhand.manager_terms import (
        LeapBidirectionalFrontierReward,
        LeapGraspTermination,
        LeapHandObjectReset,
    )

    cache = np.zeros((2, 23), dtype=np.float32)
    cache[:, 19] = 1.0
    cache[:, 0] = (0.1, 0.2)
    cache_path = tmp_path / "input.npy"
    np.save(cache_path, cache)

    reset = object.__new__(LeapHandObjectReset)
    reset._grasp_cache = cache.astype(np.float64)
    reset._grasp_cache_sampling = "sequential"
    reset.grasp_cache_index = np.array([0, 1], dtype=np.int64)
    grasp = object.__new__(LeapGraspTermination)
    grasp.valid_grasp = np.array([True, True])
    rotation = object.__new__(LeapBidirectionalFrontierReward)
    rotation.directional_speed = np.array([0.2, 0.1])
    rotation.frontier_angle = np.array([1.2, 0.4])
    env = SimpleNamespace(
        num_envs=2,
        reset_buf=np.array([False, False]),
        reset_time_outs=np.array([True, False]),
        reset_terminated=np.array([False, True]),
        extras={"log": {}},
        event_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=reset)),
        termination_manager=SimpleNamespace(
            get_term_cfg=lambda name: SimpleNamespace(func=grasp)
        ),
        reward_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=rotation)),
    )
    prefix = tmp_path / "qualified"
    cfg = RecorderTermCfg(
        func=LeapGraspQualificationRecorder,
        params={
            "reset_event_name": "reset_hand_object",
            "grasp_term_name": "dropped",
            "rotation_reward_name": "rotate",
            "input_cache_path": str(cache_path),
            "output_prefix": str(prefix),
            "episodes_per_grasp": 1,
            "mastered_minimum_survival_fraction": 1.0,
            "mastered_minimum_frontier_angle": 1.0,
            "mastered_minimum_valid_grasp_fraction": 1.0,
            "frontier_minimum_survival_fraction": 0.0,
            "frontier_minimum_frontier_angle": 0.25,
            "frontier_minimum_valid_grasp_fraction": 0.5,
            "training_cache_size": 10,
            "training_mastered_fraction": 0.3,
            "sampling_seed": 7,
        },
    )
    recorder = LeapGraspQualificationRecorder(cfg, cast(Any, env))
    recorder.record_post_reset(np.array([0, 1], dtype=np.int32))
    recorder.record_post_step()
    env.reset_buf[:] = True

    with pytest.raises(RunComplete) as caught:
        recorder.record_pre_reset(np.array([0, 1], dtype=np.int32))

    assert caught.value.reason == "grasp_qualification_complete"
    assert caught.value.summary["mastered_grasps"] == 1
    assert caught.value.summary["frontier_grasps"] == 1
    assert np.load(tmp_path / "qualified_mastered.npy").shape == (1, 23)
    assert np.load(tmp_path / "qualified_frontier.npy").shape == (1, 23)
    assert np.load(tmp_path / "qualified_training.npy").shape == (10, 23)
    scores = np.load(tmp_path / "qualified_scores.npz")
    np.testing.assert_array_equal(scores["classification"], (2, 1))


def test_leap_grasp_contract_distinguishes_support_from_geometric_fall():
    from unilab.managers import TerminationTermCfg
    from unilab.tasks.manipulation.leap_inhand.manager_terms import (
        LeapGraspTermination,
        LeapRotationObservation,
    )

    observation = object.__new__(LeapRotationObservation)
    observation.object_pos = np.array(
        [[0.0, 0.0, 0.25], [0.0, 0.0, 0.23], [0.0, 0.0, 0.209]], dtype=np.float32
    )
    observation.init_object_pos = np.full((3, 3), (0.0, 0.0, 0.25), dtype=np.float32)
    observation.object_quat = np.tile((1.0, 0.0, 0.0, 0.0), (3, 1)).astype(np.float32)
    observation.init_object_quat = observation.object_quat.copy()
    observation._last_counter = 0

    sensor_values = {
        ("if", "mf", "rf"): np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        ),
        ("th",): np.ones((3, 1), dtype=np.float32),
        ("palm",): np.array([[0.0], [1.0], [0.0]], dtype=np.float32),
    }

    class _Scene:
        def bind_sensor_data(self, names):
            values = sensor_values[tuple(names)]
            return SimpleNamespace(
                dimensions=(1,) * values.shape[1], read=lambda values=values: values
            )

    env = SimpleNamespace(
        num_envs=3,
        common_step_counter=0,
        step_dt=0.05,
        scene=_Scene(),
        observation_manager=SimpleNamespace(
            get_term_cfg=lambda group, name: SimpleNamespace(func=observation)
        ),
    )
    cfg = TerminationTermCfg(
        func=LeapGraspTermination,
        params={
            "observation_group": "critic",
            "observation_term": "rotation",
            "opposing_contact_sensor_groups": [["if"], ["mf"], ["rf"]],
            "thumb_contact_sensor_names": ["th"],
            "palm_contact_sensor_names": ["palm"],
            "maximum_valid_vertical_drop": 0.02,
            "maximum_valid_lateral_drift": 0.02,
            "maximum_valid_tilt_deg": 20.0,
            "termination_vertical_drop": 0.04,
            "termination_lateral_drift": 0.05,
            "absolute_minimum_height": 0.125,
            "minimum_opposing_contacts": 1,
            "desired_opposing_contacts": 2,
            "finger_contact_grace_seconds": 0.15,
            "palm_support_grace_seconds": 0.25,
            "palm_support_minimum_drop": 0.015,
            "vertical_quality_scale": 0.015,
            "lateral_quality_scale": 0.015,
            "tilt_quality_scale_deg": 20.0,
        },
    )
    term = LeapGraspTermination(cfg, cast(Any, env))

    for counter in range(1, 6):
        env.common_step_counter = counter
        observation._last_counter = counter
        fallen = term(cast(Any, env))

    np.testing.assert_array_equal(fallen, [False, False, True])
    np.testing.assert_array_equal(term.palm_supported, [False, True, False])
    np.testing.assert_array_equal(term.valid_grasp, [True, False, False])
    assert term.grasp_quality[0] > 0.0
    assert term.grasp_quality[1] == 0.0
    assert term.grasp_quality[2] == 0.0


def test_allegro_manager_runtime_transition_contract():
    _require_mujoco_runtime()
    ensure_registries()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv
    from unilab.tasks.manipulation.allegro_inhand.manager_terms import (
        AllegroIncrementalPositionAction,
        AllegroRotationObservation,
    )

    manager_override = _allegro_manager_override()
    manager_override["observations"]["policy"]["terms"]["rotation"]["params"]["joint_noise"] = 0.0
    env = registry.make(
        "AllegroInhandRotation",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=manager_override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        manager_initial = env.init_state()
        history = manager_initial.obs["obs"].reshape(2, 3, 35)
        np.testing.assert_array_equal(history[:, 0], history[:, 1])
        np.testing.assert_array_equal(history[:, 1], history[:, 2])

        action = env.action_manager.get_term("hand")
        assert isinstance(action, AllegroIncrementalPositionAction)
        target_before = action.target.copy()
        actions = np.full((2, 16), 0.25, dtype=np.float32)
        manager_state = env.step(actions)
        expected_target = np.clip(
            target_before + 0.25 / 24.0,
            action.ctrl_lower,
            action.ctrl_upper,
        )
        np.testing.assert_allclose(action.target, expected_target, rtol=0.0, atol=1.0e-7)

        observation = env.observation_manager.get_term_cfg("policy", "rotation").func
        assert isinstance(observation, AllegroRotationObservation)
        current_frame = observation(env)
        np.testing.assert_allclose(
            manager_state.obs["obs"][:, -35:], current_frame, rtol=0.0, atol=1.0e-7
        )
        assert np.isfinite(manager_state.obs["obs"]).all()
        assert np.isfinite(manager_state.reward).all()
        assert manager_state.terminated.dtype == np.bool_
        assert action.action_dim == 16
        assert env.obs_groups_spec == {"obs": 105}
    finally:
        env.close()


@pytest.mark.parametrize("sim_backend", ["mujoco", "motrix"])
def test_allegro_grasp_manager_runtime_uses_zero_increment_action(sim_backend: str, tmp_path: Path):
    if sim_backend == "mujoco":
        _require_mujoco_runtime()
    else:
        pytest.importorskip("motrixsim")
    ensure_registries()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv
    from unilab.tasks.manipulation.allegro_inhand.grasp_gen import (
        AllegroGraspQualityTermination,
        AllegroGraspRecorder,
    )
    from unilab.tasks.manipulation.allegro_inhand.manager_terms import (
        AllegroIncrementalPositionAction,
    )

    override = _allegro_manager_override(sim_backend, task="allegro_inhand_grasp")
    override["auto_reset"] = False
    override["terminations"]["invalid_grasp"]["params"]["enabled"] = False
    override["recorders"]["grasp_cache"]["params"].update(
        {"output_path": str(tmp_path / f"{sim_backend}.npy"), "auto_save": False}
    )
    env = registry.make(
        "AllegroInhandRotationGrasp",
        num_envs=2,
        sim_backend=sim_backend,
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        initial = env.init_state()
        action = env.action_manager.get_term("hand")
        quality = env.termination_manager.get_term_cfg("invalid_grasp").func
        recorder = env.recorder_manager.get_term("grasp_cache")
        assert isinstance(action, AllegroIncrementalPositionAction)
        assert isinstance(quality, AllegroGraspQualityTermination)
        assert isinstance(recorder, AllegroGraspRecorder)
        target = action.target.copy()

        state = env.step(np.ones((2, 16), dtype=np.float32))
        np.testing.assert_array_equal(action.target, target)
        np.testing.assert_array_equal(state.reward, np.zeros(2, dtype=state.reward.dtype))
        assert initial.obs["obs"].shape == (2, 105)
        assert state.obs["obs"].shape == (2, 105)
        assert quality.last_counter == env.common_step_counter
    finally:
        env.close()


_MOTION_CORE_RUNTIME_CASES = (
    pytest.param("ppo", "g1_motion_tracking", "G1MotionTracking", 160, 286, 29, False),
    pytest.param(
        "ppo",
        "g1_motion_tracking_deploy",
        "G1MotionTrackingDeploy",
        154,
        286,
        29,
        False,
    ),
    pytest.param("ppo", "g1_23dof_motion_tracking", "G1MotionTracking23Dof", 130, 256, 23, False),
    pytest.param(
        "ppo",
        "g1_23dof_motion_tracking_deploy",
        "G1MotionTracking23DofDeploy",
        124,
        256,
        23,
        False,
    ),
    pytest.param("appo", "g1_motion_tracking", "G1MotionTracking", 160, 286, 29, False),
    pytest.param("appo", "g1_23dof_motion_tracking", "G1MotionTracking23Dof", 130, 256, 23, False),
    pytest.param(
        "sac",
        "g1_motion_tracking",
        "G1MotionTrackingSAC",
        160,
        289,
        29,
        True,
    ),
    pytest.param(
        "sac",
        "g1_23dof_motion_tracking",
        "G1MotionTrackingSAC23Dof",
        130,
        259,
        23,
        True,
    ),
)


def test_g1_motion_core_registrations_are_manager_only() -> None:
    from unilab.base import registry

    ensure_registries()
    metadata = registry.list_registered_envs()
    for task_name in (
        "G1MotionTracking",
        "G1MotionTrackingDeploy",
        "G1MotionTracking23Dof",
        "G1MotionTracking23DofDeploy",
        "G1MotionTrackingSAC23Dof",
    ):
        assert metadata[task_name] == {
            "config_factory": "ManagerBasedRlEnvCfg",
            "available_backends": ["mujoco", "motrix"],
        }
    # mjwarp is registered for G1MotionTrackingSAC only (benchmark scope, #1292).
    assert metadata["G1MotionTrackingSAC"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco", "motrix", "mjwarp"],
    }


def test_g1_motion_manager_ppo_wraps_only_active_rows_in_one_state_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_registries()
    _require_mujoco_runtime()
    from unilab.base import registry

    _, override = _motion_manager_override("g1_motion_tracking", "mujoco")
    env = registry.make(
        "G1MotionTracking",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=override,
    )
    try:
        env.init_state()
        command = env.command_manager.get_term("motion")
        command.time_steps[:] = command.sampler.current_clip_end_frames
        env.reset_buf[:] = [True, False]

        set_state_env_ids: list[np.ndarray] = []
        original_set_state = env._backend.set_state

        def record_set_state(
            env_ids: np.ndarray,
            qpos: np.ndarray,
            qvel: np.ndarray,
            *,
            randomization: Any = None,
        ) -> Any:
            set_state_env_ids.append(env_ids.copy())
            return original_set_state(
                env_ids,
                qpos,
                qvel,
                randomization=randomization,
            )

        monkeypatch.setattr(env._backend, "set_state", record_set_state)
        all_ids = np.arange(env.num_envs, dtype=np.int32)
        with env._reset_state.scoped(all_ids):
            env.command_manager.compute(dt=0.0)
        env.command_manager.post_compute()

        assert len(set_state_env_ids) == 1
        np.testing.assert_array_equal(set_state_env_ids[0], [1])
        assert command.time_steps[0] == command.sampler.current_clip_end_frames[0]
        assert command.time_steps[1] <= command.sampler.current_clip_end_frames[1]
        expected_motion = command.motion.get_motion_at_frame(command.time_steps)
        np.testing.assert_array_equal(command.joint_pos, expected_motion.joint_pos)
        np.testing.assert_array_equal(
            command._robot_body_pos_w,
            command.robot.data.body_link_pos_w[:, command._robot_body_ids],
        )
        assert command._robot_cache_step == env.common_step_counter
    finally:
        env.close()


def test_g1_motion_manager_sac_clip_end_is_truncation() -> None:
    ensure_registries()
    _require_mujoco_runtime()
    from unilab.base import registry

    _, override = _motion_manager_override(
        "g1_motion_tracking",
        "mujoco",
        config_root="sac",
    )
    override["auto_reset"] = False
    env = registry.make(
        "G1MotionTrackingSAC",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=override,
    )
    try:
        env.init_state()
        command = env.command_manager.get_term("motion")
        command.time_steps[:] = command.sampler.current_clip_end_frames

        state = env.step(np.zeros((2, 29), dtype=np.float32))

        np.testing.assert_array_equal(state.terminated, [False, False])
        np.testing.assert_array_equal(state.truncated, [True, True])
        np.testing.assert_array_equal(command.time_steps, command.sampler.current_clip_end_frames)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("config_root", "task", "identity", "actor_dim", "critic_dim", "action_dim", "truncate"),
    _MOTION_CORE_RUNTIME_CASES,
)
@pytest.mark.parametrize("sim_backend", ["mujoco", "motrix"])
def test_g1_motion_core_manager_reset_and_step(
    config_root: str,
    task: str,
    identity: str,
    actor_dim: int,
    critic_dim: int,
    action_dim: int,
    truncate: bool,
    sim_backend: str,
) -> None:
    ensure_registries()
    from unilab.base import registry
    from unilab.envs import ManagerBasedRlEnv

    if sim_backend == "mujoco":
        _require_mujoco_runtime()
    else:
        pytest.importorskip("motrixsim")

    task_name, override = _motion_manager_override(
        task,
        sim_backend,
        config_root=config_root,
    )
    assert task_name == identity
    env = registry.make(
        identity,
        num_envs=2,
        sim_backend=sim_backend,
        env_cfg_override=override,
    )
    assert isinstance(env, ManagerBasedRlEnv)
    try:
        assert env.obs_groups_spec == {"obs": actor_dim, "critic": critic_dim}
        assert env.action_space.shape == (action_dim,)
        command = env.command_manager.get_term("motion")
        assert command.cfg.params.truncate_on_clip_end is truncate
        if sim_backend == "motrix" and "deploy" in task:
            assert env._cfg.events["foot_friction"] is None
            assert env._cfg.events["push_robot"] is None
        if sim_backend == "motrix" and config_root in {"ppo", "appo"}:
            root_pos = env._cfg.rewards["motion_global_root_pos"]
            action_rate = env._cfg.rewards["action_rate_l2"]
            assert root_pos is not None
            assert action_rate is not None
            expected_weights = (1.0, -0.05) if config_root == "ppo" else (0.5, -0.1)
            assert (root_pos.weight, action_rate.weight) == pytest.approx(expected_weights)

        state = env.init_state()
        assert state.obs["obs"].shape == (2, actor_dim)
        assert state.obs["critic"].shape == (2, critic_dim)

        state = env.step(np.zeros((2, action_dim), dtype=np.float32))
        assert state.reward.shape == (2,)
        assert state.terminated.shape == (2,)
        assert state.truncated.shape == (2,)
        assert np.isfinite(state.reward).all()
        assert all(np.isfinite(values).all() for values in state.obs.values())
    finally:
        env.close()
