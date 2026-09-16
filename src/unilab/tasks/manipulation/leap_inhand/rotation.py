"""Registry owner for the LEAP in-hand rotation task."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

registry.register_env_config("LeapInhandRotation", ManagerBasedRlEnvCfg)

registry.register_env(
    "LeapInhandRotation",
    make_manager_based_rl_env,
    sim_backend="mujoco",
)
