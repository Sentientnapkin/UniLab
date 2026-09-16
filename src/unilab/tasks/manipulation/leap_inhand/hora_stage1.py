"""Registry owner for the isolated HORA-style LEAP Stage 1 task."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

registry.register_env_config("LeapInhandHoraStage1", ManagerBasedRlEnvCfg)
registry.register_env(
    "LeapInhandHoraStage1",
    make_manager_based_rl_env,
    sim_backend="mujoco",
)
