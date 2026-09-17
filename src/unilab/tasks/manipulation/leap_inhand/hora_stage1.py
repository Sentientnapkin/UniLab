"""Registry owner for the isolated HORA-style LEAP Stage 1 task."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg

from .hora_grasp import make_leap_hora_env

registry.register_env_config("LeapInhandHoraStage1", ManagerBasedRlEnvCfg)
registry.register_env(
    "LeapInhandHoraStage1",
    make_leap_hora_env,
    sim_backend="mujoco",
)
