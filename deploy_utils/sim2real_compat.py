"""
Works around mani-skill/ManiSkill#1498: Sim2RealEnv.reset() recurses forever when the sim env is wrapped.

Since ManiSkill #1308 (merged 2026-06-10), Sim2RealEnv.base_sim_env is a property, `self.sim_env.unwrapped`.
During reset()/step() Sim2RealEnv temporarily replaces the last wrapper's .env with a stand-in whose .unwrapped
returns the Sim2RealEnv itself, so inside that window base_sim_env resolves to the Sim2RealEnv and get_obs() calls
itself until Python raises RecursionError. Resolving the simulator once, before any swap, restores the behaviour of
mani-skill 3.0.1 and the nightlies before #1308. Harmless on versions without the bug.
"""

from mani_skill.envs.sim2real_env import Sim2RealEnv as _ManiSkillSim2RealEnv


class Sim2RealEnv(_ManiSkillSim2RealEnv):
    def __init__(self, sim_env, *args, **kwargs):
        self._base_sim_env = sim_env.unwrapped   # resolved before reset()/step() swap the last wrapper's env
        super().__init__(sim_env, *args, **kwargs)

    @property
    def base_sim_env(self):
        return self._base_sim_env
