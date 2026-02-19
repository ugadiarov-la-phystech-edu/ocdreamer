from .target import CwTargetEnv
from gym.envs.registration import register

register('ReachingHard-v0', entry_point='embodied.envs.cw_envs.target:ReachingHard')