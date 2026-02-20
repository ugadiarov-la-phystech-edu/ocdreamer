import numpy as np
from PIL import Image

import gym
from gym.envs.registration import register
import gymnasium
import mani_skill.envs


register('push-cube-v0', entry_point='embodied.envs.maniskill3:ManiSkillEnv', kwargs=dict(task='push-cube'))


MANISKILL_TASKS = {
	'push-cube': dict(
		env='PushCube-v1',
		control_mode='pd_joint_delta_pos',
	),
}


class ManiSkillEnv(gym.Env):
	def __init__(self, task, timelimit, size, seed):
		assert task in MANISKILL_TASKS, f'Expected tasks={list(MANISKILL_TASKS.keys())}. Actual task={task}'
		task_cfg = MANISKILL_TASKS[task]
		env = gymnasium.make(
			task_cfg['env'],
			obs_mode='rgbd',
			control_mode=task_cfg['control_mode'],
			render_mode='rgb_array',
			sensor_configs=dict(width=224, height=224),
			# time limit will be enforced in wrapper
			max_episode_steps=timelimit['duration'] + 1,
		)
		self.env = env
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(*size, 3), dtype=np.uint8)
		self.action_space = gym.spaces.Box(
			low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
			high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
			dtype=self.env.action_space.dtype,
		)
		self.seed = seed
		self.env.reset(seed=self.seed)
		self.last_observation = None

	@staticmethod
	def _unravel(step_result):
		unravel_result = [step_result[0]['sensor_data']['base_camera']['rgb'][0]]
		unravel_result += [x[0] if hasattr(x, '__len__') else x for x in step_result[1:-1]]
		info = {key: value[0] if hasattr(value, '__len__') else value for key, value in step_result[-1].items()}
		unravel_result.append(info)

		return unravel_result

	def _process_observation(self, observation):
		self.last_source_observation = observation.numpy()
		shape = self.observation_space.shape
		self.last_observation = np.array(Image.fromarray(self.last_source_observation).resize(shape[:2]))
		return self.last_observation.copy()

	def reset(self, *args, **kwargs):
		obs = self._unravel(self.env.reset())[0]
		return self._process_observation(obs), {}

	def step(self, action):
		obs, r, terminated, truncated, info = self._unravel(self.env.step(action))
		info = {k: v.item() for k, v in info.items()}
		info["success"] = int(info.get("success", 0))
		return self._process_observation(obs), r, False, False, info

	def render(self, *args, **kwargs):
		return self.last_observation.copy()
