import gym
import numpy as np


class RemoveTextObs(gym.Wrapper):
  """Gym wrapper (v0.26+) that removes non-numeric observations.
  
  Filters out text, boolean, and other non-numeric observations to ensure
  compatibility with JAX-based models that only accept numeric arrays.
  """

  def __init__(self, env):
    super().__init__(env)
    self._filter_observation_space()

  def _filter_observation_space(self):
    """Keep only numeric spaces"""
    if not isinstance(self.env.observation_space, gym.spaces.Dict):
      return
    
    filtered_spaces = {}
    for key, space in self.env.observation_space.spaces.items():
      if self._is_valid_space(key, space):
        filtered_spaces[key] = space
    
    self.observation_space = gym.spaces.Dict(filtered_spaces)

  def _is_valid_space(self, key, space):
    # Reject text-related keys
    invalid_keywords = ['text', 'token', 'language', 'instruction', 'log']
    if any(kw in key.lower() for kw in invalid_keywords):
      return False
    
    # Reject non-numeric spaces
    if not isinstance(space, (gym.spaces.Box, gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
      return False
    
    # Reject boolean spaces
    if hasattr(space, 'dtype') and space.dtype == bool:
      return False
    
    return True

  def _is_valid_obs_key(self, key):
    """Check if observation key should be kept."""
    invalid_keywords = ['text', 'token', 'language', 'instruction', 'log']
    return not any(kw in key.lower() for kw in invalid_keywords)

  def _is_valid_obs_value(self, value):
    """Check if observation value is numeric (not bool)."""
    if isinstance(value, (bool, np.bool_)):
      return False
    if hasattr(value, 'dtype') and value.dtype == bool:
      return False
    return True

  def _filter_obs(self, obs):
    """Filter observation dict to keep only valid numeric observations."""
    if not isinstance(obs, dict):
      return obs
    
    return {k: v for k, v in obs.items() 
            if self._is_valid_obs_key(k) and self._is_valid_obs_value(v)}

  def reset(self):
    """Reset environment and filter observations (Gym v0.26+ API)."""
    result = self.env.reset()
    # Handle cases where env returns >2 values (take first as obs, last as info)
    if isinstance(result, tuple):
      obs = result[0]
      info = result[-1] if isinstance(result[-1], dict) else {}
    else:
      obs = result
      info = {}
    return self._filter_obs(obs), info

  def step(self, action):
    """Step environment and filter observations (Gym v0.26+ API)."""
    # Convert numpy array actions to scalars for discrete action spaces
    if isinstance(action, np.ndarray):
      action = int(action.item() if action.size == 1 else action[0])
    
    result = self.env.step(action)
    # Handle cases where env returns different number of values
    if len(result) == 5:
      obs, reward, terminated, truncated, info = result
    elif len(result) == 4:
      obs, reward, done, info = result
      terminated, truncated = done, False
    else:
      # Fallback for unexpected return values
      obs = result[0]
      reward = result[1] if len(result) > 1 else 0.0
      terminated = result[2] if len(result) > 2 else False
      truncated = result[3] if len(result) > 3 else False
      info = result[-1] if isinstance(result[-1], dict) else {}
    
    return self._filter_obs(obs), reward, terminated, truncated, info
