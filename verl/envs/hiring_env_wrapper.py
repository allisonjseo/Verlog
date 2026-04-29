# verl/envs/hiring_env_wrapper.py
"""
Wrapper for hiring_env submodule to integrate with VERL framework.
"""

import gym
from typing import Any, Dict, List, Optional, Tuple

# Import from submodule
from verl.envs.hiring_env.env import AsyncTickerAdmissionsEnv


class AsyncTickerEnvWrapper(gym.Wrapper):
    """
    VERL wrapper for AsyncTickerAdmissionsEnv.

    Handles:
    - Observation caching for get_last_obs()
    - Auto-reset logic
    - Episode state management
    - Integration with VERL training loops
    """

    def __init__(self, env: AsyncTickerAdmissionsEnv):
        super().__init__(env)
        self.last_obs = None
        self.last_info = None
        self.episode_done = False

    def get_last_obs(self):
        """Return last observation (used by VERL for resuming after reset)."""
        return self.last_obs, self.last_info

    def reset(self, **kwargs):
        """Reset environment and cache initial observation."""
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        self.last_info = info
        self.episode_done = False
        return obs, info

    def step(self, action):
        """Step environment with auto-reset on termination."""
        # If episode already done, auto-reset
        if self.episode_done:
            obs, info = self.reset()
            return obs, 0.0, False, False, info

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Cache for get_last_obs()
        self.last_obs = obs
        self.last_info = info

        # Track episode status
        if terminated or truncated:
            self.episode_done = True

        return obs, reward, terminated, truncated, info


def make_async_ticker_env(config: Dict[str, Any], tokenizer=None) -> AsyncTickerEnvWrapper:
    """
    Factory function to create wrapped AsyncTickerAdmissionsEnv.

    Args:
        config: Environment configuration dict with keys:
            - professor_ids: List[str]
            - students_per_batch: int
            - token_budget: int
            - feature_dim: int (default 5)
            - vote_threshold: float (default 0.5)
            - max_steps: int (optional)
            - reward_mode: str ("individual"/"group"/"combined")
            - reward_alpha: float (for "combined" mode)
            - system_prompt: str or Dict[str, str]
            - prompt_length: int (optional, for token-based truncation)
            - max_prompt_words: int (optional, for word-based truncation)
            - seed: int (optional)
        tokenizer: Optional tokenizer for prompt truncation

    Returns:
        Wrapped environment ready for VERL training
    """
    env = AsyncTickerAdmissionsEnv(config, tokenizer=tokenizer)
    return AsyncTickerEnvWrapper(env)
