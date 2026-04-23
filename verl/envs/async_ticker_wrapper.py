"""Verlog-side single-agent adapter for ``AsyncTickerAdmissionsEnv``.

The underlying env (vendored as a submodule at ``verl/envs/hiring_env/``) is
multi-agent: ``reset()`` returns ``(observations, infos_dict_keyed_by_agent)``
and ``step()`` returns five dicts keyed by agent_id.  Verlog's
``tool_agent_loop.run()`` expects a single-agent gym interface — flat
``(obs, reward, terminated, truncated, info)`` tuples and a ``get_last_obs()``
method for resuming partially-completed training episodes across rollout
calls.

This wrapper is Verlog-specific plumbing and deliberately lives outside the
submodule so that hiring_env stays a clean multi-agent env usable from any
training framework.
"""
from typing import Any, Dict, List, Optional, Tuple

import gym


class AsyncTickerEnvWrapper(gym.Wrapper):
    """Single-agent adapter over the multi-agent ``AsyncTickerAdmissionsEnv``.

    - ``reset()`` returns ``(observations, active_agent_info)`` instead of the
      raw ``(observations, {agent_id: info, ...})``.
    - ``step()`` returns flat ``(obs, float_reward, bool_terminated,
      bool_truncated, info)`` for the agent that acted.
    - ``get_last_obs()`` returns the cached last obs/info so the rollout can
      resume, or ``(None, None)`` when the episode has ended (caller is
      expected to fall back to ``reset()`` in that case).
    """

    def __init__(self, env):
        super().__init__(env)
        self._last_observations: Optional[List[Dict[str, str]]] = None
        self._last_infos: Optional[Dict[str, Dict]] = None
        self._episode_done: bool = False

    def reset(
        self, agent_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, str]], Dict[str, Dict]]:
        observations, infos = self.env.reset()
        info = infos[self.env.episode_state["active_agent"]]
        self._last_observations = observations
        self._last_infos = info
        self._episode_done = False
        return observations, info

    def step(
        self, action: Any
    ) -> Tuple[
        List[Dict[str, str]],
        float,
        bool,
        bool,
        Dict[str, Any],
    ]:
        if self._episode_done:
            observations, info = self.reset()
            # Discard the stale action from the previous episode; return the
            # fresh initial observation so the caller generates a new response.
            return observations, 0.0, False, False, info
        acting_agent = self.env.episode_state["active_agent"]
        observations, rewards, terminations, truncations, infos = self.env.step(action)
        reward = rewards[acting_agent]
        terminated = terminations[acting_agent]
        truncated = truncations[acting_agent]
        info = infos[acting_agent]
        self._last_observations = observations
        self._last_infos = info
        if terminated or truncated:
            self._episode_done = True
        return observations, reward, terminated, truncated, info

    def get_last_obs(
        self, agent_id: Optional[str] = None
    ) -> Tuple[Optional[List[Dict[str, str]]], Optional[Dict[str, Any]]]:
        """Return the last ``(observations, info)`` from ``reset`` or ``step``.

        Returns ``(None, None)`` when the episode has ended, signalling to the
        caller that it should invoke ``reset()`` before continuing.
        """
        if (
            self._last_observations is None
            or self._last_infos is None
            or self._episode_done
        ):
            return None, None
        return self._last_observations, self._last_infos
