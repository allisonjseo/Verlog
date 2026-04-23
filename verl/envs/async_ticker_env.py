"""Backwards-compatible re-export shim.

Assembles the pair of classes that used to live in this file from their new
homes:

- ``AsyncTickerAdmissionsEnv`` comes from the ``hiring_env`` submodule at
  ``verl/envs/hiring_env/`` (https://github.com/WentseChen/hiring_env). Its
  internal imports are flat (``from history_manager import X``), so we
  prepend the submodule directory to ``sys.path`` before importing.
- ``AsyncTickerEnvWrapper`` is the Verlog-side single-agent adapter and
  lives in ``verl/envs/async_ticker_wrapper.py``.

Existing callers that ``from verl.envs.async_ticker_env import
AsyncTickerAdmissionsEnv, AsyncTickerEnvWrapper`` continue to work without
change.
"""
import os
import sys

_SUBMODULE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hiring_env"
)
if _SUBMODULE_DIR not in sys.path:
    sys.path.insert(0, _SUBMODULE_DIR)

from env import AsyncTickerAdmissionsEnv  # noqa: E402
from verl.envs.async_ticker_wrapper import AsyncTickerEnvWrapper  # noqa: E402

__all__ = ["AsyncTickerAdmissionsEnv", "AsyncTickerEnvWrapper"]
