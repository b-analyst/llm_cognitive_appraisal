"""Delegate ``pipeline.<name>`` modules to their real implementation under ``stage_*`` / ``core``."""
from __future__ import annotations

import runpy
from importlib import import_module
from typing import Callable


def bind_shim(target: str, module_globals: dict) -> Callable[[], None]:
    """
    Install ``__getattr__`` / ``__dir__`` on *module_globals* and return a no-arg runner
    for ``if __name__ == "__main__"`` (``runpy.run_module`` on *target*).
    """
    def __getattr__(name: str):
        return getattr(import_module(target), name)

    def __dir__():
        return sorted(
            set(vars(import_module(target)).keys()) | set(module_globals.keys())
        )

    module_globals["__getattr__"] = __getattr__
    module_globals["__dir__"] = __dir__

    def _run() -> None:
        runpy.run_module(target, run_name="__main__", alter_sys=True)

    return _run
