"""Process-wide CPU thread limits for BLAS / OpenMP / PyTorch (optional)."""
from __future__ import annotations

import os


def apply_cpu_thread_limits() -> None:
    """
    Cap intra-op CPU threads when PIPELINE_CPU_THREADS is set to a positive integer.
    Does not override OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS if already set.
    When unset, no changes are made (library defaults apply).
    """
    raw = os.environ.get("PIPELINE_CPU_THREADS", "").strip()
    if not raw:
        return
    try:
        n = int(raw, 10)
    except ValueError:
        return
    if n < 1:
        return
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if key not in os.environ:
            os.environ[key] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(max(1, min(4, n)))
        except Exception:
            pass
    except Exception:
        pass
