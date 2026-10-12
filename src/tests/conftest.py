"""Pytest configuration: add src/ to path so `import qrnn_uq_rul` works."""
import os
import sys
from pathlib import Path

# Ensure the package is importable when running from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# On macOS, PyTorch and XGBoost each bundle an OpenMP runtime (libomp); loading
# both with multi-threading in one process can segfault (double-init). The test
# suite imports torch arms before the C-XGBQ arm in a single process, so pin the
# OpenMP pool to 1 thread here — BEFORE any of these libraries are imported.
# This is test-harness hygiene only; real Phase-4 runs launch arms as separate
# processes/scripts and keep full threading.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
