"""Ensures src/ is importable as a path root regardless of which script is invoked.

Every package __init__ imports this module first so that cross-package imports
(`from model.grasp_model import ...`, `from paths import ...`) work whether a
script is run as `python3 src/training/train_grasp.py` or via `python3 -m`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
