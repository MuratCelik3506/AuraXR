"""Aggregator re-exporting the OakInk and HOT3D dataset implementations."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.dataset_hot3d import Hot3DTemporalDataset, assert_hot3d_contract, collate_hot3d  # noqa: E402, F401
from data.dataset_oakink import OakInkStaticDataset, assert_oakink_contract, collate_oakink  # noqa: E402, F401
