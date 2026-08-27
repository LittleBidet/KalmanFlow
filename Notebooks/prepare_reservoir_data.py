"""Notebook compatibility imports for application preparation helpers.

The implementation lives in :mod:`applications.preparation`; this module
keeps existing notebook cells and imports working without duplicating data
logic.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from applications.preparation import (  # noqa: E402
    PreparedReservoirData,
    ReservoirSources,
    prepare_reservoir_data,
    read_aquarius_series,
    sources_for_reservoir,
)

__all__ = [
    "PreparedReservoirData",
    "ReservoirSources",
    "prepare_reservoir_data",
    "read_aquarius_series",
    "sources_for_reservoir",
]
