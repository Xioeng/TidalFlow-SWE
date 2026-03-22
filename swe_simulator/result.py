"""Configuration dataclass for SWE Simulator."""

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from .config import SimulationConfig
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SWEResult:
    meshgrid_coord: tuple[np.ndarray | float, np.ndarray | float] | None = None
    meshgrid_metric: tuple[np.ndarray, np.ndarray] | None = None
    solution: np.ndarray | None = None
    bathymetry: np.ndarray | None = None
    initial_condition: np.ndarray | None = None
    wind_forcing: tuple[float | np.ndarray, float | np.ndarray] | None = None
    config: SimulationConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, filepath: Path | str) -> None:
        logger.info(f"Saving SWe simulation result to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path | str) -> "SWEResult":
        logger.info(f"Loading SWE simulation result from {filepath}")
        with open(filepath, "rb") as f:
            return cast("SWEResult", pickle.load(f))
