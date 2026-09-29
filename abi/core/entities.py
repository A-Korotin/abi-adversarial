"""Real-estate domain types."""

from dataclasses import dataclass

import numpy as np


FEATURE_NAMES = ["rate", "area", "location_score"]
N_FEATURES = len(FEATURE_NAMES)

FEATURE_DIRECTIONS = np.array([-1.0, 1.0, 1.0], dtype=np.float32)


@dataclass
class Property:
    property_id: int
    rate: float
    area: float
    location_score: float

    @property
    def features(self) -> np.ndarray:
        return np.array([self.rate, self.area, self.location_score], dtype=np.float32)


@dataclass
class Competitor:
    name: str
    features: np.ndarray  # shape (N_FEATURES,)
    cluster_size: float = 1.0  # number of real units this cluster aggregates

    def __post_init__(self):
        self.features = np.asarray(self.features, dtype=np.float32)


@dataclass
class TenantAgent:
    w: np.ndarray
    ref: np.ndarray
    directions: np.ndarray
    switching_cost: float = 0.0
    current_property: int = -1
    is_tenant: bool = False

    def __post_init__(self):
        self.w = np.asarray(self.w, dtype=np.float32)
        self.ref = np.asarray(self.ref, dtype=np.float32)
        self.directions = np.asarray(self.directions, dtype=np.float32)
