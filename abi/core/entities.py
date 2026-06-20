from dataclasses import dataclass
from typing import List

import numpy as np

from abi.core.functions import prospect_np


class Provider:
    def __init__(self, name: str, p: np.array):
        self.name = name
        self.p = p

    def __repr__(self):
        return f"Provider '{self.name}': {self.p}"


class Agent:
    def __init__(self, w: np.array, ref: np.array, directions: np.array):
        self.w = w
        self.ref = ref
        self.directions = directions
        self.current_provider = -1
        self.switching_cost = 0

    def utility_for(self, option: np.array):
        deltas = (option - self.ref) * self.directions
        prospect_activation = prospect_np(deltas)

        return (prospect_activation @ self.w).item()

    def utility_for_matrix(self, options: np.array):
        deltas = (options - self.ref) * self.directions
        prospect_activation = prospect_np(deltas)
        utilities = prospect_activation @ self.w
        return utilities.flatten()

    def activated_output_for(self, providers: List[Provider], actiavtion_func) -> np.array:
        utilities = np.array([self.utility_for(provider.p) for provider in providers])
        return actiavtion_func(utilities)

    def __repr__(self):
        return f"Agent[w={np.round(self.w, 2)}, ref={np.round(self.ref, 2)}, sw_cost={self.switching_cost}, current_provider={self.current_provider}]"


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
