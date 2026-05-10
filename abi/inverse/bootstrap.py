from dataclasses import dataclass
from typing import List

import numpy as np

from abi.core.entities import Provider
from abi.core.functions import prospect_np
from abi.inverse.sampler import MarketScenario


def build_providers(xi: np.ndarray, competitors: np.ndarray) -> List[Provider]:
    xi = np.asarray(xi, dtype=np.float32).copy()
    xi = np.clip(xi, 0.0, 1.0)
    us = Provider(name="Wir", p=np.squeeze(xi))

    competitors = np.asarray(competitors, dtype=np.float32).copy()
    competitors = np.clip(competitors, 0.0, 1.0)
    n_competitors, _ = competitors.shape

    names = [f"Competitor_{i}" for i in range(n_competitors)]
    return [us] + [Provider(name=names[i], p=competitors[i]) for i in range(n_competitors)]


@dataclass
class Agent:
    w: np.ndarray
    ref: np.ndarray
    directions: np.ndarray
    switching_cost: float = 0.0
    current_provider: int = -1

    def utility_for_matrix(self, options: np.ndarray):
        deltas = (options - self.ref) * self.directions
        prospect_activation = prospect_np(deltas)
        return prospect_activation @ self.w


def build_agents(phi: MarketScenario, weights: np.ndarray, feature_directions: np.ndarray, n_agents: int):
    agents = []

    for _ in range(n_agents):
        ref = np.random.normal(
            loc=0.5 + phi.ref_shift,
            scale=phi.ref_std,
            size=3
        )

        ref = np.clip(ref, 0.0, 1.0)

        switching_cost = np.random.normal(0.2, 0.05)
        switching_cost *= phi.switching_cost_scale

        agents.append(
            Agent(
                w=weights.copy(),
                ref=ref.astype(np.float32),
                directions=feature_directions.copy(),
                switching_cost=float(max(1e-6, switching_cost))
            )
        )

    return agents
