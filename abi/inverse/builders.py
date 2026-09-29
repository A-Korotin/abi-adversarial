"""
Factory functions for building portfolio components.

- build_agents()           — generate heterogeneous tenant agent arrays
- build_competitors()      — competitor feature matrix
- build_competitor_log_sizes() — log cluster sizes for competitors
"""

import numpy as np
from abi.inverse.scenario import MarketScenario


# ── Default competitor clusters (budget / mid / premium) ────────────────────
DEFAULT_COMPETITORS = np.array([
    # [rate, area, location_score]
    [0.35, 0.35, 0.30],   # budget
    [0.55, 0.45, 0.45],   # mid
    [0.75, 0.55, 0.60],   # premium
], dtype=np.float32)

COMPETITOR_NAMES = ["budget", "mid", "premium"]

DEFAULT_COMPETITOR_SIZES = np.array([3.0, 2.0, 1.0], dtype=np.float32)
DEFAULT_OUTSIDE_SIZE = 1.0


def build_competitors(
    competitor_features: np.ndarray = None,
) -> np.ndarray:
    if competitor_features is None:
        return DEFAULT_COMPETITORS.copy()
    return np.asarray(competitor_features, dtype=np.float32)


def build_competitor_log_sizes(
    competitor_sizes: np.ndarray = None,
) -> np.ndarray:
    if competitor_sizes is None:
        sizes = DEFAULT_COMPETITOR_SIZES
    else:
        sizes = np.asarray(competitor_sizes, dtype=np.float32)
    return np.log(sizes).astype(np.float32)


def build_agents(
    scenario: MarketScenario,
    weights: np.ndarray,
    feature_directions: np.ndarray,
    n_props: int = 1,
    K: int = 8,
) -> dict:
    """Generate a population of heterogeneous tenant agents.

    Parameters
    ----------
    scenario : MarketScenario
        Population-level parameters (n_agents, ref_center, ref_std, switching_cost_scale).
    weights : array (N_FEATURES,)
        Feature weights for prospect utility (shared across agents).
    feature_directions : array (N_FEATURES,)
        Sign per feature: -1 = lower is better (rate), +1 = higher is better.
    n_props : int
        Number of portfolio properties (for random property visibility assignment).
    K : int
        Number of properties visible to each agent.

    Returns
    -------
    dict with keys:
        refs               — (M, N_F) float32, initial reference points
        w                  — (N_F,) float32, feature weights
        directions         — (N_F,) float32, feature directions
        switching_costs    — (M,) float32, per-agent switching cost
        current_choice     — (M,) int32, current option index (-1 = none)
        agent_to_properties — (M, K) int32, visible property indices
    """
    M = scenario.n_agents
    N_FEATURES = len(weights)

    rng = np.random.default_rng()
    refs = rng.normal(
        loc=scenario.ref_center,
        scale=scenario.ref_std,
        size=(M, N_FEATURES),
    ).astype(np.float32)
    refs = np.clip(refs, 0.0, 1.0)

    switching_costs = rng.normal(0.2, 0.05, size=M) * scenario.switching_cost_scale
    switching_costs = np.clip(switching_costs, 1e-6, None).astype(np.float32)

    agent_to_properties = rng.integers(0, n_props, size=(M, K)).astype(np.int32)

    return {
        "refs": refs,
        "w": np.asarray(weights, dtype=np.float32),
        "directions": np.asarray(feature_directions, dtype=np.float32),
        "switching_costs": switching_costs,
        "current_choice": np.full(M, -1, dtype=np.int32),
        "agent_to_properties": agent_to_properties,
    }
