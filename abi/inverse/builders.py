import numpy as np
from abi.inverse.sampler import MarketScenario


def build_portfolio(
    rates: np.ndarray,
    fixed_features: np.ndarray,
) -> np.ndarray:
    rates = np.asarray(rates, dtype=np.float32)
    fixed_features = np.asarray(fixed_features, dtype=np.float32)
    return np.concatenate([rates[:, None], fixed_features], axis=1)


# DEFAULT_COMPETITORS = np.array([
#     [0.30, 0.50, 0.40],
#     [0.45, 0.58, 0.52],
#     [0.58, 0.62, 0.65],
#     [0.68, 0.72, 0.75],
#     [0.80, 0.82, 0.88],
# ], dtype=np.float32)
#
# COMPETITOR_NAMES = ["budget", "economy", "mid", "comfort", "premium"]

DEFAULT_COMPETITORS = np.array([
    # [rate, area, location_score]
    [0.35, 0.35, 0.30],   # budget:  дёшево, но плохое качество
    [0.55, 0.45, 0.45],   # mid:     средняя ставка, посредственное качество
    [0.75, 0.55, 0.60],   # premium: дорого, но не намного лучше нас по качеству
], dtype=np.float32)

COMPETITOR_NAMES = ["budget", "mid", "premium"]


def build_competitors(
    competitor_features: np.ndarray = None,
) -> np.ndarray:
    if competitor_features is None:
        return DEFAULT_COMPETITORS.copy()
    return np.asarray(competitor_features, dtype=np.float32)


def build_agents(
    phi: MarketScenario,
    weights: np.ndarray,
    feature_directions: np.ndarray,
    n_props: int = 1,
    ref_center: float = 0.5,
) -> dict:
    M = phi.n_agents
    N_FEATURES = len(weights)

    rng = np.random.default_rng()
    refs = rng.normal(
        loc=ref_center + phi.ref_shift,
        scale=phi.ref_std,
        size=(M, N_FEATURES),
    ).astype(np.float32)
    refs = np.clip(refs, 0.0, 1.0)

    switching_costs = rng.normal(0.2, 0.05, size=M) * phi.switching_cost_scale
    switching_costs = np.clip(switching_costs, 1e-6, None).astype(np.float32)

    agent_to_property = rng.integers(0, n_props, size=M).astype(np.int32)

    return {
        "refs": refs,
        "w": np.asarray(weights, dtype=np.float32),
        "directions": np.asarray(feature_directions, dtype=np.float32),
        "switching_costs": switching_costs,
        "current": np.full(M, -1, dtype=np.int32),
        "agent_to_property": agent_to_property,
    }