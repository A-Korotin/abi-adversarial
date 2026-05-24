import numpy as np
from abi.inverse.sampler import MarketScenario


def build_portfolio(
    rates: np.ndarray,
    fixed_features: np.ndarray,
) -> np.ndarray:
    rates = np.asarray(rates, dtype=np.float32)
    fixed_features = np.asarray(fixed_features, dtype=np.float32)
    return np.concatenate([rates[:, None], fixed_features], axis=1)


DEFAULT_COMPETITORS = np.array([
    # [rate, area, location_score]
    [0.35, 0.35, 0.30],   # budget
    [0.55, 0.45, 0.45],   # mid
    [0.75, 0.55, 0.60],   # premium
], dtype=np.float32)

COMPETITOR_NAMES = ["budget", "mid", "premium"]

DEFAULT_COMPETITOR_SIZES = np.array([3.0, 4.0, 2.0], dtype=np.float32)
DEFAULT_OUTSIDE_SIZE = 5.0


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
    phi: MarketScenario,
    weights: np.ndarray,
    feature_directions: np.ndarray,
    n_props: int = 1,
    ref_center: float = 0.5,
    K: int = 8,
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

    agent_to_properties = rng.integers(0, n_props, size=(M, K)).astype(np.int32)

    return {
        "refs": refs,
        "w": np.asarray(weights, dtype=np.float32),
        "directions": np.asarray(feature_directions, dtype=np.float32),
        "switching_costs": switching_costs,
        "current_nest": np.full(M, -1, dtype=np.int32),
        "current_inner": np.full(M, -1, dtype=np.int32),
        "agent_to_properties": agent_to_properties,
    }
