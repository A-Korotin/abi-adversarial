"""
Simulated moments: run the forward simulator for a parameter set over several
seeds and average the calibration moments.
"""

import numpy as np

from abi.forward.estate import RealEstateSimulator
from abi.inverse.scenario import MarketScenario
from abi.inverse.builders import build_agents
from abi.calibration.moments import compute_moments

MOMENT_KEYS = ("occupancy_area", "occupancy_area_std", "occupancy_count",
               "median_exposure", "p_exposure_gt_tail", "stay_prob")


def simulate_once(params: dict, portfolio: dict, seed: int) -> dict:
    """One simulation run -> moments. Independent sub-seeds for agents, noise, churn, order."""
    agent_seed, gumbel_seed, churn_seed, order_seed = (
        int(x) for x in np.random.SeedSequence(seed).generate_state(4)
    )
    n_props = len(portfolio["rates"])
    M = params["n_agents"]

    sim = RealEstateSimulator(
        T=params["T"],
        K=params["K"],
        outside_utility=params["outside_utility"],
        outside_size=params["outside_size"],
        competitor_sizes=params["competitor_sizes"],
        utility_scale=params["utility_scale"],
        churn_prob=params["churn_prob"],
    )
    scenario = MarketScenario(
        n_agents=M,
        ref_center=params["ref_center"],
        ref_std=params["ref_std"],
        switching_cost_scale=params["switching_cost_scale"],
    )
    agents = build_agents(scenario, portfolio["weights"], portfolio["directions"],
                          n_props=n_props, K=params["K"], seed=agent_seed)

    res = sim.run(
        rates=portfolio["rates"],
        fixed_features=portfolio["fixed_features"],
        competitor_features=portfolio["competitors"],
        agents_ref=agents["refs"],
        agents_w=agents["w"],
        agents_directions=agents["directions"],
        agents_switching_cost=agents["switching_costs"],
        agents_current_choice=agents["current_choice"],
        agents_to_properties=agents["agent_to_properties"],
        gumbel=sim.sample_gumbel(M, seed=gumbel_seed),
        churn_u=sim.sample_churn(M, seed=churn_seed),
        order=sim.sample_order(M, seed=order_seed),
    )
    return compute_moments(res["occupied_history"], portfolio["area"],
                           burn_in=params["burn_in"], tail_steps=params["tail_steps"])


def simulate_moments(params: dict, portfolio: dict, seeds) -> dict:
    """Average moments over seeds; key + '_se' holds the Monte Carlo standard error."""
    runs = [simulate_once(params, portfolio, s) for s in seeds]
    out = {}
    for k in MOMENT_KEYS:
        vals = np.array([r[k] for r in runs], dtype=float)
        vals = vals[~np.isnan(vals)]
        out[k] = float(vals.mean()) if len(vals) else float("nan")
        out[k + "_se"] = (float(vals.std(ddof=1) / np.sqrt(len(vals)))
                          if len(vals) > 1 else float("nan"))
    return out