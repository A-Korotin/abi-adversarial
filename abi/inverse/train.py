"""
Rate optimisation via Natural Evolution Strategies (NES).

Single-loop optimisation: sample a population of rate vectors, evaluate each
against a fixed market scenario (same agents, same Gumbel noise via CRN),
estimate the NES gradient, and update rates.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from abi.inverse.grad import evolution_gradient


def evaluate_candidate(
        simulator,
        loss_fn,
        rates: np.ndarray,
        fixed_features: np.ndarray,
        competitor_features: np.ndarray,
        agents: dict,
        gumbel=None,
) -> dict:
    """Run one simulation and compute loss."""
    result = simulator.run(
        rates=rates,
        fixed_features=fixed_features,
        competitor_features=competitor_features,
        agents_ref=agents["refs"],
        agents_w=agents["w"],
        agents_directions=agents["directions"],
        agents_switching_cost=agents["switching_costs"],
        agents_current_choice=agents["current_choice"],
        agents_to_properties=agents["agent_to_properties"],
        gumbel=gumbel,
    )

    loss = loss_fn(occupancy=result["occupancy"], mean_rate=result["mean_rate"])

    return {
        "loss": loss,
        "occupancy": result["occupancy"],
        "mean_rate": result["mean_rate"],
        "eq_step": result["eq_step"],
    }


def optimize_rates(
        simulator,
        generator,
        loss_fn,
        fixed_features: np.ndarray,
        competitor_features: np.ndarray,
        agents: dict,
        outer_steps: int = 50,
        pop_size: int = 50,
        lr: float = 0.05,
) -> dict:
    """Optimize rental rates via NES against a fixed market scenario.

    Parameters
    ----------
    simulator : RealEstateSimulator
    generator : RateGenerator — holds current rates xi, sigma, bounds.
    loss_fn : RealEstateLoss
    fixed_features : (N_PROPS, N_F-1) float32 — area, location_score per property.
    competitor_features : (N_COMP, N_F) float32
    agents : dict — pre-built agent arrays (fixed scenario).
    outer_steps : int — number of NES iterations.
    pop_size : int — population size per iteration.
    lr : float — learning rate for rate updates.

    Returns
    -------
    dict with history keys: gen_loss, occupancy, mean_rate, feasible, rates.
    """
    history = {
        "gen_loss": [],
        "occupancy": [],
        "mean_rate": [],
        "feasible": [],
        "rates": [],
    }

    M = agents["refs"].shape[0]

    with ThreadPoolExecutor() as pool:
        for _ in (pbar := tqdm(range(outer_steps), desc="Rate optimisation")):
            # Sample rate population (mirrored for variance reduction)
            xi_pop = generator.sample_population(pop_size, mirrored=True)

            # CRN: shared Gumbel noise across all candidates
            g = simulator.sample_gumbel(M)

            results = list(pool.map(
                lambda xi: evaluate_candidate(
                    simulator, loss_fn, xi,
                    fixed_features, competitor_features,
                    agents, g,
                ),
                xi_pop,
            ))
            losses = np.array([r["loss"] for r in results], dtype=np.float32)

            # NES gradient estimate and update
            grad = evolution_gradient(generator.xi, xi_pop, losses, generator.sigma)
            generator.update(grad, lr)

            best_idx = int(np.argmin(losses))
            best = results[best_idx]

            history["gen_loss"].append(float(losses.mean()))
            history["occupancy"].append(best["occupancy"])
            history["mean_rate"].append(best["mean_rate"])
            history["feasible"].append(loss_fn.is_feasible(best["occupancy"]))
            history["rates"].append(generator.xi.copy())

            pbar.set_postfix_str(
                f"loss={losses.mean():.4f} "
                f"occ={best['occupancy']:.2%} "
                f"rate={best['mean_rate']:.3f} "
                f"{'OK' if history['feasible'][-1] else '--'}"
            )

    return history
