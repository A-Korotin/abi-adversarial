import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from abi.inverse.grad import evolution_gradient
from abi.inverse.sampler import phi_to_vector


def evaluate_candidate(
        simulator,
        loss_fn,
        rates: np.ndarray,
        fixed_features: np.ndarray,
        competitor_features: np.ndarray,
        weights: np.ndarray,
        feature_directions: np.ndarray,
        agents: dict,
        gumbel=None,
) -> dict:
    result = simulator.run(
        rates=rates,
        fixed_features=fixed_features,
        competitor_features=competitor_features,
        agents_ref=agents["refs"],
        agents_w=agents["w"],
        agents_directions=agents["directions"],
        agents_switching_cost=agents["switching_costs"],
        agents_current_nest=agents["current_nest"],
        agents_current_inner=agents["current_inner"],
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


def train_adversarial(
        simulator,
        generator,
        discriminator,
        loss_fn,
        fixed_features: np.ndarray,
        competitor_features: np.ndarray,
        weights: np.ndarray,
        feature_directions: np.ndarray,
        build_agents_fn,
        outer_steps: int = 50,
        pop_size_xi: int = 4,
        pop_size_phi: int = 4,
        lr_g: float = 0.05,
        lr_d: float = 0.05,
) -> dict:
    history = {
        "gen_loss": [],
        "disc_loss": [],
        "occupancy": [],
        "mean_rate": [],
        "feasible": [],
        "rates": [],
        "phi": [],
    }

    M = discriminator.n_agents
    N_PROPS = generator.xi.shape[0]

    with ThreadPoolExecutor() as pool:
        for _ in (pbar := tqdm(range(outer_steps), desc="Portfolio optimisation")):
            # --- Discriminator step: find worst-case phi ---
            phi_pop = discriminator.sample_population(pop_size_phi, mirrored=True)
            xi_snapshot = generator.xi.copy()

            # CRN: same noise for all phi-candidates (they share rates)
            g_disc = simulator.sample_gumbel(M)

            disc_results = list(pool.map(
                lambda phi: evaluate_candidate(
                    simulator, loss_fn, xi_snapshot,
                    fixed_features, competitor_features,
                    weights, feature_directions,
                    build_agents_fn(phi, weights, feature_directions),
                    g_disc,
                ),
                phi_pop,
            ))
            disc_losses = np.array([r["loss"] for r in disc_results], dtype=np.float32)

            pop_phi_vec = np.array([phi_to_vector(phi) for phi in phi_pop], dtype=np.float32)
            grad_phi = evolution_gradient(
                discriminator.phi_vec, pop_phi_vec, disc_losses, discriminator.sigma
            )
            discriminator.update(-grad_phi, lr_d)

            worst_idx = int(np.argmax(disc_losses))
            worst_phi = phi_pop[worst_idx]

            # --- Generator step: optimize rates against worst-case phi ---
            # Build agents once — all xi-candidates face the same phi
            shared_agents = build_agents_fn(worst_phi, weights, feature_directions)
            xi_pop = generator.sample_population(pop_size_xi, mirrored=True)

            # CRN: same noise for all xi-candidates (they share agents and phi)
            g_gen = simulator.sample_gumbel(M)

            gen_results = list(pool.map(
                lambda xi: evaluate_candidate(
                    simulator, loss_fn, xi,
                    fixed_features, competitor_features,
                    weights, feature_directions,
                    shared_agents,
                    g_gen,
                ),
                xi_pop,
            ))
            gen_losses = np.array([r["loss"] for r in gen_results], dtype=np.float32)

            grad_xi = evolution_gradient(
                generator.xi, xi_pop, gen_losses, generator.sigma
            )
            generator.update(grad_xi, lr_g)

            best_idx = int(np.argmin(gen_losses))
            best_r = gen_results[best_idx]

            history["gen_loss"].append(float(gen_losses.mean()))
            history["disc_loss"].append(float(disc_losses.mean()))
            history["occupancy"].append(best_r["occupancy"])
            history["mean_rate"].append(best_r["mean_rate"])
            history["feasible"].append(loss_fn.is_feasible(best_r["occupancy"]))
            history["rates"].append(generator.xi.copy())
            history["phi"].append(discriminator.phi_vec.copy())

            pbar.set_postfix_str(
                f"loss={gen_losses.mean():.4f} "
                f"occ={best_r['occupancy']:.2%} "
                f"rate={best_r['mean_rate']:.3f} "
                f"{'OK' if history['feasible'][-1] else '--'}"
            )

    return history
