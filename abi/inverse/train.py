import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from abi.inverse.grad import evolution_gradient
from abi.inverse.sampler import phi_to_vector


def evaluate_candidate(
        simulator,
        loss_fn,
        rates: np.ndarray,
        phi,
        fixed_features: np.ndarray,
        competitor_features: np.ndarray,
        weights: np.ndarray,
        feature_directions: np.ndarray,
        build_agents_fn,
) -> dict:
    agents = build_agents_fn(phi, weights, feature_directions)

    result = simulator.run(
        rates=rates,
        fixed_features=fixed_features,
        competitor_features=competitor_features,
        agents_ref=agents["refs"],
        agents_w=agents["w"],
        agents_directions=agents["directions"],
        agents_switching_cost=agents["switching_costs"],
        agents_current=agents["current"],
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

    for _ in (pbar := tqdm(range(outer_steps), desc="Portfolio optimisation")):
        phi_pop = discriminator.sample_population(pop_size_phi, mirrored=True)

        xi_snapshot = generator.xi.copy()
        with ThreadPoolExecutor() as pool:
            disc_results = list(pool.map(
                lambda phi: evaluate_candidate(
                    simulator, loss_fn, xi_snapshot, phi,
                    fixed_features, competitor_features,
                    weights, feature_directions, build_agents_fn,
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

        xi_pop = generator.sample_population(pop_size_xi, mirrored=True)

        with ThreadPoolExecutor() as pool:
            gen_results = list(pool.map(
                lambda xi: evaluate_candidate(
                    simulator, loss_fn, xi, worst_phi,
                    fixed_features, competitor_features,
                    weights, feature_directions, build_agents_fn,
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
