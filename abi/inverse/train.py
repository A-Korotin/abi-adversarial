import numpy as np
from tqdm import tqdm

from abi.inverse.grad import evolution_gradient
from abi.inverse.sampler import phi_to_vector


def evaluate_candidate(simulator, loss_fn, xi, phi, build_providers, build_agents):
    providers = build_providers(xi)
    agents = build_agents(phi)

    eq_step, eq_shares, history = simulator.run(agents, providers)
    loss = loss_fn(eq_shares)

    return loss, eq_step, eq_shares, history


def train_adversarial_simple(
        simulator,
        generator,
        discriminator,
        loss_fn,
        build_providers,
        build_agents,
        outer_steps: int = 50,
        pop_size_xi: int = 4,
        pop_size_phi: int = 4,
        lr_g: float = 0.05,
        lr_d: float = 0.05,
):
    history = {
        "gen_loss": [],
        "disc_loss": [],
        "xi": [],
        "phi": [],
        "eq_share": [],
    }

    for _ in (pbar := tqdm(range(outer_steps), "Inverse train")):
        # находим неприятный сценарий для текущего xi
        phi_pop = discriminator.sample_population(pop_size_phi, mirrored=True)

        disc_losses = []
        for phi in phi_pop:
            loss, _, _, _ = evaluate_candidate(
                simulator, loss_fn, generator.xi, phi, build_providers, build_agents
            )
            disc_losses.append(loss)

        disc_losses = np.asarray(disc_losses, dtype=np.float32)
        pop_phi_vec = np.array([phi_to_vector(phi) for phi in phi_pop], dtype=np.float32)

        grad_phi = evolution_gradient(
            discriminator.phi_vec,
            pop_phi_vec,
            disc_losses,
            discriminator.sigma
        )

        # дискриминатор максимизирует loss
        discriminator.update(-grad_phi, lr_d)

        # минимизируем loss на текущем худшем phi
        # берём лучший (хуже всего для нас) phi из этой маленькой популяции
        worst_idx = int(np.argmax(disc_losses))
        worst_phi = phi_pop[worst_idx]

        xi_pop = generator.sample_population(pop_size_xi, mirrored=True)

        gen_losses = []
        best_eq = None

        for xi in xi_pop:
            loss, eq_step, eq_shares, _ = evaluate_candidate(
                simulator, loss_fn, xi, worst_phi, build_providers, build_agents
            )
            gen_losses.append(loss)

            if best_eq is None or loss < min(gen_losses):
                best_eq = eq_shares

        gen_losses = np.asarray(gen_losses, dtype=np.float32)

        grad_xi = evolution_gradient(
            generator.xi,
            xi_pop,
            gen_losses,
            generator.sigma
        )

        generator.update(grad_xi, lr_g)

        history["gen_loss"].append(float(gen_losses.mean()))
        history["disc_loss"].append(float(disc_losses.mean()))
        history["xi"].append(generator.xi.copy())
        history["phi"].append(discriminator.phi_vec.copy())
        history["eq_share"].append(best_eq.copy() if best_eq is not None else None)

        pbar.set_postfix_str(f"gen_loss={history['gen_loss'][-1]:.4f} disc_loss={history['disc_loss'][-1]:.4f}")

    return history
