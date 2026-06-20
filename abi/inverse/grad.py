import numpy as np


def evolution_gradient(
        theta: np.ndarray,
        population: np.ndarray,
        losses: np.ndarray,
        sigma: np.ndarray,
) -> np.ndarray:
    eps = (population - theta) / (sigma + 1e-8)

    losses = (losses - losses.mean()) / (losses.std() + 1e-8)

    reshape = (len(losses),) + (1,) * (eps.ndim - 1)
    losses = losses.reshape(reshape)

    return np.mean(losses * eps, axis=0)
