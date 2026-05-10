import numpy as np

def evolution_gradient(theta, population, losses, sigma):
    eps = (population - theta) / sigma

    # нормализация уменьшает дисперсию
    losses = (losses - losses.mean()) / (losses.std() + 1e-8)

    # делаем reshape автоматически под размерность eps
    reshape = (len(losses),) + (1,) * (eps.ndim - 1)
    losses = losses.reshape(reshape)

    grad = np.mean(losses * eps, axis=0)
    return grad
