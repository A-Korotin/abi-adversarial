import numpy as np


class RateGenerator:

    def __init__(
            self,
            initial_rates: np.ndarray,
            sigma: np.ndarray,
            rate_min: float = 0.0,
            rate_max: float = 1.0,
    ):
        self.xi = np.asarray(initial_rates, dtype=np.float32).copy()
        self.sigma = np.asarray(sigma, dtype=np.float32)
        self.rate_min = rate_min
        self.rate_max = rate_max

    def sample_population(self, n: int, mirrored: bool = True) -> np.ndarray:
        N = self.xi.shape[0]

        if mirrored:
            half = n // 2
            eps = np.random.normal(size=(half, N)).astype(np.float32)
            eps = np.vstack([eps, -eps])
            if n % 2 == 1:
                eps = np.vstack([eps, np.random.normal(size=(1, N)).astype(np.float32)])
        else:
            eps = np.random.normal(size=(n, N)).astype(np.float32)

        population = self.xi[None, :] + self.sigma[None, :] * eps
        return np.clip(population, self.rate_min, self.rate_max)

    def update(self, grad: np.ndarray, lr: float):
        self.xi -= lr * grad
        self.xi = np.clip(self.xi, self.rate_min, self.rate_max)
