import numpy as np

class Generator:
    def __init__(self, base_xi: np.ndarray, sigma: np.ndarray):
        self.xi = np.asarray(base_xi, dtype=np.float32)      # (P, F)
        self.sigma = np.asarray(sigma, dtype=np.float32)     # (P, F)

    def sample_population(self, n: int, mirrored: bool = True) -> np.ndarray:
        shape = self.xi.shape  # (P, F)

        if mirrored:
            half = n // 2
            eps = np.random.normal(size=(half, *shape)).astype(np.float32)
            eps = np.vstack([eps, -eps])
            if n % 2 == 1:
                extra = np.random.normal(size=(1, *shape)).astype(np.float32)
                eps = np.vstack([eps, extra])
        else:
            eps = np.random.normal(size=(n, *shape)).astype(np.float32)

        return self.xi[None, :, :] + self.sigma[None, :, :] * eps

    def update(self, grad: np.ndarray, lr: float):
        self.xi -= lr * grad
        self.xi = np.clip(self.xi, 0.0, 1.0)
