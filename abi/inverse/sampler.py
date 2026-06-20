from dataclasses import dataclass

import numpy as np


def softplus(x: float) -> float:
    return float(np.log1p(np.exp(x)))


def inverse_softplus(x: float) -> float:
    return float(np.log(np.exp(x) - 1))


@dataclass
class MarketScenario:
    n_agents: int
    ref_shift: np.ndarray  # shape (N_FEATURES,)
    ref_std: float
    switching_cost_scale: float

    def __post_init__(self):
        self.ref_shift = np.asarray(self.ref_shift, dtype=np.float32)


def phi_to_vector(phi: MarketScenario) -> np.ndarray:
    return np.concatenate([
        phi.ref_shift,
        np.array([
            inverse_softplus(phi.ref_std),
            inverse_softplus(phi.switching_cost_scale),
        ], dtype=np.float32)
    ]).astype(np.float32)


def vector_to_phi(vec: np.ndarray, n_agents: int, n_features: int = 3) -> MarketScenario:
    vec = np.asarray(vec, dtype=np.float32)
    ref_shift = vec[:n_features]
    ref_std = softplus(float(vec[n_features])) + 1e-4
    switching_cost_scale = softplus(float(vec[n_features + 1])) + 1e-4
    return MarketScenario(
        n_agents=n_agents,
        ref_shift=ref_shift.copy(),
        ref_std=ref_std,
        switching_cost_scale=switching_cost_scale,
    )


class ScenarioSampler:

    def __init__(
            self,
            base_phi: MarketScenario,
            sigma: np.ndarray,
            n_agents: int,
            n_features: int = 3,
    ):
        self.phi = base_phi
        self.sigma = np.asarray(sigma, dtype=np.float32)
        self.n_agents = n_agents
        self.n_features = n_features
        self.phi_vec = phi_to_vector(base_phi)

    def sample_population(self, n: int, mirrored: bool = True):
        d = self.phi_vec.shape[0]

        if mirrored:
            half = n // 2
            eps = np.random.normal(size=(half, d)).astype(np.float32)
            eps = np.vstack([eps, -eps])
            if n % 2 == 1:
                eps = np.vstack([eps, np.random.normal(size=(1, d)).astype(np.float32)])
        else:
            eps = np.random.normal(size=(n, d)).astype(np.float32)

        samples = self.phi_vec[None, :] + self.sigma[None, :] * eps
        return [vector_to_phi(s, self.n_agents, self.n_features) for s in samples]

    def update(self, grad: np.ndarray, lr: float):
        self.phi_vec -= lr * grad
        self.phi = vector_to_phi(self.phi_vec, self.n_agents, self.n_features)
