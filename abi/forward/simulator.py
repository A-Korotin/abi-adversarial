import numpy as np
from abi.core.functions import prospect_np

class VectorizedSimulator:
    def __init__(self, T: int, LR: float):
        self.T = T
        self.LR = LR
        self.rng = np.random.default_rng()

    @staticmethod
    def __find_equilibrium_step(history, window=20, eps=1e-2, patience=4):
        T, N = history.shape

        pad = np.vstack([np.zeros((1, N), dtype=history.dtype), np.cumsum(history, axis=0)])
        window_means = (pad[window:] - pad[:-window]) / window  # (T-window+1, N)

        shifts = np.max(np.abs(window_means[window:] - window_means[:-window]), axis=1)

        stable_count = 0
        for i, shift in enumerate(shifts, start=2 * window - 1):
            if shift < eps:
                stable_count += 1
                if stable_count >= patience:
                    eq_step = i - window + 1
                    return eq_step, window_means[eq_step]
            else:
                stable_count = 0

        return None, window_means[-1]
    
    def __gumbel_random_choice(self, utilities: np.array) -> int:
        U = self.rng.random(utilities.shape, dtype=np.float32)
        U = np.clip(U, 1e-8, 1 - 1e-8)
        gumbel_noise = -np.log(-np.log(U))
        return np.argmax(utilities + gumbel_noise, axis=1)

    def run(self, agents, providers):
        M = len(agents)
        P = len(providers)

        market_share_history = np.zeros((self.T, P), dtype=np.float32)

        provider_options = np.ascontiguousarray(
            np.stack([p.p for p in providers]).astype(np.float32)
        )

        refs = np.ascontiguousarray(np.stack([a.ref for a in agents]).astype(np.float32))
        current = np.array([a.current_provider for a in agents], dtype=np.int32)
        switch_cost = np.array([a.switching_cost for a in agents], dtype=np.float32)

        weights = np.asarray(agents[0].w, dtype=np.float32).reshape(-1)
        directions = np.asarray(agents[0].directions, dtype=np.float32).reshape(-1)

        for t in range(self.T):
            deltas = (provider_options[None, :, :] - refs[:, None, :]) * directions[None, None, :]
            activation = prospect_np(deltas)
            utilities = np.tensordot(activation, weights, axes=([2], [0]))  # (M, P)

            utilities -= switch_cost[:, None]

            valid = current >= 0
            rows = np.nonzero(valid)[0]
            utilities[rows, current[valid]] += switch_cost[valid]

            choices = self.__gumbel_random_choice(utilities)           

            counts = np.bincount(choices, minlength=P).astype(np.float32)
            market_share_history[t] = counts / M

            refs = (1 - self.LR) * refs + self.LR * provider_options[choices]
            current = choices.astype(np.int32)

        eq_step, eq_shares = VectorizedSimulator.__find_equilibrium_step(market_share_history)
        return eq_step, eq_shares, market_share_history
