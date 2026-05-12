import numpy as np
from abi.core.functions import prospect_np


class RealEstateSimulator:

    def __init__(
        self,
        T: int,
        lr_ref: float = 0.05,
        ref_market_weight: float = 0.3,
        outside_utility: float = 0.0,
    ):
        self.T = T
        self.lr_ref = lr_ref
        self.ref_market_weight = ref_market_weight
        self.outside_utility = outside_utility

    @staticmethod
    def _find_equilibrium(history: np.ndarray, window: int = 20, eps: float = 1e-2, patience: int = 4):
        T, N = history.shape
        if T < 2 * window:
            return None, history[-1]

        pad = np.vstack([np.zeros((1, N), dtype=history.dtype), np.cumsum(history, axis=0)])
        window_means = (pad[window:] - pad[:-window]) / window

        if len(window_means) <= window:
            return None, window_means[-1]

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

    @staticmethod
    def _gumbel_choice(utilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        U = rng.random(utilities.shape, dtype=np.float32)
        U = np.clip(U, 1e-8, 1 - 1e-8)
        gumbel_noise = -np.log(-np.log(U))
        return np.argmax(utilities + gumbel_noise, axis=1)

    @staticmethod
    def _inclusive_value(utilities: np.ndarray) -> np.ndarray:
        """Normalized log-sum-exp over portfolio; eliminates set-size bias."""
        u_max = utilities.max(axis=1, keepdims=True)
        lse = u_max[:, 0] + np.log(np.exp(utilities - u_max).sum(axis=1))
        return (lse - np.log(utilities.shape[1])).astype(np.float32)

    def run(
        self,
        rates: np.ndarray,
        fixed_features: np.ndarray,
        competitor_features: np.ndarray,
        agents_ref: np.ndarray,
        agents_w: np.ndarray,
        agents_directions: np.ndarray,
        agents_switching_cost: np.ndarray,
        agents_current: np.ndarray,
    ):
        M = agents_ref.shape[0]
        N_PROPS = rates.shape[0]
        N_COMP = competitor_features.shape[0]

        portfolio_options = np.concatenate(
            [rates[:, None], fixed_features], axis=1
        ).astype(np.float32)  # (N_PROPS, N_FEATURES)
        competitor_options = np.asarray(competitor_features, dtype=np.float32)  # (N_COMP, N_FEATURES)

        N_FEATURES = portfolio_options.shape[1]

        market_mean = np.vstack([portfolio_options, competitor_options]).mean(axis=0)

        current = agents_current.copy().astype(np.int32)
        refs = agents_ref.copy().astype(np.float32)
        directions = agents_directions.astype(np.float32)
        w = agents_w[0]

        portfolio_share_history = np.zeros(self.T, dtype=np.float32)
        mean_rate_history = np.zeros(self.T, dtype=np.float32)

        rng = np.random.default_rng()

        for t in range(self.T):
            portfolio_utils = np.zeros((M, N_PROPS), dtype=np.float32)
            for f in range(N_FEATURES):
                d = (portfolio_options[:, f] - refs[:, f:f+1]) * directions[f]
                portfolio_utils += prospect_np(d) * w[f]

            competitor_utils = np.zeros((M, N_COMP), dtype=np.float32)
            for f in range(N_FEATURES):
                d = (competitor_options[:, f] - refs[:, f:f+1]) * directions[f]
                competitor_utils += prospect_np(d) * w[f]

            stage1 = np.empty((M, N_COMP + 2), dtype=np.float32)
            stage1[:, 0] = self._inclusive_value(portfolio_utils)
            stage1[:, 1:N_COMP + 1] = competitor_utils
            stage1[:, N_COMP + 1] = self.outside_utility

            stage1_choices = self._gumbel_choice(stage1, rng)

            chose_portfolio = (stage1_choices == 0)
            chose_outside   = (stage1_choices == N_COMP + 1)
            chose_comp_mask = ~chose_portfolio & ~chose_outside
            comp_chosen_idx = (stage1_choices - 1).clip(0, N_COMP - 1)

            occupancy_t = float(chose_portfolio.sum()) / N_PROPS
            portfolio_share_history[t] = occupancy_t

            property_choices = np.full(M, -1, dtype=np.int32)

            if chose_portfolio.any():
                port_u = portfolio_utils[chose_portfolio].copy()

                in_portfolio_now = (current >= 0) & (current < N_PROPS)
                port_u -= agents_switching_cost[chose_portfolio, None]

                cp_idx = np.where(chose_portfolio)[0]
                prev_in = in_portfolio_now[cp_idx]
                if prev_in.any():
                    local = np.where(prev_in)[0]
                    prev_props = current[cp_idx[prev_in]]
                    prev_costs = agents_switching_cost[cp_idx[prev_in]]
                    port_u[local, prev_props] += prev_costs

                prop = self._gumbel_choice(port_u, rng)
                property_choices[chose_portfolio] = prop

            if chose_portfolio.any():
                mean_rate_history[t] = rates[property_choices[chose_portfolio]].mean()

            active = ~chose_outside
            if active.any():
                chosen_feats = np.zeros((M, N_FEATURES), dtype=np.float32)

                if chose_portfolio.any():
                    chosen_feats[chose_portfolio] = portfolio_options[
                        property_choices[chose_portfolio]
                    ]
                if chose_comp_mask.any():
                    chosen_feats[chose_comp_mask] = competitor_options[
                        comp_chosen_idx[chose_comp_mask]
                    ]

                refs[active] = (
                    (1 - self.lr_ref) * refs[active]
                    + self.lr_ref * (
                        self.ref_market_weight * market_mean[None, :]
                        + (1 - self.ref_market_weight) * chosen_feats[active]
                    )
                )

            current[:] = property_choices

        eq_step, eq_shares = RealEstateSimulator._find_equilibrium(
            portfolio_share_history[:, None]
        )

        eq_occupancy = float(eq_shares[0]) if eq_shares is not None else float(portfolio_share_history[-1])
        eq_mean_rate = float(mean_rate_history[-20:].mean())

        return {
            "eq_step": eq_step,
            "occupancy": eq_occupancy,
            "mean_rate": eq_mean_rate,
            "portfolio_share_history": portfolio_share_history,
            "mean_rate_history": mean_rate_history,
        }
