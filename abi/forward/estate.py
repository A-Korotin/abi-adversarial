import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, inline='always')
def _prospect_utility(options, opt_idx, refs, ref_idx, directions, w):
    u = np.float32(0.0)
    for f in range(len(w)):
        d = (options[opt_idx, f] - refs[ref_idx, f]) * directions[f]
        if d >= np.float32(0.0):
            u += (d ** np.float32(0.8)) * w[f]
        else:
            u += np.float32(-2.25) * ((-d) ** np.float32(0.8)) * w[f]
    return u


@njit(cache=True, fastmath=True, nogil=True)
def _simulate_core(
        T,
        portfolio_options,        # (N_PROPS, N_F) float32
        competitor_options,       # (N_COMP, N_F) float32
        competitor_log_sizes,     # (N_COMP,) float32 — ln(S_k) per cluster
        outside_log_size,         # float32
        lambda_p,                 # float32 — nest dissimilarity for portfolio
        lambda_c,                 # float32 — nest dissimilarity for competitors
        demand_per_unit,          # float32 — M / total_supply_units
        directions,               # (N_F,) float32
        w,                        # (N_F,) float32
        market_mean,              # (N_F,) float32
        refs,                     # (M, N_F) float32, modified in-place
        current_nest,             # (M,) int32: 0=portfolio 1=comp 2=outside -1=init
        current_inner,            # (M,) int32: prop_idx or comp_idx, -1 otherwise
        switching_costs,          # (M,) float32
        agent_to_properties,      # (M, K) int32
        lr_ref,
        ref_market_weight,
        outside_utility,          # float32 (base utility before log-size)
        gumbel_top,               # (T, M, 3) float32 — nest-level noise
        gumbel_inner_p,           # (T, M, K) float32 — within-portfolio noise
        gumbel_inner_c,           # (T, M, N_COMP) float32 — within-competitor noise
        disable_early_exit,       # int32: 1 = always run all T steps
        lease_capacity,           # float32: chose_count[j] >= this → property leased (M-independent)
):
    M = refs.shape[0]
    N_PROPS = portfolio_options.shape[0]
    N_COMP = competitor_options.shape[0]
    N_F = portfolio_options.shape[1]
    K = agent_to_properties.shape[1]

    portfolio_share_history = np.zeros(T, dtype=np.float32)
    mean_rate_history = np.zeros(T, dtype=np.float32)
    first_leased = np.full(N_PROPS, -1, dtype=np.int32)

    lp = np.float32(lambda_p)
    lc = np.float32(lambda_c)
    lr = np.float32(lr_ref)
    rmw = np.float32(ref_market_weight)
    # outside inclusive value = outside_utility + outside_log_size (single element, no inner noise)
    I_out_base = np.float32(outside_utility) + np.float32(outside_log_size)
    dpu = np.float32(demand_per_unit)

    total_area = np.float32(0.0)
    for j in range(N_PROPS):
        total_area += portfolio_options[j, 1]

    # Reusable buffers (one agent at a time, sequential)
    v_p_buf = np.empty(K, dtype=np.float32)
    v_c_buf = np.empty(N_COMP, dtype=np.float32)
    chose_count = np.zeros(N_PROPS, dtype=np.int32)
    chosen_nest_arr = np.empty(M, dtype=np.int32)
    chosen_inner_arr = np.empty(M, dtype=np.int32)

    # Early-exit (window=20, eps=1e-2, patience=4)
    eq_window = 20
    eq_eps = np.float32(1e-2)
    stable_count = 0
    t_used = T

    for t in range(T):
        for j in range(N_PROPS):
            chose_count[j] = np.int32(0)

        # --- Per-agent nested logit choice ---
        for i in range(M):
            sc = switching_costs[i]
            cn = current_nest[i]

            # Utilities for K visible portfolio properties
            for k in range(K):
                v_p_buf[k] = _prospect_utility(
                    portfolio_options, agent_to_properties[i, k],
                    refs, i, directions, w
                )

            # Utilities for N_COMP competitor clusters
            for k in range(N_COMP):
                v_c_buf[k] = _prospect_utility(
                    competitor_options, k, refs, i, directions, w
                )

            # Inclusive value: portfolio nest
            # I_p = max_vp + lp * log(Σ exp((v_p - max_vp) / lp))
            # + lp * log(N_PROPS / K)  ← McFadden size correction for K-out-of-N sampling
            max_vp = v_p_buf[0]
            for k in range(1, K):
                if v_p_buf[k] > max_vp:
                    max_vp = v_p_buf[k]
            s_p = np.float32(0.0)
            for k in range(K):
                s_p += np.exp((v_p_buf[k] - max_vp) / lp)
            I_p = max_vp + lp * (np.log(s_p) + np.log(np.float32(N_PROPS) / np.float32(K)))

            # Inclusive value: competitor nest
            # I_c = max_adj + lc * log(Σ exp((v_c + log_s - max_adj) / lc))
            max_vc = v_c_buf[0] + competitor_log_sizes[0]
            for k in range(1, N_COMP):
                adj = v_c_buf[k] + competitor_log_sizes[k]
                if adj > max_vc:
                    max_vc = adj
            s_c = np.float32(0.0)
            for k in range(N_COMP):
                s_c += np.exp((v_c_buf[k] + competitor_log_sizes[k] - max_vc) / lc)
            I_c = max_vc + lc * np.log(s_c)

            # Apply switching cost at the nest level (not within-nest)
            I_p_adj = I_p
            I_c_adj = I_c
            I_out_adj = I_out_base
            if cn >= np.int32(0):
                if cn != np.int32(0):
                    I_p_adj = I_p - sc
                if cn != np.int32(1):
                    I_c_adj = I_c - sc
                if cn != np.int32(2):
                    I_out_adj = I_out_base - sc

            # Top-level Gumbel-max over the 3 nests
            g0 = I_p_adj + gumbel_top[t, i, 0]
            g1 = I_c_adj + gumbel_top[t, i, 1]
            g2 = I_out_adj + gumbel_top[t, i, 2]

            if g0 >= g1 and g0 >= g2:
                nest = np.int32(0)
            elif g1 >= g2:
                nest = np.int32(1)
            else:
                nest = np.int32(2)

            # Within-nest Gumbel-max
            if nest == np.int32(0):
                best_k = np.int32(0)
                best_v = v_p_buf[0] / lp + gumbel_inner_p[t, i, 0]
                for k in range(1, K):
                    v = v_p_buf[k] / lp + gumbel_inner_p[t, i, k]
                    if v > best_v:
                        best_v = v
                        best_k = np.int32(k)
                inner = agent_to_properties[i, best_k]
                chose_count[inner] += np.int32(1)
            elif nest == np.int32(1):
                best_k = np.int32(0)
                best_v = (v_c_buf[0] + competitor_log_sizes[0]) / lc + gumbel_inner_c[t, i, 0]
                for k in range(1, N_COMP):
                    v = (v_c_buf[k] + competitor_log_sizes[k]) / lc + gumbel_inner_c[t, i, k]
                    if v > best_v:
                        best_v = v
                        best_k = np.int32(k)
                inner = best_k
            else:
                inner = np.int32(-1)

            chosen_nest_arr[i] = nest
            chosen_inner_arr[i] = inner

        # --- Continuous occupancy: occ_j = min(1, chose_count[j] / demand_per_unit) ---
        occupied_area = np.float32(0.0)
        rate_weighted = np.float32(0.0)
        for j in range(N_PROPS):
            if first_leased[j] < 0 and np.float32(chose_count[j]) >= lease_capacity:
                first_leased[j] = t
            frac = np.float32(chose_count[j]) / dpu
            if frac > np.float32(1.0):
                frac = np.float32(1.0)
            area_j = portfolio_options[j, 1]
            occupied_area += area_j * frac
            rate_weighted += portfolio_options[j, 0] * area_j * frac
        if total_area > np.float32(0.0):
            portfolio_share_history[t] = occupied_area / total_area
        if occupied_area > np.float32(0.0):
            mean_rate_history[t] = rate_weighted / occupied_area

        # --- Update state and references ---
        for i in range(M):
            nest = chosen_nest_arr[i]
            inner = chosen_inner_arr[i]
            current_nest[i] = nest
            if nest != np.int32(2):
                current_inner[i] = inner

            if nest == np.int32(0):
                for f in range(N_F):
                    refs[i, f] = (
                        (np.float32(1.0) - lr) * refs[i, f]
                        + lr * (rmw * market_mean[f]
                                + (np.float32(1.0) - rmw) * portfolio_options[inner, f])
                    )
            elif nest == np.int32(1):
                for f in range(N_F):
                    refs[i, f] = (
                        (np.float32(1.0) - lr) * refs[i, f]
                        + lr * (rmw * market_mean[f]
                                + (np.float32(1.0) - rmw) * competitor_options[inner, f])
                    )

        # --- Early equilibrium exit ---
        if disable_early_exit == 0 and t >= 2 * eq_window - 1:
            curr_s = np.float32(0.0)
            prev_s = np.float32(0.0)
            for k in range(eq_window):
                curr_s += portfolio_share_history[t - k]
                prev_s += portfolio_share_history[t - eq_window - k]
            if abs(curr_s - prev_s) < eq_eps * np.float32(eq_window):
                stable_count += 1
                if stable_count >= 4:
                    t_used = t + 1
                    break
            else:
                stable_count = 0

    return portfolio_share_history, mean_rate_history, t_used, first_leased


_rng = np.random.default_rng()


def _gumbel_noise(shape):
    E = _rng.standard_exponential(size=shape, dtype=np.float32)
    np.clip(E, 1e-30, None, out=E)
    np.log(E, out=E)
    np.negative(E, out=E)
    return E


class RealEstateSimulator:

    def __init__(
            self,
            T: int,
            K: int = 8,
            lr_ref: float = 0.05,
            ref_market_weight: float = 0.3,
            outside_utility: float = 0.0,
            outside_log_size: float = None,  # ln(outside_size); default ln(100)
            lambda_portfolio: float = 0.6,
            lambda_comp: float = 0.6,
            competitor_log_sizes: np.ndarray = None,
            competitor_sizes: np.ndarray = None,
            outside_size: float = 100.0,
    ):
        self.T = T
        self.K = K
        self.lr_ref = lr_ref
        self.ref_market_weight = ref_market_weight
        self.outside_utility = outside_utility
        self.lambda_portfolio = float(lambda_portfolio)
        self.lambda_comp = float(lambda_comp)

        if outside_log_size is not None:
            self.outside_log_size = float(outside_log_size)
        else:
            self.outside_log_size = float(np.log(outside_size))

        if competitor_log_sizes is not None:
            self.competitor_log_sizes = np.asarray(competitor_log_sizes, dtype=np.float32)
        elif competitor_sizes is not None:
            self.competitor_log_sizes = np.log(np.asarray(competitor_sizes, dtype=np.float32))
        else:
            from abi.inverse.builders import DEFAULT_COMPETITOR_SIZES
            self.competitor_log_sizes = np.log(DEFAULT_COMPETITOR_SIZES).astype(np.float32)

        self._comp_supply = float(np.sum(np.exp(self.competitor_log_sizes)))
        self._out_supply = float(np.exp(self.outside_log_size))

    def sample_gumbel(self, M: int):
        T, K = self.T, self.K
        N_COMP = len(self.competitor_log_sizes)
        return (
            _gumbel_noise((T, M, 3)),
            _gumbel_noise((T, M, K)),
            _gumbel_noise((T, M, N_COMP)),
        )

    def run(
            self,
            rates: np.ndarray,
            fixed_features: np.ndarray,
            competitor_features: np.ndarray,
            agents_ref: np.ndarray,
            agents_w: np.ndarray,
            agents_directions: np.ndarray,
            agents_switching_cost: np.ndarray,
            agents_current_nest: np.ndarray,
            agents_current_inner: np.ndarray,
            agents_to_properties: np.ndarray,
            gumbel=None,
            disable_early_exit: bool = False,
            lease_capacity: float = 1.0,
    ):
        portfolio_options = np.concatenate(
            [rates[:, None], fixed_features], axis=1
        ).astype(np.float32)
        competitor_options = np.asarray(competitor_features, dtype=np.float32)
        market_mean = competitor_options.mean(axis=0).astype(np.float32)

        N_PROPS = portfolio_options.shape[0]
        M = agents_ref.shape[0]

        total_supply = N_PROPS + self._comp_supply + self._out_supply
        demand_per_unit = np.float32(M / total_supply)

        if gumbel is None:
            gumbel = self.sample_gumbel(M)
        gumbel_top, gumbel_inner_p, gumbel_inner_c = gumbel

        refs = agents_ref.copy().astype(np.float32)
        current_nest = agents_current_nest.copy().astype(np.int32)
        current_inner = agents_current_inner.copy().astype(np.int32)
        w = np.asarray(agents_w, dtype=np.float32)
        directions = np.asarray(agents_directions, dtype=np.float32)
        switching_costs = agents_switching_cost.astype(np.float32)
        agent_to_properties = np.asarray(agents_to_properties, dtype=np.int32)

        portfolio_share_history, mean_rate_history, t_used, first_leased = _simulate_core(
            self.T,
            portfolio_options,
            competitor_options,
            self.competitor_log_sizes,
            np.float32(self.outside_log_size),
            np.float32(self.lambda_portfolio),
            np.float32(self.lambda_comp),
            demand_per_unit,
            directions,
            w,
            market_mean,
            refs,
            current_nest,
            current_inner,
            switching_costs,
            agent_to_properties,
            self.lr_ref,
            self.ref_market_weight,
            np.float32(self.outside_utility),
            gumbel_top.astype(np.float32),
            gumbel_inner_p.astype(np.float32),
            gumbel_inner_c.astype(np.float32),
            np.int32(1 if disable_early_exit else 0),
            np.float32(lease_capacity),
        )

        window = min(20, t_used)
        eq_occupancy = float(portfolio_share_history[t_used - window:t_used].mean())
        eq_mean_rate = float(mean_rate_history[t_used - window:t_used].mean())

        return {
            "eq_step": t_used,
            "occupancy": eq_occupancy,
            "mean_rate": eq_mean_rate,
            "portfolio_share_history": portfolio_share_history[:t_used],
            "mean_rate_history": mean_rate_history[:t_used],
            "first_leased_step": first_leased,
        }
