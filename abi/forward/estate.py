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
        portfolio_options,    # (N_PROPS, N_F) float32
        competitor_options,   # (N_COMP, N_F) float32
        directions,           # (N_F,) float32
        w,                    # (N_F,) float32
        market_mean,          # (N_F,) float32
        refs,                 # (M, N_F) float32  — modified in place
        current,              # (M,) int32 — option idx: 0=own, 1..N_COMP=comp_k, N_COMP+1=outside, -1=init
        switching_costs,      # (M,) float32
        agent_to_property,    # (M,) int32 — which portfolio property each agent evaluates
        lr_ref,               # float64
        ref_market_weight,    # float64
        outside_utility,      # float64
        gumbel,               # (T, M, N_COMP+2) float32
):
    M = refs.shape[0]
    N_PROPS = portfolio_options.shape[0]
    N_COMP = competitor_options.shape[0]
    N_F = portfolio_options.shape[1]
    N_OUT = np.int32(N_COMP + 1)  # outside option index in choice set

    portfolio_share_history = np.zeros(T, dtype=np.float32)
    mean_rate_history = np.zeros(T, dtype=np.float32)

    lr = np.float32(lr_ref)
    rmw = np.float32(ref_market_weight)
    out_u = np.float32(outside_utility)

    # Precompute total portfolio area (fixed throughout simulation)
    total_area = np.float32(0.0)
    for j in range(N_PROPS):
        total_area += portfolio_options[j, 1]

    # Allocate working buffers once
    own_utility = np.empty(M, dtype=np.float32)
    competitor_utils = np.empty((M, N_COMP), dtype=np.float32)
    chose_portfolio = np.empty(M, dtype=np.bool_)
    chose_outside = np.empty(M, dtype=np.bool_)
    comp_chosen_idx = np.empty(M, dtype=np.int32)
    property_choices = np.empty(M, dtype=np.int32)
    chose_count = np.zeros(N_PROPS, dtype=np.int32)

    # Early-exit state (window=20, eps=1e-2, patience=4)
    eq_window = 20
    eq_eps = np.float32(1e-2)
    stable_count = 0
    t_used = T

    for t in range(T):
        # --- Own property utility for each agent ---
        for i in range(M):
            own_utility[i] = _prospect_utility(
                portfolio_options, agent_to_property[i], refs, i, directions, w
            )

        # --- Competitor utilities (M x N_COMP) ---
        for i in range(M):
            for k in range(N_COMP):
                competitor_utils[i, k] = _prospect_utility(
                    competitor_options, k, refs, i, directions, w
                )

        # --- Choice: own_property | comp_1..N_COMP | outside ---
        for i in range(M):
            sc = switching_costs[i]
            cur = current[i]
            have_cur = cur >= np.int32(0)

            # Option 0: own property
            sc_adj = np.float32(0.0) if (not have_cur or cur == np.int32(0)) else -sc
            best_val = own_utility[i] + sc_adj + gumbel[t, i, 0]
            best = np.int32(0)

            # Options 1..N_COMP: competitors
            for k in range(N_COMP):
                opt = np.int32(k + 1)
                sc_adj = np.float32(0.0) if (not have_cur or cur == opt) else -sc
                v = competitor_utils[i, k] + sc_adj + gumbel[t, i, opt]
                if v > best_val:
                    best_val = v
                    best = opt

            # Option N_COMP+1: outside
            sc_adj = np.float32(0.0) if (not have_cur or cur == N_OUT) else -sc
            v = out_u + sc_adj + gumbel[t, i, N_OUT]
            if v > best_val:
                best = N_OUT

            current[i] = best

            if best == np.int32(0):
                chose_portfolio[i] = True
                chose_outside[i] = False
                property_choices[i] = agent_to_property[i]
            elif best == N_OUT:
                chose_portfolio[i] = False
                chose_outside[i] = True
                property_choices[i] = np.int32(-1)
            else:
                chose_portfolio[i] = False
                chose_outside[i] = False
                comp_chosen_idx[i] = best - np.int32(1)
                property_choices[i] = np.int32(-1)

        # --- Per-property occupancy (area-weighted) and mean rate over occupied ---
        for j in range(N_PROPS):
            chose_count[j] = np.int32(0)
        for i in range(M):
            if chose_portfolio[i]:
                chose_count[property_choices[i]] += np.int32(1)
        occupied_area = np.float32(0.0)
        rate_sum = np.float32(0.0)
        n_occupied = np.int32(0)
        for j in range(N_PROPS):
            if chose_count[j] > np.int32(0):
                occupied_area += portfolio_options[j, 1]
                rate_sum += portfolio_options[j, 0]
                n_occupied += np.int32(1)
        portfolio_share_history[t] = occupied_area / total_area
        if n_occupied > np.int32(0):
            mean_rate_history[t] = rate_sum / np.float32(n_occupied)

        # --- Reference update ---
        for i in range(M):
            if not chose_outside[i]:
                if chose_portfolio[i]:
                    prop = property_choices[i]
                    for f in range(N_F):
                        refs[i, f] = (
                            (np.float32(1.0) - lr) * refs[i, f]
                            + lr * (rmw * market_mean[f]
                                    + (np.float32(1.0) - rmw) * portfolio_options[prop, f])
                        )
                else:
                    comp = comp_chosen_idx[i]
                    for f in range(N_F):
                        refs[i, f] = (
                            (np.float32(1.0) - lr) * refs[i, f]
                            + lr * (rmw * market_mean[f]
                                    + (np.float32(1.0) - rmw) * competitor_options[comp, f])
                        )

        # --- Early equilibrium exit ---
        if t >= 2 * eq_window - 1:
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

    return portfolio_share_history, mean_rate_history, t_used


_rng = np.random.default_rng()  # PCG64 — 2-3x faster than legacy MT


def _gumbel_noise(shape):
    # Gumbel(0,1) = -log(Exp(1)); dtype=float32 skips the float64 intermediate
    E = _rng.standard_exponential(size=shape, dtype=np.float32)
    np.clip(E, 1e-30, None, out=E)  # prevent log(0) for subnormal float32 draws
    np.log(E, out=E)
    np.negative(E, out=E)
    return E


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
            agents_to_property: np.ndarray,
            gumbel: np.ndarray = None,
    ):
        portfolio_options = np.concatenate(
            [rates[:, None], fixed_features], axis=1
        ).astype(np.float32)
        competitor_options = np.asarray(competitor_features, dtype=np.float32)

        market_mean = competitor_options.mean(axis=0).astype(np.float32)

        M = agents_ref.shape[0]
        N_COMP = competitor_options.shape[0]

        if gumbel is None:
            gumbel = _gumbel_noise((self.T, M, N_COMP + 2))

        refs = agents_ref.copy().astype(np.float32)
        current = agents_current.copy().astype(np.int32)
        w = np.asarray(agents_w, dtype=np.float32)
        directions = np.asarray(agents_directions, dtype=np.float32)
        switching_costs = agents_switching_cost.astype(np.float32)
        agent_to_property = np.asarray(agents_to_property, dtype=np.int32)

        portfolio_share_history, mean_rate_history, t_used = _simulate_core(
            self.T,
            portfolio_options,
            competitor_options,
            directions,
            w,
            market_mean,
            refs,
            current,
            switching_costs,
            agent_to_property,
            self.lr_ref,
            self.ref_market_weight,
            self.outside_utility,
            gumbel,
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
        }
