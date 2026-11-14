"""
Real-estate market simulator — flat MNL with prospect-theory utility.

Choice model (per agent, per step):
    Each agent chooses among K visible portfolio properties + N_COMP competitor
    clusters + 1 outside option via Gumbel-max (= sampling from a multinomial
    logit).

    Utility of an option:
        portfolio property j :  v = beta * prospect_utility(features_j, ref_i)
        competitor cluster k :  v = beta * prospect_utility(features_k, ref_i)
                                    + competitor_log_size[k]
        outside option       :  v = outside_utility + outside_log_size

    Switching cost: subtract sc_i from every option EXCEPT the agent's current
    choice.  "Moving anywhere else costs sc."

    Gumbel-max: add iid Gumbel(0,1) noise to each v; pick argmax.

Dynamics:
    Reference update (EMA to chosen option):
        ref_i ← (1 - lr_ref) * ref_i + lr_ref * chosen_features
    Only updated when the agent chooses a portfolio property or competitor
    (not outside).

    Leases and exclusive occupancy:
        A portfolio property hosts at most one tenant at a time.  An agent who
        leases a portfolio property or a competitor unit is locked in; each
        step the lease expires with probability churn_prob (geometric lease
        term with mean 1 / churn_prob steps).  On expiry the agent re-decides:
        renew (stay, no switching cost) or move (pay switching cost).  Agents
        on the outside option search every step and pay no switching cost to
        enter the market.  Searchers act in a random order each step; a
        property taken earlier in the step is unavailable to later searchers.
        Competitor clusters aggregate many units and are not capacity-limited.

    Occupancy and mean_rate are area-weighted over the portfolio.
"""

import numpy as np
from numba import njit


# ── Prospect-theory utility (per-option, inlined by numba) ──────────────────

@njit(cache=True, fastmath=True, inline='always')
def _prospect_utility(options, opt_idx, refs, ref_idx, directions, w):
    """Prospect-theory utility of option opt_idx for agent ref_idx.

    For each feature f:
        delta = (option[f] - ref[f]) * direction[f]
        if delta >= 0:  gain  →  delta^0.8 * w[f]
        if delta <  0:  loss  → -2.25 * (-delta)^0.8 * w[f]

    Parameters (alpha=0.8, lambda=2.25) follow Tversky & Kahneman (1992).
    """
    u = np.float32(0.0)
    for f in range(len(w)):
        d = (options[opt_idx, f] - refs[ref_idx, f]) * directions[f]
        if d >= np.float32(0.0):
            u += (d ** np.float32(0.8)) * w[f]
        else:
            u += np.float32(-2.25) * ((-d) ** np.float32(0.8)) * w[f]
    return u


# ── Main simulation kernel (numba-compiled) ─────────────────────────────────

@njit(cache=True, fastmath=True, nogil=True)
def _simulate_core(
        T,                        # int:   number of time steps
        portfolio_options,        # (N_PROPS, N_F) float32
        competitor_options,       # (N_COMP, N_F) float32
        competitor_log_sizes,     # (N_COMP,) float32 — ln(cluster_size_k)
        outside_utility,          # float32: base utility of outside option
        outside_log_size,         # float32: ln(outside_size)
        directions,               # (N_F,) float32
        w,                        # (N_F,) float32: feature weights
        utility_scale,            # float32: temperature beta
        refs,                     # (M, N_F) float32, modified in-place
        current_choice,           # (M,) int32: flat index into options (-1 = none), modified in-place
        switching_costs,          # (M,) float32
        agent_to_properties,      # (M, K) int32: visible property indices
        lr_ref,                   # float32: EMA learning rate for references
        gumbel,                   # (T, M, K+N_COMP+1) float32: iid Gumbel noise
        lease_capacity,           # float32: kept for API compatibility; only 1 is supported
        churn_u,                  # (T, M) float32: uniform noise for lease-expiry events
        churn_prob,               # float32: per-step lease-expiry probability
        order,                    # (T, M) int32: per-step random permutation of agents
):
    """MNL choice with exclusive occupancy and leases.

    Option indexing (flat, per agent):
        [0 .. K-1]              — K visible portfolio properties
        [K .. K+N_COMP-1]       — N_COMP competitor clusters
        [K+N_COMP]              — outside option

    Returns
    -------
    portfolio_share_history : (T,) float32  — area-weighted occupancy per step
    mean_rate_history       : (T,) float32  — area-weighted mean rate of let units
    first_leased            : (N_PROPS,) int32 — first step each prop is leased (-1 = never)
    occupied_history        : (T, N_PROPS) bool
    """
    M = refs.shape[0]
    N_PROPS = portfolio_options.shape[0]
    N_COMP = competitor_options.shape[0]
    K = agent_to_properties.shape[1]
    N_OPTIONS = K + N_COMP + 1
    OUTSIDE = K + N_COMP
    N_F = portfolio_options.shape[1]

    beta = np.float32(utility_scale)
    lr = np.float32(lr_ref)
    v_out = np.float32(outside_utility) + np.float32(outside_log_size)

    portfolio_share_history = np.zeros(T, dtype=np.float32)
    mean_rate_history = np.zeros(T, dtype=np.float32)
    first_leased = np.full(N_PROPS, -1, dtype=np.int32)
    occupied_history = np.zeros((T, N_PROPS), dtype=np.bool_)

    total_area = np.float32(0.0)
    for j in range(N_PROPS):
        total_area += portfolio_options[j, 1]  # feature index 1 = area

    # occupant[j] = agent holding property j (-1 = vacant)
    occupant = np.full(N_PROPS, -1, dtype=np.int32)
    # in_lease[i] = agent i holds a portfolio or competitor lease
    in_lease = np.zeros(M, dtype=np.bool_)
    for i in range(M):
        cc = current_choice[i]
        if cc >= 0 and cc < K:
            j = agent_to_properties[i, cc]
            if occupant[j] < 0:
                occupant[j] = i
                in_lease[i] = True
            else:
                current_choice[i] = np.int32(-1)
        elif cc >= K and cc < OUTSIDE:
            in_lease[i] = True

    v_buf = np.empty(N_OPTIONS, dtype=np.float32)
    avail = np.empty(N_OPTIONS, dtype=np.bool_)

    for t in range(T):
        for r in range(M):
            i = order[t, r]

            # Locked-in tenants skip the market unless the lease expires now
            if in_lease[i] and churn_u[t, i] >= churn_prob:
                continue

            cc = current_choice[i]
            holds_lease = in_lease[i]
            sc = switching_costs[i]

            # Portfolio properties: available if vacant or held by this agent
            for k in range(K):
                j = agent_to_properties[i, k]
                if occupant[j] >= 0 and occupant[j] != i:
                    avail[k] = False
                    continue
                avail[k] = True
                v_buf[k] = beta * _prospect_utility(
                    portfolio_options, j, refs, i, directions, w
                )

            for k in range(N_COMP):
                avail[K + k] = True
                v_buf[K + k] = beta * _prospect_utility(
                    competitor_options, k, refs, i, directions, w
                ) + competitor_log_sizes[k]

            avail[OUTSIDE] = True
            v_buf[OUTSIDE] = v_out

            # Switching cost: only a tenant whose lease expires pays to move
            if holds_lease:
                for o in range(N_OPTIONS):
                    if o != cc:
                        v_buf[o] -= sc

            # Gumbel-max over available options (outside is always available)
            best_o = np.int32(OUTSIDE)
            best_v = v_buf[OUTSIDE] + gumbel[t, i, OUTSIDE]
            for o in range(OUTSIDE):
                if not avail[o]:
                    continue
                v = v_buf[o] + gumbel[t, i, o]
                if v > best_v:
                    best_v = v
                    best_o = np.int32(o)

            # Release the old portfolio unit if the agent leaves it
            if cc >= 0 and cc < K and best_o != cc:
                j_old = agent_to_properties[i, cc]
                if occupant[j_old] == i:
                    occupant[j_old] = np.int32(-1)

            current_choice[i] = best_o

            if best_o < K:
                j = agent_to_properties[i, best_o]
                occupant[j] = np.int32(i)
                in_lease[i] = True
                for f in range(N_F):
                    refs[i, f] = (np.float32(1.0) - lr) * refs[i, f] + lr * portfolio_options[j, f]
            elif best_o < OUTSIDE:
                in_lease[i] = True
                comp_idx = best_o - K
                for f in range(N_F):
                    refs[i, f] = (np.float32(1.0) - lr) * refs[i, f] + lr * competitor_options[comp_idx, f]
            else:
                in_lease[i] = False

        # --- Area-weighted metrics ---
        occupied_area = np.float32(0.0)
        rate_weighted = np.float32(0.0)
        for j in range(N_PROPS):
            if occupant[j] >= 0:
                occupied_history[t, j] = True
                if first_leased[j] < 0:
                    first_leased[j] = t
                area_j = portfolio_options[j, 1]
                occupied_area += area_j
                rate_weighted += portfolio_options[j, 0] * area_j
        if total_area > np.float32(0.0):
            portfolio_share_history[t] = occupied_area / total_area
        if occupied_area > np.float32(0.0):
            mean_rate_history[t] = rate_weighted / occupied_area

    return portfolio_share_history, mean_rate_history, first_leased, occupied_history


# ── Gumbel noise generator ──────────────────────────────────────────────────

_rng = np.random.default_rng()


def _gumbel_noise(shape):
    """Standard Gumbel(0, 1) noise via inverse-CDF: -log(Exp(1))."""
    E = _rng.standard_exponential(size=shape, dtype=np.float32)
    np.clip(E, 1e-30, None, out=E)
    np.log(E, out=E)
    np.negative(E, out=E)
    return E


# ── Simulator wrapper ────────────────────────────────────────────────────────

class RealEstateSimulator:
    """Flat MNL real-estate market simulator.

    Parameters
    ----------
    T : int
        Number of simulation time steps (weeks).
    K : int
        Number of portfolio properties visible to each agent.
    lr_ref : float
        EMA learning rate for agent reference-point updates.
    outside_utility : float
        Base deterministic utility of the outside option.
    outside_size : float
        Size of the outside option (enters as ln(outside_size)).
    outside_log_size : float or None
        Explicit ln(outside_size); overrides outside_size if given.
    competitor_log_sizes : array or None
        Explicit ln(cluster_size) per competitor; overrides competitor_sizes.
    competitor_sizes : array or None
        Cluster sizes per competitor (converted to log).
    utility_scale : float
        Temperature beta that scales prospect utilities.
    """

    def __init__(
            self,
            T: int,
            K: int = 8,
            lr_ref: float = 0.05,
            outside_utility: float = 0.0,
            outside_log_size: float = None,
            competitor_log_sizes: np.ndarray = None,
            competitor_sizes: np.ndarray = None,
            outside_size: float = 100.0,
            utility_scale: float = 1.0,
            churn_prob: float = 1.0 / 104.0,
    ):
        self.T = T
        self.K = K
        self.lr_ref = lr_ref
        self.outside_utility = outside_utility
        self.utility_scale = float(utility_scale)
        self.churn_prob = float(churn_prob)

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

    def sample_gumbel(self, M: int, seed: int = None):
        """Sample iid Gumbel(0,1) noise for one simulation run.

        Returns array of shape (T, M, K + N_COMP + 1).
        If seed is given, noise is reproducible (for common random numbers).
        """
        N_COMP = len(self.competitor_log_sizes)
        N_OPTIONS = self.K + N_COMP + 1
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        E = rng.standard_exponential(size=(self.T, M, N_OPTIONS), dtype=np.float32)
        np.clip(E, 1e-30, None, out=E)
        np.log(E, out=E)
        np.negative(E, out=E)
        return E

    def sample_churn(self, M: int, seed: int = None):
        """Sample uniform noise for churn events, shape (T, M).

        Uses an independent stream from sample_gumbel even with the same seed.
        """
        rng = np.random.default_rng([seed, 1]) if seed is not None else np.random.default_rng()
        return rng.random((self.T, M), dtype=np.float32)

    def sample_order(self, M: int, seed: int = None):
        """Per-step random permutation of agents, shape (T, M).

        Independent stream from sample_gumbel / sample_churn even with the same seed.
        """
        rng = np.random.default_rng([seed, 2]) if seed is not None else np.random.default_rng()
        base = np.broadcast_to(np.arange(M, dtype=np.int32), (self.T, M))
        return rng.permuted(base, axis=1).astype(np.int32)

    def run(
            self,
            rates: np.ndarray,
            fixed_features: np.ndarray,
            competitor_features: np.ndarray,
            agents_ref: np.ndarray,
            agents_w: np.ndarray,
            agents_directions: np.ndarray,
            agents_switching_cost: np.ndarray,
            agents_current_choice: np.ndarray,
            agents_to_properties: np.ndarray,
            gumbel=None,
            churn_u=None,
            lease_capacity: float = 1.0,
            order=None,
    ):
        """Run the simulation and return equilibrium metrics.

        Parameters
        ----------
        rates : (N_PROPS,) float32
        fixed_features : (N_PROPS, N_F-1) float32 — area, location_score
        competitor_features : (N_COMP, N_F) float32
        agents_ref : (M, N_F) float32 — reference points (copied internally)
        agents_w : (N_F,) float32 — feature weights
        agents_directions : (N_F,) float32 — feature directions
        agents_switching_cost : (M,) float32
        agents_current_choice : (M,) int32 — current flat option index (-1 = none)
        agents_to_properties : (M, K) int32 — visible property indices
        gumbel : (T, M, N_OPTIONS) float32 or None — pre-sampled noise (CRN)
        lease_capacity : float — must be 1 (one tenant per unit); kept for API compatibility
        order : (T, M) int32 or None — per-step agent order (CRN); sampled if None

        Returns
        -------
        dict with keys: eq_step, occupancy, mean_rate, portfolio_share_history,
            mean_rate_history, first_leased_step, frac_vacant.
        """
        portfolio_options = np.concatenate(
            [rates[:, None], fixed_features], axis=1
        ).astype(np.float32)
        competitor_options = np.asarray(competitor_features, dtype=np.float32)

        M = agents_ref.shape[0]

        if gumbel is None:
            gumbel = self.sample_gumbel(M)
        if lease_capacity != 1.0:
            raise ValueError("only lease_capacity=1 (one tenant per unit) is supported")
        if churn_u is None:
            churn_u = np.random.default_rng().random((self.T, M), dtype=np.float32)
        if order is None:
            order = self.sample_order(M)

        refs = agents_ref.copy().astype(np.float32)
        current_choice = agents_current_choice.copy().astype(np.int32)
        w = np.asarray(agents_w, dtype=np.float32)
        directions = np.asarray(agents_directions, dtype=np.float32)
        switching_costs = agents_switching_cost.astype(np.float32)
        agent_to_properties = np.asarray(agents_to_properties, dtype=np.int32)

        portfolio_share_history, mean_rate_history, first_leased, occupied_history = _simulate_core(
            self.T,
            portfolio_options,
            competitor_options,
            self.competitor_log_sizes,
            np.float32(self.outside_utility),
            np.float32(self.outside_log_size),
            directions,
            w,
            np.float32(self.utility_scale),
            refs,
            current_choice,
            switching_costs,
            agent_to_properties,
            self.lr_ref,
            gumbel.astype(np.float32),
            np.float32(lease_capacity),
            churn_u.astype(np.float32),
            np.float32(self.churn_prob),
            np.asarray(order, dtype=np.int32),
        )

        # Equilibrium metrics: average over last min(20, T) steps
        window = min(20, self.T)
        eq_occupancy = float(portfolio_share_history[-window:].mean())
        eq_mean_rate = float(mean_rate_history[-window:].mean())

        return {
            "eq_step": self.T,
            "occupancy": eq_occupancy,
            "mean_rate": eq_mean_rate,
            "portfolio_share_history": portfolio_share_history,
            "mean_rate_history": mean_rate_history,
            "first_leased_step": first_leased,
            "occupied_history": occupied_history,
            "frac_vacant": float(1.0 - eq_occupancy),
        }
