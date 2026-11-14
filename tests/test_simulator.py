"""
Verification and docking tests for abi.forward.estate.

Run from the project root:
    python -m unittest tests.test_simulator -v

Groups:
    Verification   — the code does what the model description says.
    Docking        — the simulator matches closed-form results in special cases.
    Market realism — properties any rental-market model must have
                     (exclusive occupancy, lease persistence).
"""

import unittest

import numpy as np

from abi.core.functions import prospect_np
from abi.forward.estate import RealEstateSimulator, _simulate_core
from abi.inverse.builders import build_agents, build_competitors, DEFAULT_COMPETITOR_SIZES
from abi.inverse.scenario import MarketScenario

W = np.array([0.4, 0.3, 0.3], np.float32)
D = np.array([-1.0, 1.0, 1.0], np.float32)


def _portfolio(n_props, seed=42, rate=0.5):
    rng = np.random.default_rng(seed)
    fixed = rng.uniform(0.2, 0.8, size=(n_props, 2)).astype(np.float32)
    rates = np.full(n_props, rate, np.float32)
    return rates, fixed


def _run_kernel(rates, fixed, *, n_agents, K, T, seed=0, outside_utility=1.0,
                comp_log_sizes=None, sc_scale=0.3, utility_scale=3.0, lr_ref=0.05, churn=0.0):
    """Call the numba kernel directly so that final agent state is observable."""
    n_props = len(rates)
    comps = build_competitors().astype(np.float32)
    if comp_log_sizes is None:
        comp_log_sizes = np.log(DEFAULT_COMPETITOR_SIZES * 5).astype(np.float32)
    ag = build_agents(MarketScenario(n_agents=n_agents, switching_cost_scale=sc_scale), W, D,
                      n_props=n_props, K=K, seed=seed)
    port = np.concatenate([rates[:, None], fixed], 1).astype(np.float32)
    refs = ag["refs"].copy()
    cur = ag["current_choice"].copy()
    # independent streams: a run with T+1 steps shares its first T steps with a run of T steps
    gumbel = -np.log(np.clip(np.random.default_rng([seed, 0]).standard_exponential(
        (T, n_agents, K + len(comps) + 1)), 1e-30, None)).astype(np.float32)
    churn_u = np.random.default_rng([seed, 1]).random((T, n_agents), dtype=np.float32)
    order = np.random.default_rng([seed, 2]).permuted(np.broadcast_to(np.arange(n_agents, dtype=np.int32), (T, n_agents)), axis=1)
    hist, rate_hist, first, occ = _simulate_core(
        T, port, comps, comp_log_sizes, np.float32(outside_utility), np.float32(0.0), D, W,
        np.float32(utility_scale), refs, cur, ag["switching_costs"], ag["agent_to_properties"],
        np.float32(lr_ref), gumbel, np.float32(1.0), churn_u, np.float32(churn),
        order.astype(np.int32))
    return {"share": hist, "occupied": occ, "current_choice": cur,
            "agent_to_properties": ag["agent_to_properties"], "K": K}


def _tenants_per_property(res, n_props):
    cur, a2p, K = res["current_choice"], res["agent_to_properties"], res["K"]
    in_port = cur < K
    props = a2p[np.arange(len(cur))[in_port], cur[in_port]]
    return np.bincount(props, minlength=n_props)


# ── Verification ─────────────────────────────────────────────────────────────

class TestVerification(unittest.TestCase):

    def test_reproducible_with_seeds(self):
        rates, fixed = _portfolio(30)
        a = _run_kernel(rates, fixed, n_agents=60, K=10, T=30, seed=7)
        b = _run_kernel(rates, fixed, n_agents=60, K=10, T=30, seed=7)
        np.testing.assert_array_equal(a["occupied"], b["occupied"])

    def test_occupancy_in_unit_interval(self):
        rates, fixed = _portfolio(30)
        res = _run_kernel(rates, fixed, n_agents=60, K=10, T=30)
        self.assertTrue(np.all(res["share"] >= 0.0) and np.all(res["share"] <= 1.0 + 1e-6))

    def test_higher_rate_never_raises_own_occupancy(self):
        """Excess supply + finite leases: a unit's occupancy must fall as its own rate rises."""
        rates, fixed = _portfolio(40)
        freqs = []
        for r0 in (0.1, 0.5, 1.0):
            rr = rates.copy()
            rr[0] = r0
            f = np.mean([_run_kernel(rr, fixed, n_agents=30, K=20, T=104, seed=s,
                                     churn=1.0 / 13.0)["occupied"][20:, 0].mean()
                         for s in range(30)])
            freqs.append(f)
        self.assertGreaterEqual(freqs[0], freqs[1])
        self.assertGreaterEqual(freqs[1], freqs[2])
        self.assertGreater(freqs[0], freqs[2] + 0.05, f"occupancy by rate: {freqs}")

    def test_dominant_outside_option_empties_portfolio(self):
        rates, fixed = _portfolio(30)
        res = _run_kernel(rates, fixed, n_agents=60, K=10, T=20, outside_utility=50.0)
        self.assertEqual(res["share"].max(), 0.0)


# ── Docking with closed-form MNL ─────────────────────────────────────────────

class TestDocking(unittest.TestCase):

    def test_single_agent_matches_mnl_softmax(self):
        """One agent, no competitors/outside, no switching cost, frozen reference,
        lease expiring every step: choice frequencies must equal softmax(beta * prospect utility)."""
        n_props, T, beta = 3, 40000, 3.0
        rates = np.array([0.3, 0.5, 0.7], np.float32)
        fixed = np.array([[0.5, 0.5], [0.6, 0.4], [0.4, 0.7]], np.float32)
        port = np.concatenate([rates[:, None], fixed], 1).astype(np.float32)
        ref = np.array([[0.5, 0.5, 0.5]], np.float32)

        rng = np.random.default_rng(0)
        comps = build_competitors().astype(np.float32)
        gumbel = -np.log(rng.standard_exponential((T, 1, n_props + len(comps) + 1))).astype(np.float32)
        _, _, _, occ = _simulate_core(
            T, port, comps, np.full(len(comps), -50.0, np.float32), np.float32(-50.0), np.float32(0.0),
            D, W, np.float32(beta), ref.copy(), np.full(1, -1, np.int32), np.zeros(1, np.float32),
            np.arange(n_props, dtype=np.int32)[None, :], np.float32(0.0), gumbel, np.float32(1.0),
            np.zeros((T, 1), np.float32), np.float32(1.0), np.zeros((T, 1), np.int32))
        empirical = occ.mean(axis=0)

        delta = (port - ref) * D
        v = beta * (prospect_np(delta, alpha=0.8, beta=0.8, lam=2.25) * W).sum(axis=1)
        expected = np.exp(v - v.max()) / np.exp(v - v.max()).sum()
        np.testing.assert_allclose(empirical, expected, atol=0.01)


# ── Market realism ───────────────────────────────────────────────────────────

class TestMarketRealism(unittest.TestCase):

    def test_property_has_at_most_one_tenant(self):
        """lease_capacity=1 means a unit can host one tenant at a time."""
        rates, fixed = _portfolio(150)
        res = _run_kernel(rates, fixed, n_agents=300, K=20, T=104)
        counts = _tenants_per_property(res, 150)
        self.assertLessEqual(int(counts.max()), 1,
                             f"{int((counts >= 2).sum())} properties host 2+ tenants (max {int(counts.max())})")

    def test_excess_demand_fills_portfolio(self):
        """2x more tenants than units, no competitors/outside: everything should be let."""
        rates, fixed = _portfolio(150)
        res = _run_kernel(rates, fixed, n_agents=300, K=20, T=104, outside_utility=-50.0,
                          comp_log_sizes=np.full(3, -50.0, np.float32))
        occupied_share = res["occupied"][-20:].mean()
        self.assertGreater(occupied_share, 0.95, f"only {occupied_share:.1%} of units let")

    def test_tenants_do_not_relocate_every_week(self):
        """With leases, the weekly relocation rate should be a few percent at most."""
        rates, fixed = _portfolio(150)
        a = _run_kernel(rates, fixed, n_agents=300, K=20, T=60, seed=3, churn=1.0 / 104.0)
        b = _run_kernel(rates, fixed, n_agents=300, K=20, T=61, seed=3, churn=1.0 / 104.0)
        # same seed, one extra step -> share of agents whose option changed in that step
        moved = float((a["current_choice"] != b["current_choice"]).mean())
        self.assertLess(moved, 0.05, f"{moved:.0%} of agents changed option in a single week")


    def test_lease_expiry_creates_turnover(self):
        """With finite leases (mean 26 steps) units are periodically vacated and re-let."""
        rates, fixed = _portfolio(150)
        res = _run_kernel(rates, fixed, n_agents=300, K=20, T=104, churn=1.0 / 26.0)
        occ = res["occupied"][30:]
        stay = (occ[:-1] & occ[1:]).sum() / occ[:-1].sum()
        self.assertLess(stay, 0.999, "units are never vacated despite expiring leases")
        self.assertGreater(stay, 0.90, f"stay probability {stay:.3f} is too low for 26-step leases")

    def test_unsupported_lease_capacity_rejected(self):
        rates, fixed = _portfolio(10)
        sim = RealEstateSimulator(T=5, K=5)
        ag = build_agents(MarketScenario(n_agents=10), W, D, n_props=10, K=5, seed=0)
        with self.assertRaises(ValueError):
            sim.run(rates, fixed, build_competitors(), ag["refs"], ag["w"], ag["directions"],
                    ag["switching_costs"], ag["current_choice"], ag["agent_to_properties"],
                    lease_capacity=2.0)


if __name__ == "__main__":
    unittest.main()
