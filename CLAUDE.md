# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
# Real estate portfolio optimization (NES rate optimisation)
python -m abi.demo.estate            # outputs resources/re_optimization_result.json

# Forward simulation with optimized rates
python -m abi.demo.fwd_estate        # loads result.json, runs one full simulation
python -m abi.demo.fwd_estate --rates 0.5   # override all rates

# Lease-time analysis
python -m abi.demo.lease_time        # P(leased by week t) curves, rate sweep, histograms

# Stylized facts validation
python -m abi.demo.stylized_facts    # frictional vacancy + heavy-tail exposure checks

# Jupyter notebook (full walkthrough with visualizations)
jupyter notebook model.ipynb
```

No build step — plain Python package with no `pyproject.toml` or `setup.py`. Run from the repo root so `abi/` is on the Python path. Outputs go to `resources/`.

## Architecture

```
abi/
├── core/
│   ├── entities.py   # Property, Competitor, TenantAgent (real estate domain types)
│   └── functions.py  # prospect_np (prospect theory utility), softmax
├── forward/
│   └── estate.py     # RealEstateSimulator — flat MNL with prospect theory (numba JIT)
├── inverse/
│   ├── scenario.py   # MarketScenario — fixed market scenario parameters
│   ├── generator.py  # RateGenerator — optimizes rental rate vector ξ ∈ [rate_min, rate_max]^N_PROPS
│   ├── grad.py       # evolution_gradient() — NES gradient estimator (dimension-agnostic)
│   ├── loss.py       # RealEstateLoss — phased loss (penalty when infeasible, -rate when feasible)
│   ├── train.py      # optimize_rates() — single NES loop for rate optimisation
│   └── builders.py   # build_competitors(), build_agents() factories
└── demo/
    ├── estate.py         # Rate optimisation demo
    ├── fwd_estate.py     # Forward simulation with optimized rates
    ├── lease_time.py     # Lease probability curves and rate sensitivity
    └── stylized_facts.py # Model validation: frictional vacancy, heavy-tail exposure
```

### Real Estate Data Flow

1. **MarketScenario** defines fixed tenant population parameters (`n_agents`, `ref_center`, `ref_std`, `switching_cost_scale`).
2. **build_agents()** generates heterogeneous tenant agents (reference points, switching costs, property visibility).
3. **RateGenerator** samples a population of rate vectors ξ ∈ ℝ^N_PROPS around current rates (NES with mirrored sampling).
4. **RealEstateSimulator** runs flat MNL market dynamics: each agent chooses among [K visible portfolio properties] + [3 competitor clusters] + [outside option] via Gumbel-max with prospect-theory utility. Returns `occupancy` (area-weighted fraction of binary-occupied properties) and `mean_rate` (area-weighted mean rate over occupied properties) at equilibrium.
5. **RealEstateLoss** applies a phased loss: when `occupancy < threshold` — `penalty·shortage²` (no rate term); when feasible — `−mean_rate`.
6. **evolution_gradient()** estimates NES gradient; `optimize_rates()` updates rates via gradient descent.
7. Repeat for `outer_steps` iterations.

### Choice Model (Flat MNL)

Each agent on each time step chooses among `K + N_COMP + 1` options via Gumbel-max (multinomial logit sampling):

- **Portfolio property j**: `v = β · prospect_utility(features_j, ref_i)`
- **Competitor cluster k**: `v = β · prospect_utility(features_k, ref_i) + ln(cluster_size_k)`
- **Outside option**: `v = outside_utility + ln(outside_size)`

**Switching cost**: subtracted from every option except the agent's current choice ("moving costs `sc`").

**Prospect-theory utility** (Tversky & Kahneman 1992):
- `delta = (option[f] - ref[f]) * direction[f]` per feature
- Gains: `delta^0.8 * w[f]`; Losses: `-2.25 * (-delta)^0.8 * w[f]`

**Reference update** (EMA to chosen option, portfolio/competitor only):
- `ref ← (1 - lr_ref) * ref + lr_ref * chosen_features`

### Key Design Decisions

- **Flat MNL** (not nested logit): single-level Gumbel-max over all options. Transparent sensitivity to rate via logit elasticity.
- **Binary occupancy**: property j is occupied if `chose_count[j] ≥ lease_capacity` (default 1), vacant otherwise. `occupancy` is the area-weighted fraction. `mean_rate` is area-weighted over occupied properties.
- **NES instead of backprop**: the simulator uses Gumbel-max for stochastic discrete choice; gradients through it are unreliable, so ES gradient estimation is used.
- **Mirrored sampling**: antithetic noise pairs `(+ε, −ε)` reduce NES variance.
- **CRN (Common Random Numbers)**: all rate candidates in one NES step share the same Gumbel noise, reducing evaluation variance.
- **Prospect theory + reference points**: agents compare options to their adaptive reference vector — deviations go through an S-shaped, loss-averse function. Temperature `utility_scale` (β) sharpens price sensitivity.
- **Outside option**: configurable `outside_utility` and `outside_log_size`. Non-trivial `outside_utility` (e.g. 1.0) makes it a credible alternative.
- **Competitor clusters**: 3 clusters (budget/mid/premium) with `DEFAULT_COMPETITOR_SIZES = [3.0, 2.0, 1.0]`. Sizes enter as `ln(size)` additive correction. Demo scales these ×5 for elastic occupancy.
- **Phased loss**: when infeasible, `penalty·shortage²` only (no rate term → unambiguous gradient toward feasibility). When feasible, `−mean_rate` (pure rate maximisation).

## Feature Space (Real Estate)

All features are normalized to [0, 1]:

| Feature | Direction | Managed? |
|---|---|---|
| `rate` | −1 (lower is better for tenant) | Yes — this is ξ |
| `area` | +1 (larger is better) | No — fixed per property |
| `location_score` | +1 (higher is better) | No — fixed per property |

## Dependencies

Core: `numpy`, `matplotlib`, `tqdm`, `numba`. No package manager file — install manually or use the `.venv/` virtual environment already in the repo.

`numba` JIT-compiles the real-estate simulator hot loop (`_simulate_core` in `abi/forward/estate.py`). On first run it compiles and caches to `__pycache__/*.nbi` (~2–5 s one-time overhead); subsequent runs load from cache instantly. Install: `pip install numba`.
