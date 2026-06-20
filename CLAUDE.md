# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
# Real estate portfolio optimization (adversarial training)
python -m abi.demo.estate            # outputs resources/re_optimization_result.json

# Forward simulation with optimized rates
python -m abi.demo.fwd_estate        # loads result.json, runs one full simulation
python -m abi.demo.fwd_estate --rates 0.5   # override all rates

# Lease-time analysis
python -m abi.demo.lease_time        # P(leased by week t) curves, rate sweep, histograms

# Stylized facts validation
python -m abi.demo.stylized_facts    # frictional vacancy + heavy-tail exposure checks

# Original abstract market demos
python -m abi.demo.fwd               # forward simulation only
python -m abi.demo.inverse           # adversarial training (abstract market)

# Jupyter notebook (full walkthrough with visualizations)
jupyter notebook model.ipynb
```

No build step — plain Python package with no `pyproject.toml` or `setup.py`. Run from the repo root so `abi/` is on the Python path. Outputs go to `resources/`.

## Architecture

The codebase contains two domains that share the same adversarial NES infrastructure:

1. **Abstract market** (`demo/fwd.py`, `demo/inverse.py`) — original model, providers with price/speed/quality.
2. **Real estate** (`demo/estate.py`) — portfolio of N properties, optimize rental rates to maximize area-weighted mean rate subject to occupancy ≥ 60%.

```
abi/
├── core/
│   ├── entities.py   # Provider, Agent (abstract); Property, Competitor, TenantAgent (real estate)
│   └── functions.py  # prospect_np (prospect theory utility), softmax
├── forward/
│   ├── simulator.py  # VectorizedSimulator — abstract market
│   └── estate.py     # RealEstateSimulator — nested logit with binary occupancy
├── inverse/
│   ├── sampler.py    # ScenarioSampler + MarketScenario (shared, adapted for N_FEATURES=3)
│   ├── generator.py  # RateGenerator — optimizes rental rate vector ξ ∈ [rate_min, rate_max]^N_PROPS
│   ├── grad.py       # evolution_gradient() — NES gradient estimator (shared, dimension-agnostic)
│   ├── loss.py       # MarketLoss (abstract); RealEstateLoss (maximize rate, penalize low occupancy)
│   ├── train.py      # train_adversarial() — real estate training loop
│   ├── builders.py   # build_portfolio(), build_competitors(), build_agents() factories
│   └── bootstrap.py  # Original factories: build_providers(), build_agents() (abstract market)
└── demo/
    ├── fwd.py            # Abstract forward demo
    ├── inverse.py        # Abstract adversarial demo
    ├── estate.py         # Real estate adversarial training
    ├── fwd_estate.py     # Forward simulation with optimized rates
    ├── lease_time.py     # Lease probability curves and rate sensitivity
    └── stylized_facts.py # Model validation: frictional vacancy, heavy-tail exposure
```

### Real Estate Data Flow

1. **ScenarioSampler** (discriminator) samples worst-case φ: tenant `ref_shift`, `ref_std`, `switching_cost_scale`.
2. **RateGenerator** (generator) samples a population of rate vectors ξ ∈ ℝ^N_PROPS around current rates.
3. **RealEstateSimulator** runs nested logit market dynamics: tenants choose among [N_PROPS portfolio] + [3 competitors] + [outside option] via prospect theory utility + Gumbel noise. Returns `occupancy` (area-weighted fraction of binary-occupied properties) and `mean_rate` (area-weighted mean rate over occupied properties) at equilibrium. Also tracks `first_leased_step` per property and nest choice shares.
4. **RealEstateLoss** uses a phased augmented penalty: when `occupancy < threshold` — `penalty_lin·shortage + penalty_quad·shortage²` (no rate term, avoiding gradient conflict); when `occupancy ≥ threshold` — `−mean_rate + buffer_penalty·shortage²` (rate maximisation with a small boundary buffer).
5. **evolution_gradient()** estimates NES gradients; discriminator maximizes loss (ascent), generator minimizes (descent).
6. Repeat for `outer_steps` iterations.

### Key Design Decisions

- **Nested logit**: three nests — portfolio (K visible properties per agent, with McFadden size correction `lp·log(N_PROPS/K)`), competitors (N_COMP clusters with log-sizes), outside option. Nest dissimilarity parameters `λ_p`, `λ_c` control within-nest substitution. Switching cost applies at the nest level only.
- **Binary occupancy**: property j is occupied if `chose_count[j] ≥ lease_capacity` (default 1), vacant otherwise. `occupancy` is the area-weighted fraction of occupied properties. `mean_rate` is area-weighted over occupied properties.
- **NES instead of backprop**: the simulator uses the Gumbel-max trick for stochastic discrete choice; gradients through it are unreliable, so ES gradient estimation is used throughout.
- **Mirrored sampling**: antithetic noise pairs `(+ε, −ε)` reduce variance.
- **Prospect theory utility**: agents compare options to their reference vector — deviations go through an S-shaped, loss-averse function before being weighted. Temperature parameter `utility_scale` (β) sharpens price sensitivity.
- **Reference update mix**: `RealEstateSimulator` blends personal choice history with a market benchmark (`ref_market_weight`). Agents choosing the outside option do not update their reference.
- **Outside option**: a single option with configurable `outside_utility` and `outside_log_size`. Non-trivial `outside_utility` (e.g. 1.0) makes it a credible alternative.
- **Competitor clusters**: 3 clusters (budget/mid/premium) with `DEFAULT_COMPETITOR_SIZES = [3.0, 2.0, 1.0]`. Sizes enter the nested logit as `ln(size)` log-size correction. Demo scales these ×5 for elastic occupancy.
- **Phased loss with augmented penalty**: when the model starts infeasible, `RealEstateLoss` uses only `penalty_lin·shortage + penalty_quad·shortage²` — no rate term — so the gradient points unambiguously toward reducing shortage. The linear term keeps the gradient nonzero even at large shortage. Once feasible, the loss switches to `−mean_rate` to maximise rental income.

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
