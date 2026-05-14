# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
# Real estate portfolio optimization (main demo)
python -m abi.demo.estate        # outputs re_optimization_result.json

# Original abstract market demos
python -m abi.demo.fwd           # forward simulation only
python -m abi.demo.inverse       # adversarial training (abstract market)

# Jupyter notebook (full walkthrough with visualizations)
jupyter notebook model.ipynb
```

No build step — plain Python package with no `pyproject.toml` or `setup.py`. Run from the repo root so `abi/` is on the Python path.

## Architecture

The codebase contains two domains that share the same adversarial NES infrastructure:

1. **Abstract market** (`demo/fwd.py`, `demo/inverse.py`) — original model, providers with price/speed/quality.
2. **Real estate** (`demo/estate.py`) — portfolio of N properties, optimize rental rates to maximize mean rate subject to occupancy ≥ 80%.

```
abi/
├── core/
│   ├── entities.py   # Provider, Agent (abstract); Property, Competitor, TenantAgent (real estate)
│   └── functions.py  # prospect_np (prospect theory utility), softmax
├── forward/
│   ├── simulator.py  # VectorizedSimulator — abstract market
│   └── estate.py     # RealEstateSimulator — real estate market
├── inverse/
│   ├── sampler.py    # ScenarioSampler + MarketScenario (shared, adapted for N_FEATURES=3)
│   ├── generator.py  # RateGenerator — optimizes rental rate vector ξ ∈ [rate_min, rate_max]^N_PROPS
│   ├── grad.py       # evolution_gradient() — NES gradient estimator (shared, dimension-agnostic)
│   ├── loss.py       # MarketLoss (abstract); RealEstateLoss (maximize rate, penalize low occupancy)
│   ├── train.py      # train_adversarial() — real estate training loop
│   ├── builders.py   # build_portfolio(), build_competitors(), build_agents() factories
│   └── bootstrap.py  # Original factories: build_providers(), build_agents() (abstract market)
└── demo/
    ├── fwd.py        # Abstract forward demo
    ├── inverse.py    # Abstract adversarial demo
    └── estate.py     # Real estate adversarial demo
```

### Real Estate Data Flow

1. **ScenarioSampler** (discriminator) samples worst-case φ: tenant `ref_shift`, `ref_std`, `switching_cost_scale`.
2. **RateGenerator** (generator) samples a population of rate vectors ξ ∈ ℝ^N_PROPS around current rates.
3. **RealEstateSimulator** runs market dynamics: tenants choose among [N_PROPS portfolio] + [3 competitors] + [outside option] using prospect theory utility + Gumbel noise, returning `occupancy` and `mean_rate` at equilibrium.
4. **RealEstateLoss** computes `−mean_rate` when occupancy ≥ threshold, else `penalty·(threshold − occupancy)²`.
5. **evolution_gradient()** estimates NES gradients; discriminator maximizes loss (ascent), generator minimizes (descent).
6. Repeat for `outer_steps` iterations.

### Key Design Decisions

- **NES instead of backprop**: the simulator uses the Gumbel-max trick for stochastic discrete choice; gradients through it are unreliable, so ES gradient estimation is used throughout.
- **Mirrored sampling**: antithetic noise pairs `(+ε, −ε)` reduce variance.
- **Prospect theory utility**: agents compare options to their reference vector — deviations go through an S-shaped, loss-averse function before being weighted.
- **Reference update mix**: `RealEstateSimulator` blends personal choice history with a market benchmark (`ref_market_weight`). Agents choosing the outside option do not update their reference.
- **Outside option**: a zero-feature option with fixed utility (`outside_utility=0.0`) lets tenants opt out of renting entirely.
- **Soft occupancy constraint**: `RealEstateLoss` uses a quadratic penalty on occupancy shortfall rather than a hard constraint, with `penalty` controlling tightness.

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
