"""
Real-estate rate optimisation demo.

Optimises per-property rental rates to maximise area-weighted mean rate
subject to occupancy >= threshold, using NES against a fixed market scenario.

Run:
    python -m abi.demo.estate

Output:
    resources/re_optimization_result.json
"""

import os
import sys
import time
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESOURCES_DIR = os.path.join(_PROJECT_ROOT, "resources")
os.makedirs(RESOURCES_DIR, exist_ok=True)

from abi.forward.estate import RealEstateSimulator
from abi.inverse.loss import RealEstateLoss
from abi.inverse.builders import (
    build_competitors, build_agents, build_competitor_log_sizes,
    DEFAULT_COMPETITOR_SIZES, DEFAULT_OUTSIDE_SIZE,
)
from abi.inverse.generator import RateGenerator
from abi.inverse.scenario import MarketScenario
from abi.inverse.train import optimize_rates

# ── Domain constants ────────────────────────────────────────────────────────
N_PROPS = 150
N_AGENTS = 300
T = 104
N_FEATURES = 3
K_VISIBLE = 20

OCCUPANCY_THRESHOLD = 0.65

FEATURE_DIRECTIONS = np.array([-1.0, 1.0, 1.0], dtype=np.float32)
WEIGHTS = np.array([0.4, 0.3, 0.3], dtype=np.float32)

COMPETITOR_SIZES = DEFAULT_COMPETITOR_SIZES * 5   # [15, 10, 5]
OUTSIDE_SIZE = DEFAULT_OUTSIDE_SIZE                # 1.0
OUTSIDE_UTILITY = 1.0
UTILITY_SCALE = 3.0

# ── Generate synthetic portfolio ────────────────────────────────────────────
rng = np.random.default_rng(42)
fixed_features = rng.uniform(0.2, 0.8, size=(N_PROPS, 2)).astype(np.float32)
initial_rates = np.full(N_PROPS, 0.5, dtype=np.float32)
competitor_features = build_competitors()

# ── Build components ────────────────────────────────────────────────────────
generator = RateGenerator(
    initial_rates=initial_rates,
    sigma=np.full(N_PROPS, 0.1, dtype=np.float32),
    rate_min=0.1,
    rate_max=1.0,
)

scenario = MarketScenario(
    n_agents=N_AGENTS,
    ref_center=0.5,
    ref_std=0.1,
    switching_cost_scale=0.3,
)

agents = build_agents(scenario, WEIGHTS, FEATURE_DIRECTIONS, n_props=N_PROPS, K=K_VISIBLE)

simulator = RealEstateSimulator(
    T=T,
    K=K_VISIBLE,
    lr_ref=0.05,
    outside_utility=OUTSIDE_UTILITY,
    outside_size=OUTSIDE_SIZE,
    competitor_sizes=COMPETITOR_SIZES,
    utility_scale=UTILITY_SCALE,
)

loss_fn = RealEstateLoss(
    occupancy_threshold=OCCUPANCY_THRESHOLD,
    penalty=10.0,
)

# ── Run optimisation ────────────────────────────────────────────────────────
print(f"Портфель: {N_PROPS} объектов, агенты: {N_AGENTS}, K={K_VISIBLE}, "
      f"ограничение: заполняемость >= {OCCUPANCY_THRESHOLD:.1%}")

_t0 = time.perf_counter()
history = optimize_rates(
    simulator=simulator,
    generator=generator,
    loss_fn=loss_fn,
    fixed_features=fixed_features,
    competitor_features=competitor_features,
    agents=agents,
    outer_steps=100,
    pop_size=50,
    lr=0.05,
)
_elapsed = time.perf_counter() - _t0
print(f"Время обучения: {_elapsed:.1f} сек")

# ── Report results ──────────────────────────────────────────────────────────
final_rates = history["rates"][-1]
final_occ = history["occupancy"][-1]
final_rate = history["mean_rate"][-1]
feasible = history["feasible"][-1]

print(f"\nРезультаты оптимизации:")
print(f"  Заполняемость:        {final_occ:.1%}  "
      f"({'OK' if feasible else 'FAIL'} ограничение {OCCUPANCY_THRESHOLD:.0%})")
print(f"  Средняя ставка:       {final_rate:.4f}")
print(f"  Финальный loss:       {history['gen_loss'][-1]:.6f}")
print(f"  Мин. ставка портфеля: {final_rates.min():.4f}")
print(f"  Макс. ставка:         {final_rates.max():.4f}")
print(f"  Медиана ставок:       {np.median(final_rates):.4f}")

output = {
    "final_rates": final_rates.tolist(),
    "final_occupancy": float(final_occ),
    "final_mean_rate": float(final_rate),
    "feasible": bool(feasible),
    "occupancy_history": history["occupancy"],
    "mean_rate_history": history["mean_rate"],
    "gen_loss_history": history["gen_loss"],
}

_json_path = os.path.join(RESOURCES_DIR, "re_optimization_result.json")
with open(_json_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nРезультаты сохранены в {_json_path}")
