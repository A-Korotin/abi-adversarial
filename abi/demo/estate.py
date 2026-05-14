import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import numpy as np
import json

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESOURCES_DIR = os.path.join(_PROJECT_ROOT, "resources")
os.makedirs(RESOURCES_DIR, exist_ok=True)

from abi.forward.estate import RealEstateSimulator
from abi.inverse.loss import RealEstateLoss
from abi.inverse.builders import build_competitors, build_agents, DEFAULT_COMPETITORS
from abi.inverse.generator import RateGenerator
from abi.inverse.sampler import MarketScenario, ScenarioSampler
from abi.inverse.train import train_adversarial

N_PROPS    = 150
N_AGENTS   = 1_000
T          = 104
N_FEATURES = 3

TARGET_RATE = 0.65
OCCUPANCY_THRESHOLD = 0.75

FEATURE_DIRECTIONS = np.array([-1.0, 1.0, 1.0], dtype=np.float32)

WEIGHTS = np.array([0.4, 0.3, 0.3], dtype=np.float32)

rng = np.random.default_rng(42)

fixed_features = rng.uniform(0.2, 0.8, size=(N_PROPS, 2)).astype(np.float32)
initial_rates = np.full(N_PROPS, 0.5, dtype=np.float32)
competitor_features = build_competitors()

generator = RateGenerator(
    initial_rates=initial_rates,
    sigma=np.full(N_PROPS, 0.05, dtype=np.float32),
    rate_min=0.1,
    rate_max=1.0,
)

base_phi = MarketScenario(
    n_agents=N_AGENTS,
    ref_shift=np.zeros(N_FEATURES, dtype=np.float32),
    ref_std=0.1,
    switching_cost_scale=0.3,
)

sampler_sigma = np.array([0.05, 0.05, 0.05, 0.02, 0.05], dtype=np.float32)

discriminator = ScenarioSampler(
    base_phi=base_phi,
    sigma=sampler_sigma,
    n_agents=N_AGENTS,
    n_features=N_FEATURES,
)

simulator = RealEstateSimulator(
    T=T,
    lr_ref=0.05,
    ref_market_weight=0.3,
    outside_utility=0.0,
)

loss_fn = RealEstateLoss(
    target_rate=TARGET_RATE,
    occupancy_threshold=OCCUPANCY_THRESHOLD,
    penalty=10.0,
)

print(f"Портфель: {N_PROPS} объектов, агенты: {N_AGENTS}, целевая ставка: {TARGET_RATE:.2f}, ограничение: заполняемость >= {OCCUPANCY_THRESHOLD:.1%}")

_t0 = time.perf_counter()
history = train_adversarial(
    simulator=simulator,
    generator=generator,
    discriminator=discriminator,
    loss_fn=loss_fn,
    fixed_features=fixed_features,
    competitor_features=competitor_features,
    weights=WEIGHTS,
    feature_directions=FEATURE_DIRECTIONS,
    build_agents_fn=lambda phi, w, d: build_agents(phi, w, d, n_props=N_PROPS),
    outer_steps=500,
    pop_size_xi=20,
    pop_size_phi=6,
    lr_g=0.05,
    lr_d=0.01,
)

_elapsed = time.perf_counter() - _t0
print(f"Время обучения: {_elapsed:.1f} сек")

final_rates = history["rates"][-1]
final_occ   = history["occupancy"][-1]
final_rate  = history["mean_rate"][-1]
feasible    = history["feasible"][-1]

delta = final_rate - TARGET_RATE
print(f"\nРезультаты оптимизации:")
print(f"  Заполняемость:        {final_occ:.1%}  ({'OK' if feasible else 'FAIL'} ограничение {OCCUPANCY_THRESHOLD:.0%})")
print(f"  Целевая ставка:       {TARGET_RATE:.4f}")
print(f"  Средняя ставка:       {final_rate:.4f}  (отклонение {delta:+.4f})")
print(f"  Финальный loss:       {history['gen_loss'][-1]:.6f}")
print(f"  Мин. ставка портфеля: {final_rates.min():.4f}")
print(f"  Макс. ставка:         {final_rates.max():.4f}")
print(f"  Медиана ставок:       {np.median(final_rates):.4f}")

output = {
    "target_rate": TARGET_RATE,
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