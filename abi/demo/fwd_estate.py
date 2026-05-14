"""
Прямой прогон симулятора рынка недвижимости.

Загружает оптимизированные ставки из re_optimization_result.json
и запускает один полный прогон симулятора (T=104 шага), выводя
подробную динамику становления равновесия.

Использование:
    python -m abi.demo.fwd_estate
    python -m abi.demo.fwd_estate --rates 0.5  # переопределить все ставки
    python -m abi.demo.fwd_estate --json path/to/result.json
"""

import sys
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESOURCES_DIR = os.path.join(_PROJECT_ROOT, "resources")
os.makedirs(RESOURCES_DIR, exist_ok=True)

from abi.forward.estate import RealEstateSimulator
from abi.inverse.builders import (
    build_competitors, build_agents,
    DEFAULT_COMPETITOR_SIZES, DEFAULT_OUTSIDE_SIZE,
)
from abi.inverse.sampler import MarketScenario

N_PROPS    = 150
N_AGENTS   = 5_000
N_FEATURES = 3
T          = 104
K_VISIBLE  = 8
OCCUPANCY_THRESHOLD = 0.65

FEATURE_DIRECTIONS = np.array([-1.0, 1.0, 1.0], dtype=np.float32)
WEIGHTS            = np.array([0.4, 0.3, 0.3], dtype=np.float32)

# Синтетический портфель: тот же seed, что и в demo/estate.py
_rng           = np.random.default_rng(42)
FIXED_FEATURES = _rng.uniform(0.2, 0.8, size=(N_PROPS, 2)).astype(np.float32)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--json", default=os.path.join(RESOURCES_DIR, "re_optimization_result.json"),
    help="Путь к JSON с результатами оптимизации",
)
parser.add_argument(
    "--rates", type=float, default=None,
    help="Переопределить все ставки одним числом (0–1)",
)
parser.add_argument(
    "--seed", type=int, default=0,
    help="Seed для генерации агентов",
)
args = parser.parse_args()

if args.rates is not None:
    rates = np.full(N_PROPS, args.rates, dtype=np.float32)
    print(f"Ставки: фиксированные = {args.rates:.3f}")
else:
    with open(args.json) as f:
        saved = json.load(f)
    rates = np.array(saved["final_rates"], dtype=np.float32)
    print(f"Ставки: загружены из {args.json}")
    print(f"  мин={rates.min():.3f}  медиана={np.median(rates):.3f}  макс={rates.max():.3f}")


phi = MarketScenario(
    n_agents=N_AGENTS,
    ref_shift=np.zeros(N_FEATURES, dtype=np.float32),
    ref_std=0.1,
    switching_cost_scale=0.3,
)
np.random.seed(args.seed)
agents = build_agents(phi, WEIGHTS, FEATURE_DIRECTIONS, n_props=N_PROPS, K=K_VISIBLE)

competitor_features = build_competitors()

simulator = RealEstateSimulator(
    T=T,
    K=K_VISIBLE,
    lr_ref=0.05,
    ref_market_weight=0.3,
    outside_utility=0.0,
    outside_size=DEFAULT_OUTSIDE_SIZE,
    lambda_portfolio=0.6,
    lambda_comp=0.6,
    competitor_sizes=DEFAULT_COMPETITOR_SIZES,
)

result = simulator.run(
    rates=rates,
    fixed_features=FIXED_FEATURES,
    competitor_features=competitor_features,
    agents_ref=agents["refs"],
    agents_w=agents["w"],
    agents_directions=agents["directions"],
    agents_switching_cost=agents["switching_costs"],
    agents_current_nest=agents["current_nest"],
    agents_current_inner=agents["current_inner"],
    agents_to_properties=agents["agent_to_properties"],
)

occ_hist  = result["portfolio_share_history"]
rate_hist = result["mean_rate_history"]
eq_step   = result["eq_step"]

print()
print("=" * 60)
print("Равновесие рынка")
print("=" * 60)
print(f"  Шаг равновесия : {eq_step if eq_step else 'не обнаружено (T=' + str(T) + ')'}")
print(f"  Заполняемость  : {result['occupancy']:.1%}")
print(f"  Средняя ставка : {result['mean_rate']:.4f}")
print()
print(f"Стабильность (std заполн. за последние 20 шагов): {occ_hist[-20:].std():.6f}")
print(f"Стабильность (std ставки  за последние 20 шагов): {rate_hist[-20:].std():.6f}")

# Фигура 1: динамика рыночного равновесия (T шагов симулятора)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle("Динамика становления рыночного равновесия", fontsize=14)

ax1.plot(occ_hist, color="steelblue", label="Заполняемость портфеля")
if eq_step is not None:
    ax1.axvline(eq_step, color="red", linestyle="--", label=f"Равновесие (шаг {eq_step})")
    ax1.axhline(result["occupancy"], color="steelblue", linestyle="dashed",
                linewidth=1.5, label=f"Равновесная заполняемость {result['occupancy']:.1%}")
ax1.axhline(OCCUPANCY_THRESHOLD, color="gray", linestyle=":", linewidth=1, label=f"Порог {OCCUPANCY_THRESHOLD:.0%}")
ax1.set_xlabel("Период")
ax1.set_ylabel("Доля арендаторов в портфеле")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(rate_hist, color="darkorange", label="Средняя ставка")
if eq_step is not None:
    ax2.axvline(eq_step, color="red", linestyle="--", label=f"Равновесие (шаг {eq_step})")
    ax2.axhline(result["mean_rate"], color="darkorange", linestyle="dashed",
                linewidth=1.5, label=f"Равновесная ставка {result['mean_rate']:.4f}")
ax2.set_xlabel("Период")
ax2.set_ylabel("Средняя нормированная ставка")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESOURCES_DIR, "fwd_equilibrium_dynamics.png"), dpi=120, bbox_inches="tight")
plt.show()

# Фигуры 2 и 3: история оптимизации и распределение ставок (только если загружен JSON)
if args.rates is None:
    # Фигура 2: прогресс adversarial-обучения (30 внешних шагов)
    gen_loss  = saved["gen_loss_history"]
    occ_opt   = saved["occupancy_history"]
    rate_opt  = saved["mean_rate_history"]
    steps     = list(range(len(gen_loss)))

    target_rate = saved.get("target_rate")

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 7))
    fig2.suptitle("История adversarial-оптимизации", fontsize=14)

    ax3.plot(steps, occ_opt, color="steelblue", label="Заполняемость (лучший кандидат)")
    ax3.axhline(OCCUPANCY_THRESHOLD, color="gray", linestyle=":", linewidth=1, label=f"Порог {OCCUPANCY_THRESHOLD:.0%}")
    ax3.set_xlabel("Внешний шаг")
    ax3.set_ylabel("Заполняемость")
    ax3.set_xticks(range(0, len(steps), max(1, len(steps) // 10)))
    ax3.legend()
    ax3.grid()

    ax4.plot(steps, rate_opt, color="darkorange", label="Средняя ставка")
    if target_rate is not None:
        ax4.axhline(target_rate, color="red", linestyle="--", linewidth=1.5,
                    label=f"Целевая ставка {target_rate:.2f}")
    ax4.set_xlabel("Внешний шаг")
    ax4.set_ylabel("Средняя ставка")
    ax4.set_xticks(range(0, len(steps), max(1, len(steps) // 10)))
    ax4.legend()
    ax4.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(RESOURCES_DIR, "fwd_training_history.png"), dpi=120, bbox_inches="tight")
    plt.show()

    # Фигура 3: распределение оптимизированных ставок по портфелю
    final_rates = np.array(saved["final_rates"])

    fig3, ax5 = plt.subplots(figsize=(15, 7))
    ax5.hist(final_rates, bins=30, color="steelblue", edgecolor="white", alpha=0.85,
             label="Ставки объектов портфеля")
    ax5.axvline(final_rates.mean(), color="red", linestyle="--",
                linewidth=1.5, label=f"Среднее {final_rates.mean():.3f}")
    ax5.axvline(float(np.median(final_rates)), color="darkorange", linestyle="dashed",
                linewidth=1.5, label=f"Медиана {np.median(final_rates):.3f}")
    if target_rate is not None:
        ax5.axvline(target_rate, color="green", linestyle="--",
                    linewidth=1.5, label=f"Целевая ставка {target_rate:.2f}")
    ax5.set_xlabel("Арендная ставка (нормировано)")
    ax5.set_ylabel("Количество объектов")
    ax5.set_title("Распределение оптимизированных ставок портфеля")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESOURCES_DIR, "fwd_rate_distribution.png"), dpi=120, bbox_inches="tight")
    plt.show()

print(f"\nГрафики сохранены в {RESOURCES_DIR}:")
print(f"  fwd_equilibrium_dynamics.png", end="")
if args.rates is None:
    print(", fwd_training_history.png, fwd_rate_distribution.png", end="")
print()
