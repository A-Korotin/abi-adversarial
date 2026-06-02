"""
Lease-time demo: P(leased by week t) for each portfolio property.

Run:
    python -m abi.demo.lease_time

Outputs (to resources/):
    lease_time_curves.png       — CDF curves for 3 contrasting properties
    lease_time_rate_sweep.png   — rate trade-off: higher rate → slower lease
    lease_time_histogram.png    — portfolio-wide first-lease distribution
    lease_time_summary.json     — per-property probabilities and median lease step
"""

import os
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Presentation style ────────────────────────────────────────────────────────
COLOR_BEST   = '#2E7D5B'
COLOR_MEDIAN = '#1E40AF'
COLOR_WORST  = '#B45309'
COLOR_GREY   = '#6B7280'
COLOR_TEXT   = '#1A1A1A'
COLOR_FAINT  = '#D1D5DB'
COLOR_BAND   = '#E5E7EB'

plt.rcParams.update({
    'font.family':       ['DejaVu Sans'],
    'text.color':        COLOR_TEXT,
    'axes.labelcolor':   COLOR_TEXT,
    'axes.edgecolor':    COLOR_FAINT,
    'xtick.color':       COLOR_GREY,
    'ytick.color':       COLOR_GREY,
    'axes.facecolor':    'white',
    'figure.facecolor':  'white',
    'axes.grid':         False,
})


def apply_clean_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_FAINT)
    ax.spines['bottom'].set_color(COLOR_FAINT)
    ax.tick_params(length=0)


def format_axes_percent_weeks(ax, max_week):
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max_week)
    ax.set_xlabel('Неделя', color=COLOR_GREY, fontsize=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))




def business_horizons(ax, weeks=(4, 12, 26)):
    for w in weeks:
        ax.axvline(w - 1, color=COLOR_FAINT, lw=1, zorder=0)
        ax.text(w - 1, 0.01, f'{w} нед',
                ha='center', fontsize=8, color=COLOR_GREY, va='bottom')

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from abi.forward.estate import RealEstateSimulator
from abi.inverse.builders import (
    build_competitors,
    build_agents,
    build_competitor_log_sizes,
    DEFAULT_COMPETITOR_SIZES,
    DEFAULT_OUTSIDE_SIZE,
)
from abi.inverse.sampler import MarketScenario

# ── Constants (mirror abi/demo/estate.py) ────────────────────────────────────
N_PROPS = 150
N_AGENTS = 100
T = 52
N_FEATURES = 3
K_VISIBLE = 20

FEATURE_DIRECTIONS = np.array([-1.0, 1.0, 1.0], dtype=np.float32)
WEIGHTS = np.array([0.4, 0.3, 0.3], dtype=np.float32)

COMPETITOR_SIZES = DEFAULT_COMPETITOR_SIZES
OUTSIDE_SIZE = DEFAULT_OUTSIDE_SIZE
LAMBDA_PORTFOLIO = 0.6
LAMBDA_COMP = 0.6

N_RUNS = 300       # baseline Monte-Carlo runs
N_SWEEP = 200      # runs per rate level in the sweep
RATE_GRID = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESOURCES_DIR = os.path.join(_PROJECT_ROOT, "resources")
os.makedirs(RESOURCES_DIR, exist_ok=True)

# ── Generate fixed features (same seed as estate.py) ─────────────────────────
rng_fixed = np.random.default_rng(42)
fixed_features = rng_fixed.uniform(0.2, 0.8, size=(N_PROPS, 2)).astype(np.float32)
# columns: [area, location_score]

competitor_features = build_competitors()
competitor_log_sizes = build_competitor_log_sizes(COMPETITOR_SIZES)

# ── Load optimized rates (or fall back to 0.5) ────────────────────────────────
_json_path = os.path.join(RESOURCES_DIR, "re_optimization_result.json")
if os.path.exists(_json_path):
    with open(_json_path) as f:
        _opt = json.load(f)
    rates_opt = np.array(_opt["final_rates"], dtype=np.float32)
    print(f"Загружены оптимизированные ставки из {_json_path}")
else:
    rates_opt = np.full(N_PROPS, 0.5, dtype=np.float32)
    print("re_optimization_result.json не найден — используются ставки 0.5")

# ── Build simulator and fixed agents ─────────────────────────────────────────
sim = RealEstateSimulator(
    T=T,
    K=K_VISIBLE,
    lr_ref=0.05,
    ref_market_weight=0.3,
    outside_utility=0.0,
    outside_size=OUTSIDE_SIZE,
    lambda_portfolio=LAMBDA_PORTFOLIO,
    lambda_comp=LAMBDA_COMP,
    competitor_sizes=COMPETITOR_SIZES,
)

base_phi = MarketScenario(
    n_agents=N_AGENTS,
    ref_shift=np.zeros(N_FEATURES, dtype=np.float32),
    ref_std=0.1,
    switching_cost_scale=0.3,
)

# Fix agents across all runs (only Gumbel noise varies)
agents = build_agents(base_phi, WEIGHTS, FEATURE_DIRECTIONS, n_props=N_PROPS, K=K_VISIBLE)

# ── Helper: run simulator and collect first_leased_step ──────────────────────
LEASE_CAPACITY = 1  # property leased when ≥1 tenant chooses it; M-independent

def _run_once(rates, disable_early_exit=True):
    gumbel = sim.sample_gumbel(N_AGENTS)
    result = sim.run(
        rates=rates,
        fixed_features=fixed_features,
        competitor_features=competitor_features,
        agents_ref=agents["refs"],
        agents_w=agents["w"],
        agents_directions=agents["directions"],
        agents_switching_cost=agents["switching_costs"],
        agents_current_nest=agents["current_nest"].copy(),
        agents_current_inner=agents["current_inner"].copy(),
        agents_to_properties=agents["agent_to_properties"],
        gumbel=gumbel,
        disable_early_exit=disable_early_exit,
        lease_capacity=LEASE_CAPACITY,
    )
    return result["first_leased_step"]


def _compute_cdf(first_leased_runs):
    """(n_runs, n_props) → (n_props, T) empirical CDF."""
    n_runs, n_props = first_leased_runs.shape
    cdf = np.zeros((n_props, T), dtype=np.float32)
    for t in range(T):
        leased = (first_leased_runs >= 0) & (first_leased_runs <= t)
        cdf[:, t] = leased.mean(axis=0)
    return cdf


# ── Choose 3 contrasting properties ──────────────────────────────────────────
# Score each property by area * location_score (high = attractive)
attractiveness = fixed_features[:, 0] * fixed_features[:, 1]
sorted_idx = np.argsort(attractiveness)
prop_best = int(sorted_idx[int(0.9 * N_PROPS)])    # top-10% attractive
prop_mid  = int(sorted_idx[int(0.5 * N_PROPS)])    # median
prop_worst = int(sorted_idx[int(0.1 * N_PROPS)])   # bottom-10%

contrast_props = [prop_best, prop_mid, prop_worst]
contrast_labels = ["Лучший (большая площадь, хорошая локация)",
                   "Медианный",
                   "Слабый (малая площадь, плохая локация)"]

# ── 1. Baseline Monte-Carlo runs ─────────────────────────────────────────────
print(f"\nБазовый прогон: {N_RUNS} итераций, T={T} шагов …")
first_leased_runs = np.empty((N_RUNS, N_PROPS), dtype=np.int32)
for r in range(N_RUNS):
    first_leased_runs[r] = _run_once(rates_opt)
    if (r + 1) % 50 == 0:
        print(f"  {r+1}/{N_RUNS}")

cdf = _compute_cdf(first_leased_runs)   # (N_PROPS, T)

# ── 2. Print stdout summary for selected properties ─────────────────────────
def _median_lease(fl_col):
    """Median first-leased step across runs; None if >50% never leased."""
    leased = fl_col[fl_col >= 0]
    return float(np.median(leased)) if len(leased) >= len(fl_col) * 0.5 else None

WINDOW_4W  = min(3,  T - 1)
WINDOW_12W = min(11, T - 1)
WINDOW_26W = min(25, T - 1)

print("\nСводка по контрастным объектам:")
for j, label in zip(contrast_props, contrast_labels):
    med = _median_lease(first_leased_runs[:, j])
    med_str = f"{int(med)}нед" if med is not None else "нет"
    print(f"  [{j:3d}] {label}")
    print(f"        rate={rates_opt[j]:.2f}  area={fixed_features[j,0]:.2f}  loc={fixed_features[j,1]:.2f}")
    print(f"        P(<=4нед)={cdf[j,WINDOW_4W]:.0%}  P(<=12нед)={cdf[j,WINDOW_12W]:.0%}  P(<=26нед)={cdf[j,WINDOW_26W]:.0%}  медиана={med_str}")

# ── 3. Rate sweep for contrast properties ────────────────────────────────────
print(f"\nSwoop ставок {RATE_GRID.tolist()} для {len(contrast_props)} объектов …")

# sweep_cdf[prop_idx][rate_idx] → (T,) CDF
sweep_cdf = {j: [] for j in contrast_props}
total_sweep = len(contrast_props) * len(RATE_GRID) * N_SWEEP
done = 0
for j in contrast_props:
    for rv in RATE_GRID:
        rates_sweep = rates_opt.copy()
        rates_sweep[j] = rv
        fl_runs = np.empty((N_SWEEP, N_PROPS), dtype=np.int32)
        for r in range(N_SWEEP):
            fl_runs[r] = _run_once(rates_sweep)
        done += N_SWEEP
        # CDF only for this property
        cdf_j = np.zeros(T, dtype=np.float32)
        for t in range(T):
            leased = (fl_runs[:, j] >= 0) & (fl_runs[:, j] <= t)
            cdf_j[t] = leased.mean()
        sweep_cdf[j].append(cdf_j)
        print(f"  [{j}] rate={rv:.1f}  P(<=12нед)={cdf_j[WINDOW_12W]:.0%}  ({done}/{total_sweep})")

# ── 4. Save summary JSON ─────────────────────────────────────────────────────
summary = []
for j in range(N_PROPS):
    med = _median_lease(first_leased_runs[:, j])
    summary.append({
        "property_id": j,
        "rate": round(float(rates_opt[j]), 4),
        "area": round(float(fixed_features[j, 0]), 4),
        "location_score": round(float(fixed_features[j, 1]), 4),
        "p_leased_by_4w":  round(float(cdf[j, WINDOW_4W]),  3),
        "p_leased_by_12w": round(float(cdf[j, WINDOW_12W]), 3),
        "p_leased_by_26w": round(float(cdf[j, WINDOW_26W]), 3),
        "median_lease_step": round(med, 1) if med is not None else None,
    })
with open(os.path.join(RESOURCES_DIR, "lease_time_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("\nСводка сохранена: resources/lease_time_summary.json")

# ── 5. Plot: CDF curves for 3 contrasting properties (baseline rates) ────────
weeks = np.arange(T)
curve_colors = [COLOR_BEST, COLOR_MEDIAN, COLOR_WORST]

fig, ax = plt.subplots(figsize=(11, 6))
for j, label, color in zip(contrast_props, contrast_labels, curve_colors):
    ax.plot(weeks, cdf[j], color=color, lw=2, label=f"{label}  (ставка {rates_opt[j]:.2f})")

p10 = np.percentile(cdf, 10, axis=0)
p90 = np.percentile(cdf, 90, axis=0)
ax.fill_between(weeks, p10, p90, alpha=0.18, color=COLOR_GREY, label="10–90% диапазон портфеля")

business_horizons(ax)

ax.set_xlabel("Неделя после выставления")
ax.set_ylabel("Вероятность сдачи")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_xlim(0, T - 1)
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
apply_clean_style(ax)
fig.tight_layout()
_path = os.path.join(RESOURCES_DIR, "lease_time_curves.png")
fig.savefig(_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"График сохранён: {_path}")

# ── 6. Plot: rate sweep trade-off ─────────────────────────────────────────────
n_contrast = len(contrast_props)
cmap_sweep = plt.cm.RdYlBu_r
n_rates = len(RATE_GRID)
sweep_colors = [cmap_sweep(i / max(1, n_rates - 1)) for i in range(n_rates)]

fig, axes = plt.subplots(1, n_contrast, figsize=(5 * n_contrast, 5), sharey=True)
if n_contrast == 1:
    axes = [axes]

for ax, j, label in zip(axes, contrast_props, contrast_labels):
    for i, (rv, sc) in enumerate(zip(RATE_GRID, sweep_colors)):
        cdj = sweep_cdf[j][i]
        ax.plot(weeks, cdj, color=sc, lw=2, label=f"ставка {rv:.1f}")
    ax.set_title(label, fontsize=9)
    ax.set_xlabel("Неделя")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    apply_clean_style(ax)

axes[0].set_ylabel("Вероятность сдачи")
fig.tight_layout()
_path = os.path.join(RESOURCES_DIR, "lease_time_rate_sweep.png")
fig.savefig(_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"График сохранён: {_path}")

# ── 7. Plot: portfolio-wide first-lease histogram ────────────────────────────
median_lease_per_prop = np.array([
    _median_lease(first_leased_runs[:, j]) if _median_lease(first_leased_runs[:, j]) is not None else T
    for j in range(N_PROPS)
])

bins = list(range(0, T + 4, 4))
fig, ax = plt.subplots(figsize=(11, 5))
counts, edges, patches = ax.hist(
    median_lease_per_prop, bins=bins, color=COLOR_MEDIAN, edgecolor="white", linewidth=0.6
)

never_leased = int((first_leased_runs < 0).all(axis=0).sum())
if never_leased > 0:
    patches[-1].set_facecolor(COLOR_WORST)
    patches[-1].set_label(f"Не сдан в горизонте {T} нед: {never_leased}")
    ax.legend(fontsize=9)

ax.set_xlabel("Медианная неделя первой сдачи")
ax.set_ylabel("Количество объектов")
ax.set_title(f"Распределение времени сдачи по портфелю (N={N_PROPS} объектов, оптимизированные ставки)")
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax.grid(axis="y", alpha=0.3)
apply_clean_style(ax)
fig.tight_layout()
_path = os.path.join(RESOURCES_DIR, "lease_time_histogram.png")
fig.savefig(_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"График сохранён: {_path}")

print("\nГотово.")
