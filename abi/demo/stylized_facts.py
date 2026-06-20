"""
Валидация модели на стилизованных фактах рынка аренды.

Факт 1 — Естественная фрикционная вакантность:
    Рынок не насыщается мгновенно; даже при низких ставках часть объектов
    остаётся несданной в каждый момент. Ненулевой «пол» вакантности сохраняется
    даже в равновесии.

Факт 2 — Тяжёлый правый хвост сроков экспозиции:
    Большинство объектов сдаётся относительно быстро, но есть длинный хвост
    «зависших». Эмпирический survival лежит выше экспоненциального с той же
    средней → хвост тяжелее экспоненциального.

Запуск:
    python -m abi.demo.stylized_facts

Выводит:
    resources/stylized_fact_1_vacancy.png
    resources/stylized_fact_2_exposure_tail.png
    resources/stylized_facts_summary.json
"""

import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

# ── Presentation style (mirrors lease_time.py) ────────────────────────────────
COLOR_BEST   = '#2E7D5B'
COLOR_MEDIAN = '#1E40AF'
COLOR_WORST  = '#B45309'
COLOR_GREY   = '#6B7280'
COLOR_TEXT   = '#1A1A1A'
COLOR_FAINT  = '#D1D5DB'

plt.rcParams.update({
    'font.family':      ['DejaVu Sans'],
    'text.color':       COLOR_TEXT,
    'axes.labelcolor':  COLOR_TEXT,
    'axes.edgecolor':   COLOR_FAINT,
    'xtick.color':      COLOR_GREY,
    'ytick.color':      COLOR_GREY,
    'axes.facecolor':   'white',
    'figure.facecolor': 'white',
    'axes.grid':        False,
})


def _clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_FAINT)
    ax.spines['bottom'].set_color(COLOR_FAINT)
    ax.tick_params(length=0)


# ── Constants ─────────────────────────────────────────────────────────────────
N_PROPS    = 150
N_AGENTS   = 300
T          = 104
N_FEATURES = 3
K_VISIBLE  = 20

FEATURE_DIRECTIONS = np.array([-1.0, 1.0, 1.0], dtype=np.float32)
WEIGHTS            = np.array([0.4, 0.3, 0.3],  dtype=np.float32)

COMPETITOR_SIZES = DEFAULT_COMPETITOR_SIZES * 5   # [15, 10, 5] — mirrors estate.py
OUTSIDE_SIZE     = DEFAULT_OUTSIDE_SIZE             # 1.0
OUTSIDE_UTILITY  = 0.5                            # mirrors estate.py
UTILITY_SCALE    = 1.0                             # temperature β — mirrors estate.py
LAMBDA_PORTFOLIO = 0.6
LAMBDA_COMP      = 0.6

LEASE_CAPACITY = 1  # property leased when ≥1 tenant chooses it; M-independent

N_RUNS  = 200    # baseline Monte-Carlo runs (facts 1 timeseries & fact 2)
N_SWEEP = 100    # runs per rate level for sweep (fact 1 sweep)
RATE_GRID = np.round(np.arange(0.1, 1.0, 0.1), 2).astype(np.float32)  # 0.1 … 0.9
LOW_RATE  = 0.2  # uniform level for "low-rate ramp-up" experiment

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESOURCES_DIR = os.path.join(_PROJECT_ROOT, 'resources')
os.makedirs(RESOURCES_DIR, exist_ok=True)

# ── Fixed geometry (same seed as estate.py / lease_time.py) ──────────────────
rng_fixed = np.random.default_rng(52)
fixed_features = rng_fixed.uniform(0.3, 0.8, size=(N_PROPS, 2)).astype(np.float32)
# columns: [area, location_score]

competitor_features  = build_competitors()
competitor_log_sizes = build_competitor_log_sizes(COMPETITOR_SIZES)

# ── Load optimized rates or fall back to 0.5 ─────────────────────────────────
_json_path = os.path.join(RESOURCES_DIR, 're_optimization_result.json')
if os.path.exists(_json_path):
    with open(_json_path) as _f:
        _opt = json.load(_f)
    rates_opt = np.array(_opt['final_rates'], dtype=np.float32)
    print(f'Загружены оптимизированные ставки из {_json_path}')
else:
    rates_opt = np.full(N_PROPS, 0.5, dtype=np.float32)
    print('re_optimization_result.json не найден — используются ставки 0.5')

# ── Simulator & agents ────────────────────────────────────────────────────────
sim = RealEstateSimulator(
    T=T,
    K=K_VISIBLE,
    lr_ref=0.05,
    ref_market_weight=0.3,
    outside_utility=OUTSIDE_UTILITY,
    outside_size=OUTSIDE_SIZE,
    lambda_portfolio=LAMBDA_PORTFOLIO,
    lambda_comp=LAMBDA_COMP,
    competitor_sizes=COMPETITOR_SIZES,
    utility_scale=UTILITY_SCALE,
)

base_phi = MarketScenario(
    n_agents=N_AGENTS,
    ref_shift=np.zeros(N_FEATURES, dtype=np.float32),
    ref_std=0.1,
    switching_cost_scale=10,
)

# Agents fixed across all runs; only Gumbel noise varies.
agents = build_agents(base_phi, WEIGHTS, FEATURE_DIRECTIONS, n_props=N_PROPS, K=K_VISIBLE)


# ── Core helper ───────────────────────────────────────────────────────────────
def _run_once(rates: np.ndarray):
    """Single MC run → dict with first_leased_step and portfolio_share_history."""
    gumbel = sim.sample_gumbel(N_AGENTS)
    result = sim.run(
        rates=rates,
        fixed_features=fixed_features,
        competitor_features=competitor_features,
        agents_ref=agents['refs'],
        agents_w=agents['w'],
        agents_directions=agents['directions'],
        agents_switching_cost=agents['switching_costs'],
        agents_current_nest=agents['current_nest'].copy(),
        agents_current_inner=agents['current_inner'].copy(),
        agents_to_properties=agents['agent_to_properties'],
        gumbel=gumbel,
        disable_early_exit=True,     # full T steps — needed for time-series & tail
        lease_capacity=LEASE_CAPACITY,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FACT 1  —  Фрикционная вакантность
# ═══════════════════════════════════════════════════════════════════════════════
print('\n--- Факт 1: фрикционная вакантность ---')

# 1a. Временной ряд при низких равномерных ставках (rate = LOW_RATE)
rates_low = np.full(N_PROPS, LOW_RATE, dtype=np.float32)
share_history_runs = np.zeros((N_RUNS, T), dtype=np.float32)  # occupancy (= 1-vacancy)

print(f'  Временной ряд: {N_RUNS} прогонов при ставке {LOW_RATE} …')
for r in range(N_RUNS):
    res = _run_once(rates_low)
    h = res['portfolio_share_history']
    share_history_runs[r, :len(h)] = h
    if (r + 1) % 50 == 0:
        print(f'    {r + 1}/{N_RUNS}')

mean_share   = share_history_runs.mean(axis=0)   # (T,)
p10_share    = np.percentile(share_history_runs, 10, axis=0)
p90_share    = np.percentile(share_history_runs, 90, axis=0)
eq_share_low = float(mean_share[-20:].mean())    # equilibrium = last 20 steps

# Week at which mean_share first reaches 95% of its equilibrium value
target_95 = 0.95 * eq_share_low
weeks_to_95 = int(np.argmax(mean_share >= target_95)) if np.any(mean_share >= target_95) else T

print(f'  Равновесная заполняемость (ставка {LOW_RATE}): {eq_share_low:.1%}')
print(f'  Фрикционная вакантность:  {1 - eq_share_low:.1%}')
print(f'  Недель до 95% плато:       {weeks_to_95}')

# 1b. Sweep по уровням ставок → равновесная вакантность
print(f'\n  Sweep ставок {RATE_GRID.tolist()}, {N_SWEEP} прогонов/уровень …')
eq_vacancy_by_rate = {}
for rv in RATE_GRID:
    rates_r = np.full(N_PROPS, rv, dtype=np.float32)
    occ_arr = np.empty(N_SWEEP, dtype=np.float32)
    for r in range(N_SWEEP):
        res = _run_once(rates_r)
        h = res['portfolio_share_history']
        occ_arr[r] = float(h[-20:].mean()) if len(h) >= 20 else float(h.mean())
    mean_vac = float(1.0 - occ_arr.mean())
    eq_vacancy_by_rate[f'{rv:.1f}'] = round(mean_vac, 4)
    print(f'    ставка={rv:.1f}  вакантность={mean_vac:.1%}')

frictional_floor = eq_vacancy_by_rate[f'{RATE_GRID[0]:.1f}']
print(f'\n  Фрикционный пол вакантности (ставка {RATE_GRID[0]:.1f}): {frictional_floor:.1%}')


# ═══════════════════════════════════════════════════════════════════════════════
# FACT 2  —  Тяжёлый хвост сроков экспозиции
# ═══════════════════════════════════════════════════════════════════════════════
print('\n--- Факт 2: тяжёлый хвост сроков экспозиции ---')
print(f'  {N_RUNS} прогонов при оптимизированных ставках …')

all_first_leased = np.empty((N_RUNS, N_PROPS), dtype=np.int32)
for r in range(N_RUNS):
    res = _run_once(rates_opt)
    all_first_leased[r] = res['first_leased_step']
    if (r + 1) % 50 == 0:
        print(f'    {r + 1}/{N_RUNS}')

# Flatten: each (run, property) pair is one "lease time" observation.
flat = all_first_leased.ravel()
leased_mask   = flat >= 0
leased_times  = flat[leased_mask].astype(float)
never_share   = float((~leased_mask).mean())

mean_t   = float(leased_times.mean())
median_t = float(np.median(leased_times))
p90_t    = float(np.percentile(leased_times, 90))
p99_t    = float(np.percentile(leased_times, 99))

# Fisher–Pearson skewness (no scipy needed)
m2 = float(np.mean((leased_times - mean_t) ** 2))
m3 = float(np.mean((leased_times - mean_t) ** 3))
skewness = m3 / (m2 ** 1.5) if m2 > 0 else 0.0

mean_over_median   = mean_t / median_t if median_t > 0 else float('nan')
tail_gt_2x_median  = float(np.mean(leased_times > 2 * median_t))

print(f'\n  Среднее:             {mean_t:.1f} нед')
print(f'  Медиана:             {median_t:.1f} нед')
print(f'  P90:                 {p90_t:.1f} нед')
print(f'  P99:                 {p99_t:.1f} нед')
print(f'  Skewness (Fisher):   {skewness:.3f}')
print(f'  Mean / Median:       {mean_over_median:.3f}')
print(f'  P(t > 2·median):     {tail_gt_2x_median:.1%}')
print(f'  Доля несданных:      {never_share:.1%}')


# ═══════════════════════════════════════════════════════════════════════════════
# JSON summary
# ═══════════════════════════════════════════════════════════════════════════════
summary = {
    'fact_1_frictional_vacancy': {
        'low_rate_used':            float(LOW_RATE),
        'frictional_vacancy_floor': round(1 - eq_share_low, 4),
        'weeks_to_95pct_equilibrium': int(weeks_to_95),
        'vacancy_by_rate':          eq_vacancy_by_rate,
    },
    'fact_2_exposure_tail': {
        'mean_weeks':              round(mean_t, 2),
        'median_weeks':            round(median_t, 2),
        'p90':                     round(p90_t, 2),
        'p99':                     round(p99_t, 2),
        'skewness':                round(skewness, 4),
        'mean_over_median':        round(mean_over_median, 4),
        'tail_share_gt_2x_median': round(tail_gt_2x_median, 4),
        'never_leased_share':      round(never_share, 4),
    },
}
_summary_path = os.path.join(RESOURCES_DIR, 'stylized_facts_summary.json')
with open(_summary_path, 'w', encoding='utf-8') as _f:
    json.dump(summary, _f, indent=2, ensure_ascii=False)
print(f'\nСводка сохранена: {_summary_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Фрикционная вакантность
# ═══════════════════════════════════════════════════════════════════════════════
weeks = np.arange(T)

fig, (ax_ts, ax_sw) = plt.subplots(1, 2, figsize=(14, 5))

# Left: occupancy time-series at low rate
vacancy_mean = 1.0 - mean_share
vacancy_p10  = 1.0 - p90_share   # flip: high occupancy → low vacancy
vacancy_p90  = 1.0 - p10_share

ax_ts.fill_between(weeks, vacancy_p10, vacancy_p90,
                   alpha=0.20, color=COLOR_MEDIAN, label='10–90% диапазон')
ax_ts.plot(weeks, vacancy_mean, color=COLOR_MEDIAN, lw=2,
           label=f'Средняя вакантность (ставка={LOW_RATE})')
ax_ts.axhline(1.0 - eq_share_low, color=COLOR_WORST, lw=1.2, ls='--',
              label=f'Равновесный уровень ({1 - eq_share_low:.1%})')
if weeks_to_95 < T:
    ax_ts.axvline(weeks_to_95, color=COLOR_GREY, lw=1, ls=':',
                  label=f'95% плато — неделя {weeks_to_95}')

ax_ts.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax_ts.set_ylim(0, max(vacancy_p90.max() * 1.15, 0.15))
ax_ts.set_xlim(0, T - 1)
ax_ts.set_xlabel('Неделя', color=COLOR_GREY, fontsize=10)
ax_ts.set_ylabel('Вакантность', fontsize=10)
ax_ts.set_title('Динамика вакантности (низкие ставки)', fontsize=11)
ax_ts.legend(fontsize=8, loc='upper right')
_clean(ax_ts)

# Right: equilibrium vacancy vs rate level
rate_vals  = [float(k) for k in eq_vacancy_by_rate]
vac_vals   = [eq_vacancy_by_rate[k] for k in eq_vacancy_by_rate]

ax_sw.plot(rate_vals, vac_vals, color=COLOR_MEDIAN, lw=2, marker='o',
           markersize=6, label='Равновесная вакантность')
ax_sw.axhline(frictional_floor, color=COLOR_BEST, lw=1.2, ls='--',
              label=f'Фрикционный пол ({frictional_floor:.1%})')

ax_sw.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax_sw.set_xlim(RATE_GRID[0] - 0.05, RATE_GRID[-1] + 0.05)
ax_sw.set_ylim(0, min(1.0, max(vac_vals) * 1.25))
ax_sw.set_xlabel('Уровень ставки', color=COLOR_GREY, fontsize=10)
ax_sw.set_ylabel('Равновесная вакантность', fontsize=10)
ax_sw.set_title('Вакантность vs ставка', fontsize=11)
ax_sw.legend(fontsize=8)
_clean(ax_sw)

fig.suptitle('Стилизованный факт 1 — Естественная фрикционная вакантность',
             fontsize=12, y=1.01)
fig.tight_layout()
_p1 = os.path.join(RESOURCES_DIR, 'stylized_fact_1_vacancy.png')
fig.savefig(_p1, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'График сохранён: {_p1}')


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Тяжёлый хвост
# ═══════════════════════════════════════════════════════════════════════════════
fig, (ax_hist, ax_surv) = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram of lease times
bins = list(range(0, int(leased_times.max()) + 6, 4))
ax_hist.hist(leased_times, bins=bins, color=COLOR_MEDIAN, edgecolor='white',
             linewidth=0.5, density=True, alpha=0.85, label='Сроки экспозиции')
ax_hist.axvline(mean_t,   color=COLOR_WORST, lw=1.8, ls='--',
                label=f'Среднее = {mean_t:.1f}')
ax_hist.axvline(median_t, color=COLOR_BEST,  lw=1.8, ls='-.',
                label=f'Медиана = {median_t:.1f}')

ax_hist.set_xlabel('Неделя первой сдачи', color=COLOR_GREY, fontsize=10)
ax_hist.set_ylabel('Плотность', fontsize=10)
ax_hist.set_title('Распределение сроков экспозиции', fontsize=11)
ax_hist.legend(fontsize=8)
_clean(ax_hist)

# Annotation: skewness
ax_hist.text(0.97, 0.95,
             f'skewness = {skewness:.2f}\nmean/median = {mean_over_median:.2f}',
             transform=ax_hist.transAxes, ha='right', va='top',
             fontsize=8, color=COLOR_GREY,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

# Right: survival function on semi-log scale
sorted_t = np.sort(leased_times)
n_obs    = len(sorted_t)
# empirical CCDF: P(T > t) — Kaplan–Meier style (ignoring never-leased here)
# never-leased observations are handled as right-censored at T; since we
# only plot the distribution of leased times, we state never_share separately.
ecdf_x = np.concatenate([[0], sorted_t, [sorted_t[-1] + 1]])
ecdf_y = np.concatenate([[1.0],
                          1.0 - np.arange(1, n_obs + 1) / n_obs,
                          [0.0]])

# Exponential reference with the same mean
t_ref = np.linspace(0, sorted_t[-1], 400)
exp_surv = np.exp(-t_ref / mean_t)

ax_surv.plot(ecdf_x, np.maximum(ecdf_y, 1e-4), color=COLOR_MEDIAN, lw=2,
             label='Эмпирический survival')
ax_surv.plot(t_ref, np.maximum(exp_surv, 1e-4), color=COLOR_WORST, lw=1.5,
             ls='--', label=f'Exp(mean={mean_t:.1f}) — справочно')

ax_surv.set_yscale('log')
ax_surv.set_xlim(0, sorted_t[-1])
ax_surv.set_ylim(1e-4, 1.1)
ax_surv.set_xlabel('Неделя первой сдачи', color=COLOR_GREY, fontsize=10)
ax_surv.set_ylabel('P(T_lease > t)', fontsize=10)
ax_surv.set_title('Survival-функция (полу-лог)', fontsize=11)
ax_surv.legend(fontsize=8)
_clean(ax_surv)
ax_surv.spines['left'].set_color(COLOR_FAINT)

# Footnote: never-leased fraction
ax_surv.text(0.97, 0.06,
             f'Несданных за {T} нед: {never_share:.1%}',
             transform=ax_surv.transAxes, ha='right', va='bottom',
             fontsize=8, color=COLOR_GREY,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

fig.suptitle('Стилизованный факт 2 — Тяжёлый хвост сроков экспозиции',
             fontsize=12, y=1.01)
fig.tight_layout()
_p2 = os.path.join(RESOURCES_DIR, 'stylized_fact_2_exposure_tail.png')
fig.savefig(_p2, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'График сохранён: {_p2}')

print('\nГотово.')
