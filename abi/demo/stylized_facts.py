"""
Валидация модели на стилизованных фактах рынка аренды.

Модель: ядро с исключительным занятием объекта и договорами аренды
(abi.forward.estate). Все статистики считаются после прогрева (burn-in),
чтобы исключить холодный старт: в t = 0 у агентов нет договоров, и они
разбирают объекты разом.

Факт 1 — Естественная фрикционная вакантность:
    Даже при низких ставках часть объектов остаётся несданной: договоры
    истекают, объект освобождается, а новый арендатор находится не сразу.
    Равновесная вакантность > 0 и растёт со ставкой.

Факт 2 — Тяжёлый правый хвост сроков экспозиции:
    Периоды вакантности (от освобождения до следующей сдачи) считаются
    по потоку после прогрева, с учётом цензурирования справа (Каплан — Мейер).
    Критерий: интенсивность сдачи убывает с длительностью экспозиции
    (отрицательная зависимость от длительности) ⇔ survival лежит выше
    экспоненциального ⇔ хвост тяжелее экспоненциального. Проверяется только
    внутри окна наблюдения, без экстраполяции за него.

Уровни (сколько именно процентов вакантности, сколько недель экспозиции)
здесь НЕ откалиброваны: демо проверяет качественные свойства модели.

Запуск (из корня проекта):
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
    DEFAULT_COMPETITOR_SIZES,
    DEFAULT_OUTSIDE_SIZE,
)
from abi.inverse.scenario import MarketScenario
from abi.calibration.moments import vacancy_spells, kaplan_meier, km_median, km_survival_at

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
K_VISIBLE  = 20

LEASE_MEAN_WEEKS = 104                 # средний срок договора (геометрический), недель
CHURN_PROB       = 1.0 / LEASE_MEAN_WEEKS
T_BURN           = 3 * LEASE_MEAN_WEEKS  # прогрев: ~3 средних срока договора
T_OBS            = 156                 # окно наблюдения после прогрева, недель
T                = T_BURN + T_OBS

FEATURE_DIRECTIONS = np.array([-1.0, 1.0, 1.0], dtype=np.float32)
WEIGHTS            = np.array([0.4, 0.3, 0.3],  dtype=np.float32)

COMPETITOR_SIZES = DEFAULT_COMPETITOR_SIZES * 5   # [15, 10, 5] — mirrors estate.py
OUTSIDE_SIZE     = DEFAULT_OUTSIDE_SIZE             # 1.0
OUTSIDE_UTILITY  = 0.5
UTILITY_SCALE    = 1.0                              # temperature β
SWITCHING_COST_SCALE = 10.0

N_RUNS  = 200    # Monte-Carlo прогоны (факт 1, временной ряд; факт 2)
N_SWEEP = 100    # прогонов на уровень ставки (факт 1, sweep)
RATE_GRID = np.round(np.arange(0.1, 1.0, 0.1), 2).astype(np.float32)  # 0.1 … 0.9
LOW_RATE  = 0.2

PLATEAU_TOL_PP = 0.02   # плато: скользящее среднее в пределах ±2 п.п. от равновесия
PLATEAU_WINDOW = 26     # окно скользящего среднего, недель

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESOURCES_DIR = os.path.join(_PROJECT_ROOT, 'resources')
os.makedirs(RESOURCES_DIR, exist_ok=True)

# ── Fixed geometry (same seed as estate.py / lease_time.py) ──────────────────
rng_fixed = np.random.default_rng(52)
fixed_features = rng_fixed.uniform(0.3, 0.8, size=(N_PROPS, 2)).astype(np.float32)
area = fixed_features[:, 0]   # columns: [area, location_score]

competitor_features = build_competitors()

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
    outside_utility=OUTSIDE_UTILITY,
    outside_size=OUTSIDE_SIZE,
    competitor_sizes=COMPETITOR_SIZES,
    utility_scale=UTILITY_SCALE,
    churn_prob=CHURN_PROB,
)

scenario = MarketScenario(
    n_agents=N_AGENTS,
    ref_center=0.5,
    ref_std=0.1,
    switching_cost_scale=SWITCHING_COST_SCALE,
)

# Agents fixed across all runs; only the stochastic streams vary.
agents = build_agents(scenario, WEIGHTS, FEATURE_DIRECTIONS,
                      n_props=N_PROPS, K=K_VISIBLE, seed=52)

_seed_seq = np.random.SeedSequence(20261114)


def _run_once(rates: np.ndarray) -> dict:
    """Single MC run with independent Gumbel / lease-expiry / order streams."""
    seed = int(_seed_seq.spawn(1)[0].generate_state(1)[0])
    return sim.run(
        rates=rates,
        fixed_features=fixed_features,
        competitor_features=competitor_features,
        agents_ref=agents['refs'],
        agents_w=agents['w'],
        agents_directions=agents['directions'],
        agents_switching_cost=agents['switching_costs'],
        agents_current_choice=agents['current_choice'],
        agents_to_properties=agents['agent_to_properties'],
        gumbel=sim.sample_gumbel(N_AGENTS, seed=seed),
        churn_u=sim.sample_churn(N_AGENTS, seed=seed),
        order=sim.sample_order(N_AGENTS, seed=seed),
    )


def _plateau_week(series: np.ndarray, level: float) -> int:
    """First week after which the rolling mean stays within ±tol of level (T if never)."""
    kernel = np.ones(PLATEAU_WINDOW) / PLATEAU_WINDOW
    rolling = np.convolve(series, kernel, mode='valid')     # rolling[i] = mean(series[i:i+W])
    inside = np.abs(rolling - level) <= PLATEAU_TOL_PP
    for i in range(len(inside)):
        if inside[i:].all():
            return i + PLATEAU_WINDOW - 1
    return len(series)


# ═══════════════════════════════════════════════════════════════════════════════
# FACT 1  —  Фрикционная вакантность
# ═══════════════════════════════════════════════════════════════════════════════
print('\n--- Факт 1: фрикционная вакантность ---')
print(f'  T = {T} нед: прогрев {T_BURN} + наблюдение {T_OBS}; '
      f'средний договор {LEASE_MEAN_WEEKS} нед')

# 1a. Временной ряд при низких равномерных ставках
rates_low = np.full(N_PROPS, LOW_RATE, dtype=np.float32)
vac_runs = np.zeros((N_RUNS, T), dtype=np.float32)

print(f'  Временной ряд: {N_RUNS} прогонов при ставке {LOW_RATE} …')
for r in range(N_RUNS):
    vac_runs[r] = 1.0 - _run_once(rates_low)['portfolio_share_history']
    if (r + 1) % 50 == 0:
        print(f'    {r + 1}/{N_RUNS}')

vac_mean = vac_runs.mean(axis=0)
vac_p10  = np.percentile(vac_runs, 10, axis=0)
vac_p90  = np.percentile(vac_runs, 90, axis=0)

eq_vac_low = float(vac_mean[T_BURN:].mean())
plateau_week = _plateau_week(vac_mean, eq_vac_low)

# Stationarity check inside the observation window: first half vs second half
half = T_BURN + T_OBS // 2
drift_pp = float(vac_mean[half:].mean() - vac_mean[T_BURN:half].mean())
is_stationary = abs(drift_pp) <= PLATEAU_TOL_PP

print(f'  Равновесная вакантность (ставка {LOW_RATE}): {eq_vac_low:.1%}')
print(f'  Выход на плато (±{PLATEAU_TOL_PP:.0%}): неделя {plateau_week}'
      + ('' if plateau_week <= T_BURN else '  ⚠ позже конца прогрева — увеличьте T_BURN'))
print(f'  Дрейф внутри окна наблюдения: {drift_pp:+.2%} '
      f'({"стационарно" if is_stationary else "НЕ стационарно — увеличьте T_BURN"})')

# 1b. Sweep по уровням ставок → равновесная вакантность (окно наблюдения)
print(f'\n  Sweep ставок {[round(float(x), 1) for x in RATE_GRID]}, {N_SWEEP} прогонов/уровень …')
eq_vac_by_rate = {}
for rv in RATE_GRID:
    rates_r = np.full(N_PROPS, rv, dtype=np.float32)
    vals = np.array([1.0 - _run_once(rates_r)['portfolio_share_history'][T_BURN:].mean()
                     for _ in range(N_SWEEP)])
    eq_vac_by_rate[f'{rv:.1f}'] = {
        'mean': round(float(vals.mean()), 4),
        'p10':  round(float(np.percentile(vals, 10)), 4),
        'p90':  round(float(np.percentile(vals, 90)), 4),
    }
    print(f'    ставка={rv:.1f}  вакантность={vals.mean():.1%}  '
          f'[{np.percentile(vals, 10):.1%} … {np.percentile(vals, 90):.1%}]')

vac_means = [v['mean'] for v in eq_vac_by_rate.values()]
frictional_floor = vac_means[0]
monotone = all(b >= a - 0.005 for a, b in zip(vac_means, vac_means[1:]))
print(f'\n  Фрикционный пол вакантности (ставка {RATE_GRID[0]:.1f}): {frictional_floor:.1%}')
print(f'  Вакантность не убывает по ставке: {"да" if monotone else "НЕТ"}')


# ═══════════════════════════════════════════════════════════════════════════════
# FACT 2  —  Тяжёлый хвост сроков экспозиции (периоды вакантности после прогрева)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n--- Факт 2: тяжёлый хвост сроков экспозиции ---')
print(f'  {N_RUNS} прогонов при оптимизированных ставках …')

durations_all, observed_all = [], []
for r in range(N_RUNS):
    occ = _run_once(rates_opt)['occupied_history']
    d, o = vacancy_spells(occ, burn_in=T_BURN)
    durations_all.append(d)
    observed_all.append(o)
    if (r + 1) % 50 == 0:
        print(f'    {r + 1}/{N_RUNS}')

durations = np.concatenate(durations_all)
observed  = np.concatenate(observed_all)
n_spells  = int(len(durations))
censored_share = float(1.0 - observed.mean()) if n_spells else float('nan')

km_t, km_S = kaplan_meier(durations, observed)
median_km = km_median(km_t, km_S)

# Exponential with the censoring-aware MLE rate: events / total time at risk
exp_rate = float(observed.sum() / durations.sum())
exp_mean = 1.0 / exp_rate

# Duration dependence inside the observation window (censoring-aware):
# piecewise-constant hazard = events in [a, b) / time at risk in [a, b).
# Decreasing hazard  <=>  survival lies above the exponential  <=>  heavier tail.
def interval_hazard(d: np.ndarray, o: np.ndarray, a: float, b: float) -> float:
    at_risk = np.clip(np.minimum(d, b) - a, 0, None).sum()
    events = np.sum(o & (d > a) & (d <= b))
    return float(events / at_risk) if at_risk > 0 else float('nan')

edges = np.linspace(0, T_OBS, 4)          # три равных интервала окна наблюдения
hazards = [interval_hazard(durations, observed, a, b) for a, b in zip(edges[:-1], edges[1:])]
hazard_ratio = hazards[-1] / hazards[0] if hazards[0] > 0 else float('nan')
heavier_tail = bool(hazard_ratio < 0.8)   # убывание интенсивности хотя бы на 20%

S_km_end  = km_survival_at(km_t, km_S, T_OBS - 1)
S_exp_end = float(np.exp(-exp_rate * (T_OBS - 1)))

completed = durations[observed].astype(float)
skewness = float(((completed - completed.mean()) ** 3).mean()
                 / (completed.var() ** 1.5)) if completed.var() > 0 else 0.0

print(f'\n  Периодов вакантности:          {n_spells} (цензурировано {censored_share:.1%})')
print(f'  Медиана KM:                    {median_km:.1f} нед')
print(f'  Exp (MLE с цензурированием):   среднее {exp_mean:.1f} нед')
for (a, b), h in zip(zip(edges[:-1], edges[1:]), hazards):
    print(f'  Интенсивность сдачи [{a:3.0f}, {b:3.0f}) нед: {h:.4f} в неделю')
print(f'  Отношение интенсивностей (поздний / ранний): {hazard_ratio:.2f}')
print(f'  S(t = {T_OBS - 1}): KM = {S_km_end:.3f}, Exp = {S_exp_end:.3f}')
print(f'  Skewness завершённых периодов: {skewness:.2f}')
print(f'  Хвост тяжелее экспоненциального (интенсивность убывает): {"да" if heavier_tail else "НЕТ"}')
if censored_share > 0.5:
    print(f'  ⚠ цензурировано {censored_share:.0%} периодов: окно наблюдения короче типичной '
          f'экспозиции; выводы о хвосте — только в пределах {T_OBS} нед')


# ═══════════════════════════════════════════════════════════════════════════════
# JSON summary
# ═══════════════════════════════════════════════════════════════════════════════
summary = {
    'setup': {
        'T_burn': T_BURN, 'T_obs': T_OBS, 'lease_mean_weeks': LEASE_MEAN_WEEKS,
        'n_props': N_PROPS, 'n_agents': N_AGENTS, 'n_runs': N_RUNS, 'n_sweep': N_SWEEP,
    },
    'fact_1_frictional_vacancy': {
        'low_rate_used':           float(LOW_RATE),
        'equilibrium_vacancy_low': round(eq_vac_low, 4),
        'plateau_week':            int(plateau_week),
        'drift_in_obs_window_pp':  round(drift_pp, 4),
        'stationary':              bool(is_stationary),
        'vacancy_by_rate':         eq_vac_by_rate,
        'frictional_floor':        round(frictional_floor, 4),
        'monotone_in_rate':        bool(monotone),
    },
    'fact_2_exposure_tail': {
        'n_spells':                n_spells,
        'censored_share':          round(censored_share, 4),
        'km_median_weeks':         round(median_km, 2),
        'exp_mle_mean_weeks':      round(exp_mean, 2),
        'hazard_by_interval':      {f'{a:.0f}-{b:.0f}': round(h, 5)
                                    for a, b, h in zip(edges[:-1], edges[1:], hazards)},
        'hazard_ratio_late_early': round(hazard_ratio, 4),
        'S_km_at_window_end':      round(S_km_end, 4),
        'S_exp_at_window_end':     round(S_exp_end, 4),
        'skewness_completed':      round(skewness, 4),
        'heavier_than_exponential': bool(heavier_tail),
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

ax_ts.axvspan(0, T_BURN, color=COLOR_FAINT, alpha=0.35, lw=0, label='Прогрев (не учитывается)')
ax_ts.fill_between(weeks, vac_p10, vac_p90, alpha=0.20, color=COLOR_MEDIAN,
                   label='10–90% диапазон')
ax_ts.plot(weeks, vac_mean, color=COLOR_MEDIAN, lw=2,
           label=f'Средняя вакантность (ставка={LOW_RATE})')
ax_ts.hlines(eq_vac_low, T_BURN, T - 1, color=COLOR_WORST, lw=1.4, ls='--',
             label=f'Равновесие по окну наблюдения ({eq_vac_low:.1%})')
if plateau_week < T:
    ax_ts.axvline(plateau_week, color=COLOR_GREY, lw=1, ls=':',
                  label=f'Плато ±{PLATEAU_TOL_PP:.0%} — неделя {plateau_week}')

ax_ts.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax_ts.set_ylim(0, max(float(vac_p90.max()) * 1.15, 0.15))
ax_ts.set_xlim(0, T - 1)
ax_ts.set_xlabel('Неделя', color=COLOR_GREY, fontsize=10)
ax_ts.set_ylabel('Вакантность (по площади)', fontsize=10)
ax_ts.set_title('Динамика вакантности (низкие ставки)', fontsize=11)
ax_ts.legend(fontsize=8, loc='lower right')
_clean(ax_ts)

rate_vals = [float(k) for k in eq_vac_by_rate]
p10_vals  = [v['p10'] for v in eq_vac_by_rate.values()]
p90_vals  = [v['p90'] for v in eq_vac_by_rate.values()]
ax_sw.fill_between(rate_vals, p10_vals, p90_vals, alpha=0.20, color=COLOR_MEDIAN,
                   label='10–90% по прогонам')
ax_sw.plot(rate_vals, vac_means, color=COLOR_MEDIAN, lw=2, marker='o', markersize=6,
           label='Равновесная вакантность')
ax_sw.axhline(frictional_floor, color=COLOR_BEST, lw=1.2, ls='--',
              label=f'Фрикционный пол ({frictional_floor:.1%})')

ax_sw.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax_sw.set_xlim(RATE_GRID[0] - 0.05, RATE_GRID[-1] + 0.05)
ax_sw.set_ylim(0, min(1.0, max(p90_vals) * 1.2))
ax_sw.set_xlabel('Уровень ставки', color=COLOR_GREY, fontsize=10)
ax_sw.set_ylabel('Равновесная вакантность', fontsize=10)
ax_sw.set_title('Вакантность vs ставка (после прогрева)', fontsize=11)
ax_sw.legend(fontsize=8, loc='upper left')
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

bins = np.arange(0, int(completed.max()) + 5, 4)
ax_hist.hist(completed, bins=bins, color=COLOR_MEDIAN, edgecolor='white', linewidth=0.5,
             density=True, alpha=0.85, label='Завершённые периоды вакантности')
if np.isfinite(median_km):
    ax_hist.axvline(median_km, color=COLOR_BEST, lw=1.8, ls='-.',
                    label=f'Медиана KM = {median_km:.1f}')
ax_hist.axvline(exp_mean, color=COLOR_WORST, lw=1.8, ls='--',
                label=f'Среднее Exp (MLE) = {exp_mean:.1f}')
ax_hist.set_xlabel('Срок экспозиции, недель', color=COLOR_GREY, fontsize=10)
ax_hist.set_ylabel('Плотность', fontsize=10)
ax_hist.set_title('Распределение сроков экспозиции', fontsize=11)
ax_hist.legend(fontsize=8)
_clean(ax_hist)
ax_hist.text(0.97, 0.60,
             f'периодов: {n_spells}\nцензурировано: {censored_share:.1%}\n'
             f'интенсивность поздн./ранн. = {hazard_ratio:.2f}',
             transform=ax_hist.transAxes, ha='right', va='top', fontsize=8, color=COLOR_GREY,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

km_x = np.concatenate([[0], km_t])
km_y = np.concatenate([[1.0], km_S])
t_ref = np.linspace(0, T_OBS, 400)

ax_surv.step(km_x, np.maximum(km_y, 1e-4), where='post', color=COLOR_MEDIAN, lw=2,
             label='Каплан — Мейер')
ax_surv.plot(t_ref, np.maximum(np.exp(-exp_rate * t_ref), 1e-4), color=COLOR_WORST,
             lw=1.5, ls='--', label=f'Exp(среднее={exp_mean:.1f}) — та же интенсивность')

ax_surv.set_yscale('log')
ax_surv.set_xlim(0, T_OBS)
ax_surv.set_ylim(max(1e-3, float(km_S[km_S > 0].min()) * 0.5) if np.any(km_S > 0) else 1e-3, 1.1)
ax_surv.set_xlabel('Срок экспозиции, недель', color=COLOR_GREY, fontsize=10)
ax_surv.set_ylabel('P(экспозиция > t)', fontsize=10)
ax_surv.set_title('Survival-функция (полу-лог)', fontsize=11)
ax_surv.legend(fontsize=8)
_clean(ax_surv)

fig.suptitle('Стилизованный факт 2 — Тяжёлый хвост сроков экспозиции',
             fontsize=12, y=1.01)
fig.tight_layout()
_p2 = os.path.join(RESOURCES_DIR, 'stylized_fact_2_exposure_tail.png')
fig.savefig(_p2, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'График сохранён: {_p2}')

print('\nГотово.')