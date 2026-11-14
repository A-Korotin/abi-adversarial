"""
Moment computation for simulated-method-of-moments calibration.

All functions work on the occupancy matrix returned by RealEstateSimulator.run()
(key "occupied_history", shape (T, N_PROPS), bool) and know nothing about the
simulator itself.
"""

import numpy as np


def occupancy(occupied: np.ndarray, area: np.ndarray, burn_in: int) -> dict:
    """Equilibrium occupancy after burn-in: area-weighted and by count."""
    occ = occupied[burn_in:]
    by_area = (occ * area).sum(axis=1) / area.sum()
    return {
        "occupancy_area": float(by_area.mean()),
        "occupancy_area_std": float(by_area.std()),   # разброс одного «снимка»
        "occupancy_count": float(occ.mean()),
    }


def stay_probability(occupied: np.ndarray, burn_in: int) -> float:
    """P(lot occupied at t+1 | occupied at t) after burn-in."""
    occ = occupied[burn_in:]
    n_occ = occ[:-1].sum()
    return float((occ[:-1] & occ[1:]).sum() / n_occ) if n_occ > 0 else float("nan")


def vacancy_spells(occupied: np.ndarray, burn_in: int):
    """Extract vacancy spells that start at or after burn_in (flow sample).

    Returns
    -------
    durations : (n_spells,) int — spell length in steps
    observed  : (n_spells,) bool — True if the spell ended before T,
                False if right-censored (lot still vacant at the end)
    """
    T, N = occupied.shape
    durations, observed = [], []
    for j in range(N):
        vacant = ~occupied[:, j]
        t = 0
        while t < T:
            if not vacant[t]:
                t += 1
                continue
            start = t
            while t < T and vacant[t]:
                t += 1
            if start >= burn_in:
                durations.append(t - start)
                observed.append(t < T)
    return np.asarray(durations, dtype=np.int64), np.asarray(observed, dtype=bool)


def kaplan_meier(durations: np.ndarray, observed: np.ndarray):
    """Kaplan–Meier survival curve S(t) = P(duration > t).

    Returns event times (sorted) and S evaluated right after each of them.
    """
    times = np.unique(durations[observed])
    S = np.empty(len(times))
    s = 1.0
    for k, t in enumerate(times):
        n_at_risk = np.sum(durations >= t)
        d = np.sum((durations == t) & observed)
        s *= 1.0 - d / n_at_risk
        S[k] = s
    return times, S


def km_survival_at(times: np.ndarray, S: np.ndarray, x: float) -> float:
    """S(x): survival just after x (step function)."""
    idx = np.searchsorted(times, x, side="right") - 1
    return 1.0 if idx < 0 else float(S[idx])


def km_median(times: np.ndarray, S: np.ndarray) -> float:
    """Smallest t with S(t) <= 0.5; NaN if the curve never reaches 0.5."""
    below = np.nonzero(S <= 0.5)[0]
    return float(times[below[0]]) if len(below) else float("nan")


def compute_moments(occupied: np.ndarray, area: np.ndarray,
                    burn_in: int, tail_steps: int = 26) -> dict:
    """All three calibration moments plus diagnostics for one simulation run."""
    moments = occupancy(occupied, area, burn_in)
    durations, observed = vacancy_spells(occupied, burn_in)

    if observed.sum() == 0:
        median, tail = float("nan"), float("nan")
    else:
        times, S = kaplan_meier(durations, observed)
        median = km_median(times, S)
        tail = km_survival_at(times, S, tail_steps)

    moments.update({
        "median_exposure": median,
        "p_exposure_gt_tail": tail,
        "n_spells": int(len(durations)),
        "frac_censored": float(1.0 - observed.mean()) if len(observed) else float("nan"),
        "stay_prob": stay_probability(occupied, burn_in),
    })
    return moments