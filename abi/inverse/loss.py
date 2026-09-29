"""
Loss function for real-estate rate optimisation.

Phased loss to avoid gradient conflict between rate maximisation and
occupancy restoration:

    infeasible (occupancy < threshold):
        loss = penalty * shortage²       (push toward feasibility, no rate term)
    feasible (occupancy >= threshold):
        loss = -mean_rate                (maximise rental income)

where shortage = threshold - occupancy.
"""

import numpy as np


class RealEstateLoss:
    """Phased loss for occupancy-constrained rate maximisation.

    Parameters
    ----------
    occupancy_threshold : float
        Minimum acceptable area-weighted occupancy (e.g. 0.60).
    penalty : float
        Coefficient on the quadratic shortage penalty (infeasible phase).
    """

    def __init__(
        self,
        occupancy_threshold: float = 0.7,
        penalty: float = 10.0,
    ):
        self.occupancy_threshold = occupancy_threshold
        self.penalty = penalty

    def __call__(self, occupancy: float, mean_rate: float) -> float:
        shortage = max(0.0, self.occupancy_threshold - occupancy)
        if occupancy < self.occupancy_threshold:
            return float(self.penalty * shortage ** 2)
        return float(-mean_rate)

    def is_feasible(self, occupancy: float) -> bool:
        return occupancy >= self.occupancy_threshold
