import numpy as np

class MarketLoss:
    def __init__(self, target_share: float, our_idx: int = 0):
        self.target_share = target_share
        self.our_idx = our_idx

    def __call__(self, market_share: np.ndarray) -> float:
        return float((market_share[self.our_idx] - self.target_share) ** 2)


class RealEstateLoss:
    def __init__(
        self,
        occupancy_threshold: float = 0.7,
        penalty_lin: float = 2.0,
        penalty_quad: float = 10.0,
        buffer_penalty: float = 1.0,
    ):
        self.occupancy_threshold = occupancy_threshold
        self.penalty_lin = penalty_lin
        self.penalty_quad = penalty_quad
        self.buffer_penalty = buffer_penalty

    def __call__(self, occupancy: float, mean_rate: float) -> float:
        shortage = max(0.0, self.occupancy_threshold - occupancy)
        if occupancy < self.occupancy_threshold:
            # Phase 1 (infeasible): augmented penalty only — no rate term.
            # Linear keeps gradient nonzero even at large shortage;
            # quadratic amplifies pressure further from the boundary.
            return float(self.penalty_lin * shortage + self.penalty_quad * shortage ** 2)
        # Phase 2 (feasible): maximise rental rate with a small buffer penalty.
        return float(-mean_rate + self.buffer_penalty * shortage ** 2)

    def is_feasible(self, occupancy: float) -> bool:
        return occupancy >= self.occupancy_threshold