import numpy as np

class MarketLoss:
    def __init__(self, target_share: float, our_idx: int = 0):
        self.target_share = target_share
        self.our_idx = our_idx

    def __call__(self, market_share: np.ndarray) -> float:
        return float((market_share[self.our_idx] - self.target_share) ** 2)


class RealEstateLoss:
    def __init__(self, target_rate: float, occupancy_threshold: float = 0.8, penalty: float = 10.0):
        self.target_rate = target_rate
        self.occupancy_threshold = occupancy_threshold
        self.penalty = penalty

    def __call__(self, occupancy: float, mean_rate: float) -> float:
        shortage = max(0.0, self.occupancy_threshold - occupancy)

        return float((mean_rate - self.target_rate) ** 2 + self.penalty * (shortage ** 2))

    def is_feasible(self, occupancy: float) -> bool:
        return occupancy >= self.occupancy_threshold