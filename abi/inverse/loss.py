import numpy as np

class MarketLoss:
    def __init__(self, target_share: float, our_idx: int = 0):
        self.target_share = target_share
        self.our_idx = our_idx

    def __call__(self, market_share: np.ndarray) -> float:
        return float((market_share[self.our_idx] - self.target_share) ** 2)
