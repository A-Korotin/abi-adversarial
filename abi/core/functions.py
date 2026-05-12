import numpy as np

def prospect_value_function(x: float, alpha=0.8, beta=0.8, lam=2.25) -> float:
    if x >= 0:
        return x ** alpha
    
    return -lam * ((-x) ** beta)

def prospect_np(x: np.ndarray, alpha: float=0.8, beta: float=0.8, lam: float=2.25) -> np.ndarray:
    res = np.zeros_like(x, dtype=np.float32)

    pos = x >= 0
    neg = x < 0

    res[pos] = x[pos] ** alpha
    res[neg] = -lam * ((-x[neg]) ** beta)

    return res

def softmax(x: np.ndarray) -> np.ndarray:
    exp_input = np.exp(x)
    exp_input_sum = np.sum(exp_input)

    return exp_input / exp_input_sum