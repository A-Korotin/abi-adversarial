import numpy as np
from typing import List
from abi.core.functions import prospect_np

class Provider:
    def __init__(self, name: str, p: np.array):
        self.name = name
        self.p = p

    def __repr__(self):
        return f"Provider '{self.name}': {self.p}"
    
class Agent:
    def __init__(self, w: np.array, ref: np.array, directions: np.array):
        self.w = w 
        self.ref = ref
        self.directions = directions
        self.current_provider = -1
        self.switching_cost = 0

    def utility_for(self, option: np.array):
        deltas = (option - self.ref) * self.directions
        prospect_activation = prospect_np(deltas)

        return (prospect_activation @ self.w).item()
    
    def utility_for_matrix(self, options: np.array):
        deltas = (options - self.ref) * self.directions
        prospect_activation = prospect_np(deltas)
        utilities = prospect_activation @ self.w
        return utilities.flatten()

    
    def activated_output_for(self, providers: List[Provider], actiavtion_func) -> np.array:
        utilities = np.array([self.utility_for(provider.p) for provider in providers])
        return actiavtion_func(utilities)
    
    def __repr__(self):
        return f"Agent[w={np.round(self.w, 2)}, ref={np.round(self.ref, 2)}, sw_cost={self.switching_cost}, current_provider={self.current_provider}]"