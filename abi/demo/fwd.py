import numpy as np
from abi.core.entities import Provider, Agent
from abi.forward.simulator import VectorizedSimulator
from typing import List

rules = {
    'price': {
        'min': 0,
        'max': 100,
        'expected': 50,
        'std': 15
    },
    'connection_speed': {
        'min': 0,
        'max': 1_000,
        'expected': 100,
        'std': 15
    },
    'quality': {
        'min': 0,
        'max': 100,
        'expected': 50,
        'std': 15
    },
}

def normalize(value: float, key: str) -> float:
    mn = rules[key]['min']
    mx = rules[key]['max']
    return (value - mn) / (mx - mn)

def clip_to_bounds(value: float, key: str) -> float:
    mn = rules[key]['min']
    mx = rules[key]['max']
    return float(np.clip(value, mn, mx))

def utility_range(agents: List[Agent], providers: List[Provider]):
    values = np.zeros(len(agents) * len(providers))
    idx = 0
    for agent in agents:
        for provider in providers:
            u = agent.utility_for(provider.p)
            values[idx] = u
            idx += 1

    return np.round(values.min(), 3), np.round(values.max(), 3), np.round(values.std(), 3)

def init_agents(amount: int, providers: List[Provider]) -> List[Agent]:
    weights = np.array([0.15, 0.25, 0.3]) / 0.7
    directions = np.array([-1, 1, 1])
    res = []

    for _ in range(amount):
        price_ref = np.random.normal(rules['price']['expected'], rules['price']['std'])
        speed_ref = np.random.normal(rules['connection_speed']['expected'], rules['connection_speed']['std'])
        quality_ref = np.random.normal(rules['quality']['expected'], rules['quality']['std'])

        price_ref = clip_to_bounds(price_ref, 'price')
        speed_ref = clip_to_bounds(speed_ref, 'connection_speed')
        quality_ref = clip_to_bounds(quality_ref, 'quality')

        ref = np.array([
            normalize(price_ref, 'price'),
            normalize(speed_ref, 'connection_speed'),
            normalize(quality_ref, 'quality')
        ])

        res.append(Agent(weights, ref, directions))

    sample_agents = np.random.choice(res, amount // 5)
    u_min, u_max, u_std = utility_range(sample_agents, providers)
    u_range = u_max - u_min

    print(f"Utility function: min={u_min}, max={u_max}, std={u_std}. Range={u_range}")

    for agent in res:
        agent.switching_cost = np.random.normal(u_std, 0.3 * u_std)

    return res

def init_providers() -> List[Provider]:
    res = []

    res.append(Provider("Wir", np.array(
        [
            normalize(70, 'price'),
            normalize(600, 'connection_speed'),
            normalize(70, 'quality')
        ])))
    
    res.append(Provider("Competitior1", np.array(
        [
            normalize(40, 'price'),
            normalize(100, 'connection_speed'),
            normalize(40, 'quality')
        ])))
    
    res.append(Provider("Competitior2", np.array(
        [
            normalize(90, 'price'),
            normalize(900, 'connection_speed'),
            normalize(75, 'quality')
        ])))

    return res

if __name__ == "__main__":
    M = 10_000
    T = 104
    LR = 0.1
    providers = init_providers()
    agents = init_agents(M, providers)

    simulator = VectorizedSimulator(100, 0.1)

    eq_step, eq_shares, market_share_history = simulator.run(agents, providers)

    print(f"Equilibrium occurred on step: {eq_step}. Equilibrium market shares: {np.round(eq_shares, 2)}")