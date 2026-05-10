import numpy as np

from abi.forward.simulator import VectorizedSimulator
from abi.inverse.bootstrap import build_providers, build_agents
from abi.inverse.generator import Generator
from abi.inverse.loss import MarketLoss
from abi.inverse.sampler import MarketScenario, ScenarioSampler
from abi.inverse.train import train_adversarial_simple

if __name__ == '__main__':
    rules = {
        'price': {'min': 0, 'max': 100, 'expected': 50, 'std': 15},
        'connection_speed': {'min': 0, 'max': 1_000, 'expected': 100, 'std': 15},
        'quality': {'min': 0, 'max': 100, 'expected': 50, 'std': 15},
    }

    N_AGENTS = 5_000
    T = 104
    LR = 0.05
    FEATURES = ['price', 'connection_speed', 'quality']
    FEATURE_DIRECTIONS = np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    WEIGHTS = np.array([0.15, 0.25, 0.3], dtype=np.float32) / 0.7

    phi = MarketScenario(
        n_agents=N_AGENTS,
        ref_shift=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        ref_std=0.1,
        switching_cost_scale=0.3
    )
    sampler_sigma = np.array([
        0.05, 0.05, 0.05,  # ref_shift
        0.02,  # ref_std
        0.05  # switching_cost
    ], dtype=np.float32)
    sampler = ScenarioSampler(phi, sampler_sigma, n_agents=N_AGENTS)

    # price, connection, quality
    xi_initial = np.array([[1, 0.2, 0.2]])

    xi_competitors = np.array([
        [0.4, 0.1, 0.4],
        [0.7, 0.6, 0.5],
    ])

    generator_sigma = np.array([
        [0.02, 0.02, 0.02]
    ])

    generator = Generator(base_xi=xi_initial, sigma=generator_sigma)

    simulator = VectorizedSimulator(T, LR)

    target_share = 0.45
    lf = MarketLoss(target_share)

    result = train_adversarial_simple(
        simulator=simulator,
        generator=generator,
        discriminator=sampler,
        loss_fn=lf,
        outer_steps=20,
        build_providers=lambda x: build_providers(x, competitors=xi_competitors),
        build_agents=lambda p: build_agents(p, WEIGHTS, FEATURE_DIRECTIONS, N_AGENTS)
    )

    last_xi = result['xi'][-1]
    print(last_xi)