"""
Market scenario: fixed parameters describing tenant population.

Used by build_agents() to generate a heterogeneous population of tenant agents.
In the simplified MVP, the scenario is fixed (not optimized by an adversary).
"""

from dataclasses import dataclass


@dataclass
class MarketScenario:
    """Fixed market-scenario parameters for tenant agent generation.

    Attributes
    ----------
    n_agents : int
        Number of tenant agents in the simulation.
    ref_center : float
        Mean of agent reference-point distribution (all features, [0, 1]).
    ref_std : float
        Std of agent reference-point distribution.
    switching_cost_scale : float
        Multiplier on per-agent switching costs (higher = stickier tenants).
    """
    n_agents: int
    ref_center: float = 0.5
    ref_std: float = 0.1
    switching_cost_scale: float = 0.3
