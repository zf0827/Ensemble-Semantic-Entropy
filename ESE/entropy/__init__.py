"""
Semantic entropy calculation module.

Provides four entropy metrics:
1. Predictive Entropy (MC) - Monte Carlo estimate
2. Predictive Entropy (Rao) - normalized version
3. Semantic Entropy (Standard/Soft)
4. Discrete Semantic Entropy (Frequency/Hard)
"""

from .semantic_entropy import (
    semantic_entropy,
    calc_predictive_entropy_mc,
    calc_predictive_entropy_rao,
    calc_semantic_entropy,
    calc_discrete_semantic_entropy,
)

__all__ = [
    "semantic_entropy",
    "calc_predictive_entropy_mc",
    "calc_predictive_entropy_rao",
    "calc_semantic_entropy",
    "calc_discrete_semantic_entropy",
]

__version__ = "0.2.0"

