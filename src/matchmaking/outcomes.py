"""Match-outcome model driven by players' hidden true skill.

Outcomes are sampled from a Bradley-Terry / logistic model on the *true* skill
gap (not the estimated ratings). This is what a well-calibrated rating system
should eventually recover.
"""

from __future__ import annotations

import numpy as np


def win_probability(skill_a: float, skill_b: float, scale: float = 400.0) -> float:
    """Probability that ``a`` beats ``b`` given their hidden true skills.

    Uses the same logistic form (base-10, 400-point scale) as Elo, so a
    400-point true-skill edge corresponds to ~91% win probability.
    """

    return 1.0 / (1.0 + 10.0 ** ((skill_b - skill_a) / scale))


def play_match(
    skill_a: float, skill_b: float, rng: np.random.Generator, scale: float = 400.0
) -> float:
    """Sample a match result: ``1.0`` if ``a`` wins, ``0.0`` if ``b`` wins."""

    p = win_probability(skill_a, skill_b, scale)
    return 1.0 if rng.random() < p else 0.0
