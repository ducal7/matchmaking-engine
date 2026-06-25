"""Synthetic player generation.

All data in this project is synthetic and produced by this seeded generator --
there are no external datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ratings import Rating, RatingSystem


@dataclass
class Player:
    """A simulated player with a hidden true skill and an estimated rating."""

    pid: int
    true_skill: float
    rating: Rating
    matches_played: int = 0


def generate_players(
    n: int,
    rating_system: RatingSystem,
    seed: int,
    skill_mean: float = 1500.0,
    skill_sd: float = 300.0,
) -> list[Player]:
    """Create ``n`` players with normally distributed hidden true skills.

    Every player starts at the rating system's default rating; the simulation's
    job is to drive those estimates toward each player's hidden true skill.
    """

    rng = np.random.default_rng(seed)
    skills = rng.normal(skill_mean, skill_sd, n)
    return [
        Player(pid=i, true_skill=float(s), rating=rating_system.new_rating())
        for i, s in enumerate(skills)
    ]
