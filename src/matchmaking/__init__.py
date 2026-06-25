"""Skill-based matchmaking simulator.

Public API exposes the pluggable rating systems, the queue matcher, and the
discrete-event simulation entry points.
"""

from .matcher import MatchConfig, QueueEntry, acceptable_gap, find_match
from .ratings import Elo, Glicko2, Rating, RatingSystem

__all__ = [
    "Rating",
    "RatingSystem",
    "Elo",
    "Glicko2",
    "MatchConfig",
    "QueueEntry",
    "acceptable_gap",
    "find_match",
]
