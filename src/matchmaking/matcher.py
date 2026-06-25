"""Queue matcher that trades off match quality against wait time.

A good match has a small skill gap, but forcing players to wait for a perfect
opponent is a bad experience. The matcher therefore *widens* the acceptable
skill window the longer a player has waited: a freshly queued player only
accepts near-equal opponents, while a long-waiting player accepts a much wider
spread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MatchConfig:
    """Parameters controlling the quality-vs-wait trade-off.

    The acceptable skill gap for a player who has waited ``w`` seconds is::

        min(base_window + widen_per_second * w, max_window)
    """

    base_window: float = 50.0
    widen_per_second: float = 25.0
    max_window: float = 1200.0


@dataclass
class QueueEntry:
    """A player waiting in the matchmaking queue."""

    rating: float
    enqueue_time: float
    payload: Any = None


def acceptable_gap(wait: float, cfg: MatchConfig) -> float:
    """Largest rating gap a player who has waited ``wait`` seconds accepts."""

    return min(cfg.base_window + cfg.widen_per_second * wait, cfg.max_window)


def find_match(
    entries: list[QueueEntry], now: float, cfg: MatchConfig
) -> tuple[int, int] | None:
    """Find the best acceptable pair in ``entries``.

    Returns the indices of the two entries to match, or ``None`` if no pair is
    currently acceptable. Candidate pairs are restricted to players adjacent in
    rating (sorting first), which both keeps the search cheap and guarantees the
    smallest possible gaps are considered. A pair is acceptable when the rating
    gap fits inside the window of the *longer-waiting* of the two players, so a
    player who has waited a long time can pull in a closer, fresher opponent.
    Among all acceptable pairs the one with the smallest gap is returned.
    """

    if len(entries) < 2:
        return None

    order = sorted(range(len(entries)), key=lambda idx: entries[idx].rating)
    best: tuple[int, int] | None = None
    best_gap: float | None = None
    for k in range(len(order) - 1):
        i = order[k]
        j = order[k + 1]
        gap = abs(entries[i].rating - entries[j].rating)
        wait = max(now - entries[i].enqueue_time, now - entries[j].enqueue_time)
        if gap <= acceptable_gap(wait, cfg):
            if best_gap is None or gap < best_gap:
                best = (i, j)
                best_gap = gap
    return best
