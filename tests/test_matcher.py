"""Tests for the wait-aware queue matcher."""

from matchmaking.matcher import MatchConfig, QueueEntry, acceptable_gap, find_match


def test_acceptable_gap_widens_with_wait():
    cfg = MatchConfig(base_window=50.0, widen_per_second=25.0, max_window=1200.0)
    assert acceptable_gap(0.0, cfg) == 50.0
    assert acceptable_gap(10.0, cfg) == 300.0
    assert acceptable_gap(0.0, cfg) < acceptable_gap(10.0, cfg)


def test_acceptable_gap_is_capped():
    cfg = MatchConfig(base_window=50.0, widen_per_second=25.0, max_window=200.0)
    assert acceptable_gap(1000.0, cfg) == 200.0


def test_window_widens_to_admit_distant_opponent():
    cfg = MatchConfig(base_window=50.0, widen_per_second=25.0)
    # Two players 300 rating points apart.
    # At t=0 (just queued) the gap exceeds the 50-point window -> no match.
    fresh = [
        QueueEntry(rating=1500.0, enqueue_time=0.0),
        QueueEntry(rating=1800.0, enqueue_time=0.0),
    ]
    assert find_match(fresh, now=0.0, cfg=cfg) is None

    # After waiting 20s the window is 50 + 25*20 = 550 > 300 -> they match.
    assert find_match(fresh, now=20.0, cfg=cfg) == (0, 1)


def test_matcher_prefers_smallest_gap():
    cfg = MatchConfig(base_window=1000.0)  # everything acceptable immediately
    entries = [
        QueueEntry(rating=1500.0, enqueue_time=0.0),
        QueueEntry(rating=1510.0, enqueue_time=0.0),
        QueueEntry(rating=1900.0, enqueue_time=0.0),
    ]
    pair = find_match(entries, now=0.0, cfg=cfg)
    assert pair is not None
    i, j = pair
    chosen = {round(entries[i].rating), round(entries[j].rating)}
    assert chosen == {1500, 1510}


def test_no_match_with_single_player():
    cfg = MatchConfig()
    assert find_match([QueueEntry(rating=1500.0, enqueue_time=0.0)], now=99.0, cfg=cfg) is None
