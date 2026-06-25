"""Tests for the Elo rating system."""

from matchmaking.ratings import Elo, Rating


def test_expected_score_symmetry_and_equal_ratings():
    elo = Elo()
    a = Rating(1500.0)
    b = Rating(1500.0)
    assert elo.expected_score(a, b) == 0.5
    # Expected scores of the two players must sum to 1.
    c = Rating(1700.0)
    assert abs(elo.expected_score(a, c) + elo.expected_score(c, a) - 1.0) < 1e-12


def test_winner_gains_loser_loses():
    elo = Elo(k=32.0)
    a = Rating(1500.0)
    b = Rating(1500.0)
    elo.update(a, b, score_a=1.0)  # a wins
    assert a.rating > 1500.0
    assert b.rating < 1500.0
    # Equal starting ratings, K=32: winner gains exactly K * (1 - 0.5) = 16.
    assert abs(a.rating - 1516.0) < 1e-9
    assert abs(b.rating - 1484.0) < 1e-9


def test_update_is_zero_sum():
    elo = Elo()
    a = Rating(1600.0)
    b = Rating(1400.0)
    before = a.rating + b.rating
    elo.update(a, b, score_a=0.0)  # underdog upset
    after = a.rating + b.rating
    assert abs(before - after) < 1e-9


def test_upset_moves_more_points_than_expected_win():
    elo = Elo(k=32.0)
    favourite = Rating(1800.0)
    underdog = Rating(1200.0)
    # When the heavy favourite wins (expected), they gain little.
    fav = Rating(favourite.rating)
    und = Rating(underdog.rating)
    elo.update(fav, und, score_a=1.0)
    expected_gain = fav.rating - 1800.0
    # When the underdog wins (an upset), the swing is much larger.
    fav2 = Rating(favourite.rating)
    und2 = Rating(underdog.rating)
    elo.update(fav2, und2, score_a=0.0)
    upset_gain = und2.rating - 1200.0
    assert upset_gain > expected_gain
