"""Tests for the Glicko-2 rating system.

The headline test reproduces the worked example from Glickman's Glicko-2 paper
("Example calculation"), which is the canonical correctness check.
"""

from matchmaking.ratings import Glicko2, Rating


def test_glicko2_reference_example():
    # Glickman's example: a 1500/200 player with volatility 0.06 and tau=0.5
    # plays three opponents, winning the first and losing the next two.
    glicko = Glicko2(tau=0.5)
    player = Rating(rating=1500.0, rd=200.0, vol=0.06)
    results = [
        (1400.0, 30.0, 1.0),
        (1550.0, 100.0, 0.0),
        (1700.0, 300.0, 0.0),
    ]
    glicko.rate_period(player, results)

    # Reference outputs from the paper.
    assert abs(player.rating - 1464.06) < 0.1
    assert abs(player.rd - 151.52) < 0.1
    assert abs(player.vol - 0.05999) < 1e-4


def test_glicko2_winner_gains_loser_loses():
    glicko = Glicko2()
    a = Rating(1500.0, 200.0, 0.06)
    b = Rating(1500.0, 200.0, 0.06)
    glicko.update(a, b, score_a=1.0)
    assert a.rating > 1500.0
    assert b.rating < 1500.0


def test_glicko2_rd_grows_when_idle():
    glicko = Glicko2()
    player = Rating(1500.0, 100.0, 0.06)
    glicko.rate_period(player, [])  # did not compete this period
    assert player.rd > 100.0


def test_glicko2_rd_shrinks_after_games():
    glicko = Glicko2()
    a = Rating(1500.0, 350.0, 0.06)
    b = Rating(1500.0, 80.0, 0.06)
    rd_before = a.rd
    glicko.update(a, b, score_a=1.0)
    # Playing a game provides information, so the rating deviation drops.
    assert a.rd < rd_before


def test_expected_score_symmetric_at_equal_ratings():
    glicko = Glicko2()
    a = Rating(1500.0, 50.0, 0.06)
    b = Rating(1500.0, 50.0, 0.06)
    assert abs(glicko.expected_score(a, b) - 0.5) < 1e-9
