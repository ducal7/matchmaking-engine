"""Tests that the full simulation drives ratings toward true skill."""

import pytest

from matchmaking.evaluate import convergence_curve
from matchmaking.ratings import Elo, Glicko2
from matchmaking.simulate import SimConfig, run_simulation


@pytest.fixture(scope="module")
def small_config():
    # Small but realistic enough to show convergence; fast (<~1s per system).
    return SimConfig(
        n_players=150,
        matches_per_player=40,
        mean_think_time=30.0,
        end_time=1400.0,
    )


@pytest.mark.parametrize("system_factory", [Elo, Glicko2])
def test_average_rating_error_decreases(system_factory, small_config):
    system = system_factory()
    _match_df, player_df = run_simulation(system, small_config, seed=7)
    curve = convergence_curve(player_df)

    early = curve[curve.index <= 2].mean()
    late = curve[curve.index >= curve.index.max() - 5].mean()

    # Ratings should be meaningfully closer to true skill late in the sim.
    assert late < early
    assert late < 0.85 * early


def test_simulation_produces_matches(small_config):
    match_df, _player_df = run_simulation(Elo(), small_config, seed=7)
    assert len(match_df) > 100
    # Two player rows are logged per match.
    assert set(match_df.columns) >= {"rating_gap", "queue_a", "queue_b", "score_a"}
