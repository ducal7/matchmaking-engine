"""Evaluation harness: turn raw match logs into metrics and committed plots.

Four questions are answered:

1. **Rating convergence** -- does mean ``|rating - true_skill|`` fall as players
   accumulate matches?
2. **Match fairness** -- how tight is the rating-gap distribution at match time?
3. **Queue time** -- how long do players wait?
4. **Win-rate balance** -- for evenly matched games, does the favourite win
   close to 50% of the time?
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display required (e.g. CI).
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

Results = dict[str, tuple[pd.DataFrame, pd.DataFrame]]


def convergence_curve(player_df: pd.DataFrame) -> pd.Series:
    """Mean absolute rating error indexed by number of matches played."""

    return player_df.groupby("k_played")["rating_error"].mean()


def all_queue_times(match_df: pd.DataFrame) -> np.ndarray:
    return np.concatenate([match_df["queue_a"].to_numpy(), match_df["queue_b"].to_numpy()])


def favourite_won(match_df: pd.DataFrame) -> np.ndarray:
    """1.0 when the higher-rated player won, else 0.0 (ties in rating -> a)."""

    a_is_fav = match_df["rating_a"].to_numpy() >= match_df["rating_b"].to_numpy()
    a_won = match_df["score_a"].to_numpy() == 1.0
    return np.where(a_is_fav, a_won, ~a_won).astype(float)


def summarize(results: Results) -> dict[str, dict[str, float]]:
    """Compute headline statistics for each rating system."""

    summary: dict[str, dict[str, float]] = {}
    for name, (match_df, player_df) in results.items():
        curve = convergence_curve(player_df)
        # Average error over the last fifth of each player's career.
        max_k = int(curve.index.max())
        tail = curve[curve.index >= max(1, int(max_k * 0.8))]
        even = match_df[match_df["rating_gap"] < 25.0]
        summary[name] = {
            "final_mean_error": float(tail.mean()),
            "median_queue_time": float(np.median(all_queue_times(match_df))),
            "even_winrate": float(favourite_won(even).mean()) if len(even) else float("nan"),
            "n_matches": int(len(match_df)),
        }
    return summary


def _plot_convergence(results: Results, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (_match_df, player_df) in results.items():
        curve = convergence_curve(player_df)
        ax.plot(curve.index, curve.values, label=name, linewidth=2)
    ax.set_xlabel("Matches played by a player")
    ax.set_ylabel("Mean |rating - true skill|")
    ax.set_title("Rating convergence toward hidden true skill")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=120)
    plt.close(fig)


def _plot_queue_time(results: Results, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (match_df, _player_df) in results.items():
        q = all_queue_times(match_df)
        ax.hist(q, bins=40, histtype="step", linewidth=2, label=name, density=True)
    ax.set_xlabel("Queue wait time (sim seconds)")
    ax.set_ylabel("Density")
    ax.set_title("Queue-time distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "queue_time.png", dpi=120)
    plt.close(fig)


def _plot_skill_gap(results: Results, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (match_df, _player_df) in results.items():
        ax.hist(
            match_df["rating_gap"].to_numpy(),
            bins=40,
            histtype="step",
            linewidth=2,
            label=name,
            density=True,
        )
    ax.set_xlabel("Rating gap between matched players")
    ax.set_ylabel("Density")
    ax.set_title("Match fairness: skill-gap distribution at match time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "skill_gap.png", dpi=120)
    plt.close(fig)


def _plot_winrate_balance(results: Results, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.array([0, 25, 50, 100, 150, 200, 300, 500, 1200])
    centers = (bins[:-1] + bins[1:]) / 2
    for name, (match_df, _player_df) in results.items():
        gap = match_df["rating_gap"].to_numpy()
        fav = favourite_won(match_df)
        rates = []
        for lo, hi in zip(bins[:-1], bins[1:], strict=True):
            mask = (gap >= lo) & (gap < hi)
            rates.append(fav[mask].mean() if mask.any() else np.nan)
        ax.plot(centers, rates, marker="o", linewidth=2, label=name)
    ax.axhline(0.5, color="grey", linestyle="--", label="50% (perfectly even)")
    ax.set_xlabel("Rating gap between matched players")
    ax.set_ylabel("Higher-rated player's win rate")
    ax.set_title("Win-rate balance vs rating gap")
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "win_rate_balance.png", dpi=120)
    plt.close(fig)


def generate_all(results: Results, out_dir: Path) -> None:
    """Generate and save all four evaluation plots into ``out_dir``."""

    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_convergence(results, out_dir)
    _plot_queue_time(results, out_dir)
    _plot_skill_gap(results, out_dir)
    _plot_winrate_balance(results, out_dir)
