"""Discrete-event matchmaking simulation built on SimPy.

Players arrive into a shared queue over time. A matchmaker process periodically
pairs them using :func:`matchmaking.matcher.find_match`, resolves the game from
the players' hidden true skills, and updates their ratings. Every match is
logged so the evaluation harness can measure fairness, queue time, rating
convergence and win-rate balance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simpy

from .matcher import MatchConfig, QueueEntry, find_match
from .outcomes import play_match
from .players import Player, generate_players
from .ratings import Elo, Glicko2, RatingSystem

DEFAULT_SEED = 20260626


@dataclass
class SimConfig:
    """Configuration for a single simulation run."""

    n_players: int = 1000
    matches_per_player: int = 40
    mean_think_time: float = 120.0
    tick: float = 2.0
    end_time: float = 12000.0
    match_cfg: MatchConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.match_cfg is None:
            self.match_cfg = MatchConfig()


class Simulation:
    """Holds queue state and per-match logs for one run."""

    def __init__(
        self,
        env: simpy.Environment,
        rating_system: RatingSystem,
        cfg: SimConfig,
        rng: np.random.Generator,
    ) -> None:
        self.env = env
        self.rating_system = rating_system
        self.cfg = cfg
        self.rng = rng
        self.queue: list[QueueEntry] = []
        self.match_records: list[dict] = []
        self.player_records: list[dict] = []
        self.match_counter = 0

    def enqueue(self, player: Player) -> simpy.events.Event:
        """Add ``player`` to the queue and return the event that fires when
        their match has been resolved."""

        event = self.env.event()
        entry = QueueEntry(
            rating=player.rating.rating,
            enqueue_time=self.env.now,
            payload={"player": player, "event": event},
        )
        self.queue.append(entry)
        return event

    def matchmaker(self):
        """SimPy process: every tick, drain as many matches as possible."""

        while True:
            yield self.env.timeout(self.cfg.tick)
            self._match_all()

    def _match_all(self) -> None:
        while True:
            pair = find_match(self.queue, self.env.now, self.cfg.match_cfg)
            if pair is None:
                break
            i, j = pair
            entry_a = self.queue[i]
            entry_b = self.queue[j]
            for idx in sorted((i, j), reverse=True):
                self.queue.pop(idx)
            self._resolve(entry_a, entry_b)

    def _resolve(self, entry_a: QueueEntry, entry_b: QueueEntry) -> None:
        player_a: Player = entry_a.payload["player"]
        player_b: Player = entry_b.payload["player"]
        now = self.env.now
        queue_a = now - entry_a.enqueue_time
        queue_b = now - entry_b.enqueue_time
        rating_a = player_a.rating.rating
        rating_b = player_b.rating.rating

        score_a = play_match(player_a.true_skill, player_b.true_skill, self.rng)
        self.rating_system.update(player_a.rating, player_b.rating, score_a)
        player_a.matches_played += 1
        player_b.matches_played += 1
        self.match_counter += 1

        self.match_records.append(
            {
                "match_id": self.match_counter,
                "t": now,
                "rating_a": rating_a,
                "rating_b": rating_b,
                "rating_gap": abs(rating_a - rating_b),
                "true_skill_gap": abs(player_a.true_skill - player_b.true_skill),
                "queue_a": queue_a,
                "queue_b": queue_b,
                "score_a": score_a,
            }
        )
        for player in (player_a, player_b):
            self.player_records.append(
                {
                    "match_id": self.match_counter,
                    "pid": player.pid,
                    "k_played": player.matches_played,
                    "rating_error": abs(player.rating.rating - player.true_skill),
                }
            )

        entry_a.payload["event"].succeed()
        entry_b.payload["event"].succeed()


def _player_process(
    env: simpy.Environment,
    player: Player,
    sim: Simulation,
    rng: np.random.Generator,
    cfg: SimConfig,
):
    for _ in range(cfg.matches_per_player):
        yield env.timeout(float(rng.exponential(cfg.mean_think_time)))
        event = sim.enqueue(player)
        yield event


def run_simulation(
    rating_system: RatingSystem, cfg: SimConfig, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one simulation and return ``(match_df, player_df)``."""

    players = generate_players(cfg.n_players, rating_system, seed)
    env = simpy.Environment()
    sim = Simulation(env, rating_system, cfg, np.random.default_rng(seed + 1))
    env.process(sim.matchmaker())

    think_rng = np.random.default_rng(seed + 2)
    for player in players:
        env.process(_player_process(env, player, sim, think_rng, cfg))

    env.run(until=cfg.end_time)
    return pd.DataFrame(sim.match_records), pd.DataFrame(sim.player_records)


def run_all(
    cfg: SimConfig, seed: int = DEFAULT_SEED
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Run the simulation for every rating system and return their logs."""

    systems: list[RatingSystem] = [Elo(), Glicko2()]
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for system in systems:
        match_df, player_df = run_simulation(system, cfg, seed)
        results[system.name] = (match_df, player_df)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the matchmaking simulation.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--players", type=int, default=1000)
    parser.add_argument("--matches-per-player", type=int, default=40)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (just run the sim and print a summary).",
    )
    args = parser.parse_args(argv)

    cfg = SimConfig(
        n_players=args.players, matches_per_player=args.matches_per_player
    )
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)

    print(
        f"Running simulation: {cfg.n_players} players, "
        f"{cfg.matches_per_player} matches/player, seed={args.seed}"
    )
    results = run_all(cfg, seed=args.seed)

    for name, (match_df, player_df) in results.items():
        match_df.to_csv(data_dir / f"matches_{name}.csv", index=False)
        player_df.to_csv(data_dir / f"players_{name}.csv", index=False)

    # Import lazily so the simulation can run in environments without a display.
    from . import evaluate

    summary = evaluate.summarize(results)
    print("\n=== Simulation summary ===")
    for name, stats in summary.items():
        print(
            f"[{name:8s}] final mean rating error={stats['final_mean_error']:7.2f} | "
            f"median queue time={stats['median_queue_time']:6.2f} | "
            f"even-match favourite win rate={stats['even_winrate']:.3f} | "
            f"matches={stats['n_matches']}"
        )

    if not args.no_plots:
        out_dir = repo_root / "results"
        evaluate.generate_all(results, out_dir)
        print(f"\nPlots written to {out_dir}")


if __name__ == "__main__":
    main()
