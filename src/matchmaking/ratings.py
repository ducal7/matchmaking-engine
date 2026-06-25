"""Pluggable rating systems behind a common interface.

Two implementations are provided:

* :class:`Elo` -- the classic logistic Elo update.
* :class:`Glicko2` -- Glickman's Glicko-2 system, which tracks not only a
  rating but also a rating deviation (uncertainty) and a volatility.

Both expose the same :class:`RatingSystem` interface so the simulator and the
matcher can treat them interchangeably.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Glicko-2 maps the public ("Glicko") rating scale onto an internal scale by
# dividing by this constant (173.7178 == 400 / ln(10)).
_GLICKO2_SCALE = 173.7178
_GLICKO2_CENTER = 1500.0


@dataclass
class Rating:
    """A player's rating state.

    ``rating`` is used by every system. ``rd`` (rating deviation) and ``vol``
    (volatility) are only meaningful for Glicko-2 but are carried on every
    rating so the two systems share one data type.
    """

    rating: float
    rd: float = 350.0
    vol: float = 0.06


class RatingSystem(ABC):
    """Common interface for all rating systems."""

    name: str

    @abstractmethod
    def new_rating(self) -> Rating:
        """Return a fresh rating for a previously unseen player."""

    @abstractmethod
    def expected_score(self, a: Rating, b: Rating) -> float:
        """Probability that player ``a`` beats player ``b`` (in ``[0, 1]``)."""

    @abstractmethod
    def update(self, a: Rating, b: Rating, score_a: float) -> None:
        """Update ``a`` and ``b`` in place after a game.

        ``score_a`` is ``1.0`` if ``a`` won, ``0.0`` if ``a`` lost and ``0.5``
        for a draw.
        """


class Elo(RatingSystem):
    """Classic Elo rating system."""

    name = "elo"

    def __init__(self, k: float = 32.0, initial: float = 1500.0) -> None:
        self.k = k
        self.initial = initial

    def new_rating(self) -> Rating:
        return Rating(self.initial)

    def expected_score(self, a: Rating, b: Rating) -> float:
        return 1.0 / (1.0 + 10.0 ** ((b.rating - a.rating) / 400.0))

    def update(self, a: Rating, b: Rating, score_a: float) -> None:
        ea = self.expected_score(a, b)
        # Expected score is symmetric: eb == 1 - ea, so the rating points one
        # player gains equal the points the other loses (zero sum).
        a.rating += self.k * (score_a - ea)
        b.rating += self.k * ((1.0 - score_a) - (1.0 - ea))


class Glicko2(RatingSystem):
    """Glicko-2 rating system (Glickman, 2013).

    The implementation follows the reference algorithm. A single head-to-head
    game is modelled as a one-opponent rating period for each player, using a
    snapshot of the opponent's pre-game state so both updates are symmetric.
    """

    name = "glicko2"

    def __init__(
        self,
        tau: float = 0.5,
        initial: float = 1500.0,
        initial_rd: float = 350.0,
        initial_vol: float = 0.06,
        epsilon: float = 1e-6,
    ) -> None:
        self.tau = tau
        self.initial = initial
        self.initial_rd = initial_rd
        self.initial_vol = initial_vol
        self.epsilon = epsilon

    def new_rating(self) -> Rating:
        return Rating(self.initial, self.initial_rd, self.initial_vol)

    @staticmethod
    def _g(phi: float) -> float:
        return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))

    def _e(self, mu: float, mu_j: float, phi_j: float) -> float:
        return 1.0 / (1.0 + math.exp(-self._g(phi_j) * (mu - mu_j)))

    def expected_score(self, a: Rating, b: Rating) -> float:
        mu = (a.rating - _GLICKO2_CENTER) / _GLICKO2_SCALE
        mu_j = (b.rating - _GLICKO2_CENTER) / _GLICKO2_SCALE
        phi_j = b.rd / _GLICKO2_SCALE
        return self._e(mu, mu_j, phi_j)

    def rate_period(
        self, player: Rating, results: list[tuple[float, float, float]]
    ) -> None:
        """Update ``player`` in place from a rating period of ``results``.

        ``results`` is a list of ``(opponent_rating, opponent_rd, score)``
        tuples. An empty list means the player did not compete; only their
        rating deviation grows in that case.
        """

        mu = (player.rating - _GLICKO2_CENTER) / _GLICKO2_SCALE
        phi = player.rd / _GLICKO2_SCALE
        sigma = player.vol

        if not results:
            phi_star = math.sqrt(phi * phi + sigma * sigma)
            player.rd = _GLICKO2_SCALE * phi_star
            return

        v_inv = 0.0
        delta_sum = 0.0
        for opp_rating, opp_rd, score in results:
            mu_j = (opp_rating - _GLICKO2_CENTER) / _GLICKO2_SCALE
            phi_j = opp_rd / _GLICKO2_SCALE
            g = self._g(phi_j)
            e = self._e(mu, mu_j, phi_j)
            v_inv += g * g * e * (1.0 - e)
            delta_sum += g * (score - e)

        v = 1.0 / v_inv
        delta = v * delta_sum
        sigma_p = self._new_volatility(phi, v, delta, sigma)
        phi_star = math.sqrt(phi * phi + sigma_p * sigma_p)
        phi_p = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
        mu_p = mu + phi_p * phi_p * delta_sum

        player.rating = _GLICKO2_SCALE * mu_p + _GLICKO2_CENTER
        player.rd = _GLICKO2_SCALE * phi_p
        player.vol = sigma_p

    def _new_volatility(
        self, phi: float, v: float, delta: float, sigma: float
    ) -> float:
        """Solve for the new volatility using the Illinois algorithm."""

        a = math.log(sigma * sigma)
        tau = self.tau
        delta_sq = delta * delta
        phi_sq = phi * phi

        def f(x: float) -> float:
            ex = math.exp(x)
            num = ex * (delta_sq - phi_sq - v - ex)
            den = 2.0 * (phi_sq + v + ex) ** 2
            return num / den - (x - a) / (tau * tau)

        big_a = a
        if delta_sq > phi_sq + v:
            big_b = math.log(delta_sq - phi_sq - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
            big_b = a - k * tau

        f_a = f(big_a)
        f_b = f(big_b)
        while abs(big_b - big_a) > self.epsilon:
            c = big_a + (big_a - big_b) * f_a / (f_b - f_a)
            f_c = f(c)
            if f_c * f_b <= 0:
                big_a = big_b
                f_a = f_b
            else:
                f_a = f_a / 2.0
            big_b = c
            f_b = f_c

        return math.exp(big_a / 2.0)

    def update(self, a: Rating, b: Rating, score_a: float) -> None:
        # Snapshot pre-game opponent state so each player is rated against the
        # other's rating *before* this game, keeping the update symmetric.
        a_snap = (a.rating, a.rd, a.vol)
        b_snap = (b.rating, b.rd, b.vol)
        self.rate_period(a, [(b_snap[0], b_snap[1], score_a)])
        self.rate_period(b, [(a_snap[0], a_snap[1], 1.0 - score_a)])
