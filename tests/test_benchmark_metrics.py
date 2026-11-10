"""Tests de ``theory/benchmark_metrics.py``."""

import math

import numpy as np
import pytest

from game.actions import Action
from game.constants import EQUILIBRIUM_TOLERANCE
from game.models import Role, RoundResult
from game.payoff import build_reference_matrix
from theory.benchmark_metrics import (
    best_response_cells,
    cumulative_frequencies,
    cumulative_mean_reward,
    markov_predictability,
    summarize_samples,
)
from theory.equilibrium import best_response
from theory.metrics import average_reward, empirical_strategy, exploitability


def test_summarize_samples_known_values() -> None:
    summary = summarize_samples([1.0, 2.0, 3.0, 4.0])
    assert summary.n == 4
    assert summary.mean == pytest.approx(2.5)
    assert summary.std == pytest.approx(math.sqrt(5 / 3))
    # Student, 3 degrés de liberté : quantile 97,5 % = 3,182446.
    assert summary.ci95 == pytest.approx(3.182446 * summary.std / 2, rel=1e-4)


def test_summarize_samples_single_value_has_no_spread() -> None:
    summary = summarize_samples([0.7])
    assert (summary.n, summary.mean, summary.std, summary.ci95) == (1, 0.7, 0.0, 0.0)


def test_summarize_samples_rejects_empty() -> None:
    with pytest.raises(ValueError):
        summarize_samples([])


def test_predictability_constant_sequence_is_one() -> None:
    assert markov_predictability([Action.LEFT] * 50) == 1.0


def test_predictability_detects_cycle_that_exploitability_misses() -> None:
    """G, C, D répétés : exploitabilité nulle, prédictibilité quasi totale."""
    actions = [Action(i % 3) for i in range(300)]
    strategy = empirical_strategy(actions)
    matrix = build_reference_matrix()
    assert exploitability(matrix, Role.ATTACKER, strategy) < EQUILIBRIUM_TOLERANCE
    assert markov_predictability(actions) > 0.95


def test_predictability_of_uniform_sequence_is_near_one_third() -> None:
    rng = np.random.default_rng(0)
    actions = [Action(int(i)) for i in rng.integers(0, 3, 5_000)]
    assert markov_predictability(actions) == pytest.approx(1 / 3, abs=0.05)


def test_predictability_needs_two_actions() -> None:
    with pytest.raises(ValueError):
        markov_predictability([Action.LEFT])


def test_cumulative_frequencies_are_distributions() -> None:
    actions = [Action.LEFT, Action.CENTER, Action.CENTER, Action.RIGHT]
    frequencies = cumulative_frequencies(actions)
    assert len(frequencies) == len(actions)
    for row in frequencies:
        assert all(p >= 0 for p in row)
        assert sum(row) == pytest.approx(1.0)
    assert frequencies[-1] == pytest.approx(empirical_strategy(actions))
    assert frequencies[0] == (1.0, 0.0, 0.0)


def test_cumulative_frequencies_rejects_empty() -> None:
    with pytest.raises(ValueError):
        cumulative_frequencies([])


def test_cumulative_mean_reward_ends_at_average_reward() -> None:
    rounds = [
        RoundResult(1, Action.LEFT, Action.LEFT, -1.0),
        RoundResult(2, Action.LEFT, Action.CENTER, 1.0),
        RoundResult(3, Action.CENTER, Action.CENTER, -1.0),
    ]
    means = cumulative_mean_reward(rounds, Role.ATTACKER)
    assert means[0] == -1.0
    assert means[-1] == pytest.approx(average_reward(rounds, Role.ATTACKER))
    goalkeeper = cumulative_mean_reward(rounds, Role.GOALKEEPER)
    assert goalkeeper[-1] == pytest.approx(-means[-1])


def test_best_response_cells_on_reference_matrix() -> None:
    matrix = build_reference_matrix()
    attacker = best_response_cells(matrix, Role.ATTACKER)
    goalkeeper = best_response_cells(matrix, Role.GOALKEEPER)
    # Contre un plongeon pur, deux tirs marquent ; contre un tir pur, un seul
    # plongeon l'arrête (la diagonale).
    assert len(attacker) == 6
    assert all(a != g for a, g in attacker)
    assert goalkeeper == tuple((action, action) for action in Action)


def test_best_response_cells_contain_the_single_best_response() -> None:
    matrix = build_reference_matrix()
    for column in Action:
        pure = tuple(1.0 if a == column else 0.0 for a in Action)
        chosen = best_response(matrix, Role.ATTACKER, pure)
        assert (chosen, column) in best_response_cells(matrix, Role.ATTACKER)
