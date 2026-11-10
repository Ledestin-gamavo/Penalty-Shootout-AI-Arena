"""Tests des figures de ``visualization/``."""

import math

import pandas as pd
import plotly.graph_objects as go
import pytest

from analysis.benchmark import experiments_to_samples, run_benchmark, summarize_table
from game.models import AgentType, MatchConfig, Role
from game.payoff import build_reference_matrix
from tests.fakes import fake_run_match
from theory.benchmark_metrics import best_response_cells
from visualization.benchmark_plots import (
    plot_decision_time,
    plot_matchup_heatmap,
    plot_metric_comparison,
    plot_performance_overview,
    plot_strategy_quality,
)
from visualization.strategy_plots import (
    plot_frequency_evolution,
    plot_payoff_heatmap,
    plot_reward_evolution,
    plot_strategy_bars,
)
from visualization.theme import MAX_PLOT_POINTS

UNIFORM = (1 / 3, 1 / 3, 1 / 3)


def _summary() -> pd.DataFrame:
    result = run_benchmark(
        [AgentType.DETERMINISTIC, AgentType.MINIMAX],
        [AgentType.FREQUENCY, AgentType.MINIMAX],
        num_rounds=50,
        num_seeds=3,
        runner=fake_run_match,
    )
    return summarize_table(experiments_to_samples(result.experiments))


def test_strategy_bars_with_reference() -> None:
    figure = plot_strategy_bars((0.5, 0.3, 0.2), UNIFORM)
    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 2
    assert sum(figure.data[0].y) == pytest.approx(1.0)
    assert list(figure.data[0].x) == ["Gauche", "Centre", "Droite"]


def test_strategy_bars_without_reference_and_invalid_input() -> None:
    assert len(plot_strategy_bars(UNIFORM).data) == 1
    with pytest.raises(ValueError):
        plot_strategy_bars((0.5, 0.5))


def test_payoff_heatmap_matches_matrix_and_highlights() -> None:
    matrix = build_reference_matrix()
    cells = best_response_cells(matrix, Role.ATTACKER)
    figure = plot_payoff_heatmap(matrix, cells)
    assert [list(row) for row in figure.data[0].z] == [list(r) for r in matrix]
    assert len(figure.layout.shapes) == len(cells)
    with pytest.raises(ValueError):
        plot_payoff_heatmap(((1.0, 2.0), (3.0, 4.0)))


def test_frequency_evolution_ends_on_a_distribution() -> None:
    rounds = fake_run_match(
        MatchConfig(90, AgentType.MINIMAX, AgentType.MINIMAX)
    ).rounds
    figure = plot_frequency_evolution(rounds, Role.ATTACKER, UNIFORM)
    assert len(figure.data) == 3
    assert sum(trace.y[-1] for trace in figure.data) == pytest.approx(1.0)
    assert figure.data[0].x[-1] == 90


def test_frequency_evolution_downsamples_long_matches() -> None:
    rounds = fake_run_match(
        MatchConfig(10_000, AgentType.MINIMAX, AgentType.MINIMAX)
    ).rounds
    figure = plot_frequency_evolution(rounds, Role.GOALKEEPER)
    assert len(figure.data[0].x) <= MAX_PLOT_POINTS + 1
    assert figure.data[0].x[-1] == 10_000


def test_reward_evolution() -> None:
    rounds = fake_run_match(
        MatchConfig(40, AgentType.MINIMAX, AgentType.MINIMAX)
    ).rounds
    figure = plot_reward_evolution(rounds, Role.ATTACKER, reference_value=1 / 3)
    assert len(figure.data[0].y) == 40
    assert len(figure.layout.shapes) == 1


def test_benchmark_figures_have_one_group_per_agent_and_opponent() -> None:
    summary = _summary()
    figure = plot_metric_comparison(summary, "mean_reward", Role.ATTACKER)
    assert len(figure.data) == 2  # deux gardiens adverses
    assert list(figure.data[0].x) == ["Déterministe", "Minimax"]
    assert len(plot_performance_overview(summary, Role.GOALKEEPER).data) == 6
    assert len(plot_strategy_quality(summary, Role.ATTACKER).data) == 6


def test_decision_time_is_shown_in_milliseconds() -> None:
    summary = _summary()
    figure = plot_decision_time(summary, Role.ATTACKER)
    expected = summary[summary["role"] == "attacker"]["mean_decision_time_mean"].iloc[0]
    assert figure.data[0].y[0] == pytest.approx(expected * 1000)


def test_matchup_heatmap_matches_the_table() -> None:
    summary = _summary()
    figure = plot_matchup_heatmap(summary)
    z = figure.data[0].z
    assert (len(z), len(z[0])) == (2, 2)
    row = summary[
        (summary["role"] == "attacker")
        & (summary["agent"] == "deterministic")
        & (summary["opponent"] == "frequency")
    ].iloc[0]
    assert z[0][0] == pytest.approx(row["mean_reward_mean"])
    assert all(math.isfinite(v) for line in z for v in line)


def test_benchmark_figures_reject_bad_input() -> None:
    summary = _summary()
    with pytest.raises(ValueError):
        plot_metric_comparison(
            summary.drop(columns=["goal_rate_ci95"]), "goal_rate", Role.ATTACKER
        )
    attackers_only = summary[summary["role"] == "attacker"]
    with pytest.raises(ValueError):
        plot_metric_comparison(attackers_only, "goal_rate", Role.GOALKEEPER)
