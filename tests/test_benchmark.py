"""Tests de ``analysis/benchmark.py`` avec un faux moteur de match."""

import pytest

from analysis.benchmark import (
    METRIC_COLUMNS,
    experiments_to_samples,
    run_benchmark,
    summarize_table,
)
from game.constants import DEFAULT_SEED
from game.models import AgentType
from tests.fakes import fake_run_match

ATTACKERS = [AgentType.DETERMINISTIC, AgentType.MINIMAX]
GOALKEEPERS = [AgentType.FREQUENCY, AgentType.MINIMAX]
ROUNDS = 60
SEEDS = 3


def test_run_benchmark_plays_every_pair_on_every_seed() -> None:
    result = run_benchmark(
        ATTACKERS,
        GOALKEEPERS,
        num_rounds=ROUNDS,
        num_seeds=SEEDS,
        runner=fake_run_match,
    )
    assert len(result.experiments) == 2 * 2 * SEEDS
    pairs = {
        (e.config.attacker_agent, e.config.goalkeeper_agent) for e in result.experiments
    }
    assert pairs == {(a, g) for a in ATTACKERS for g in GOALKEEPERS}


def test_run_benchmark_uses_identical_conditions_for_all_pairs() -> None:
    result = run_benchmark(
        ATTACKERS,
        GOALKEEPERS,
        num_rounds=ROUNDS,
        num_seeds=SEEDS,
        runner=fake_run_match,
    )
    seeds_per_pair: dict[tuple[object, object], set[int]] = {}
    for e in result.experiments:
        key = (e.config.attacker_agent, e.config.goalkeeper_agent)
        seeds_per_pair.setdefault(key, set()).add(e.config.seed)
        assert e.config.num_rounds == ROUNDS
    expected = {DEFAULT_SEED + k for k in range(SEEDS)}
    assert all(seeds == expected for seeds in seeds_per_pair.values())


def test_run_benchmark_is_reproducible() -> None:
    first = run_benchmark(
        ATTACKERS,
        GOALKEEPERS,
        num_rounds=ROUNDS,
        num_seeds=SEEDS,
        runner=fake_run_match,
    )
    second = run_benchmark(
        ATTACKERS,
        GOALKEEPERS,
        num_rounds=ROUNDS,
        num_seeds=SEEDS,
        runner=fake_run_match,
    )
    assert [e.rounds for e in first.experiments] == [
        e.rounds for e in second.experiments
    ]


def test_run_benchmark_reports_progress() -> None:
    calls: list[tuple[int, int]] = []
    run_benchmark(
        ATTACKERS,
        GOALKEEPERS,
        num_rounds=ROUNDS,
        num_seeds=2,
        runner=fake_run_match,
        progress=lambda done, total: calls.append((done, total)),
    )
    assert calls[0] == (1, 8)
    assert calls[-1] == (8, 8)


def test_run_benchmark_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        run_benchmark([], GOALKEEPERS, runner=fake_run_match)
    with pytest.raises(ValueError):
        run_benchmark(ATTACKERS, GOALKEEPERS, num_seeds=0, runner=fake_run_match)


def test_samples_have_two_rows_per_experiment_and_zero_sum() -> None:
    result = run_benchmark(
        ATTACKERS,
        GOALKEEPERS,
        num_rounds=ROUNDS,
        num_seeds=SEEDS,
        runner=fake_run_match,
    )
    samples = experiments_to_samples(result.experiments)
    assert len(samples) == 2 * len(result.experiments)
    for metric in METRIC_COLUMNS:
        assert metric in samples.columns
    attacker = samples[samples["role"] == "attacker"].reset_index(drop=True)
    goalkeeper = samples[samples["role"] == "goalkeeper"].reset_index(drop=True)
    assert (attacker["mean_reward"] + goalkeeper["mean_reward"]).abs().max() < 1e-12
    assert ((samples["goal_rate"] + samples["save_rate"]) - 1).abs().max() < 1e-12


def test_summary_by_opponent_and_overall() -> None:
    result = run_benchmark(
        ATTACKERS,
        GOALKEEPERS,
        num_rounds=ROUNDS,
        num_seeds=SEEDS,
        runner=fake_run_match,
    )
    samples = experiments_to_samples(result.experiments)
    detailed = summarize_table(samples)
    assert len(detailed) == 8
    assert (detailed["n_seeds"] == SEEDS).all()
    overall = summarize_table(samples, by_opponent=False)
    assert len(overall) == 4
    assert (overall["opponent"] == "all").all()
    assert (overall["n_seeds"] == 2 * SEEDS).all()
    for metric in METRIC_COLUMNS:
        assert f"{metric}_mean" in detailed.columns
        assert f"{metric}_ci95" in detailed.columns


def test_summary_reflects_predictable_agents() -> None:
    """Le faux agent « Fréquences » joue un cycle : prédictibilité élevée."""
    result = run_benchmark(
        [AgentType.MINIMAX],
        [AgentType.FREQUENCY, AgentType.MINIMAX],
        num_rounds=300,
        num_seeds=2,
        runner=fake_run_match,
    )
    table = summarize_table(
        experiments_to_samples(result.experiments), by_opponent=False
    )
    goalkeepers = table[table["role"] == "goalkeeper"]
    predictability = dict(
        zip(
            goalkeepers["agent"], [float(v) for v in goalkeepers["predictability_mean"]]
        )
    )
    assert predictability["frequency"] > 0.9
    assert predictability["minimax"] < 0.5


def test_summary_rejects_empty_or_incomplete_samples() -> None:
    result = run_benchmark(
        ATTACKERS,
        GOALKEEPERS,
        num_rounds=ROUNDS,
        num_seeds=1,
        runner=fake_run_match,
    )
    samples = experiments_to_samples(result.experiments)
    with pytest.raises(ValueError):
        summarize_table(samples.iloc[0:0])
    with pytest.raises(ValueError):
        summarize_table(samples.drop(columns=["exploitability"]))
