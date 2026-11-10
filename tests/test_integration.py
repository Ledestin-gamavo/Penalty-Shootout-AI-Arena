"""Tests d'intégration du benchmark avec le vrai moteur et les vrais agents.

Ces tests sont ignorés tant que ``agents/`` et ``game.simulation`` ne sont
pas disponibles. Ils couvrent les critères d'acceptation du cahier des
charges (§10.2) et du contrat d'API (§11) qui concernent le benchmark.
"""

import pytest

pytest.importorskip("agents")
pytest.importorskip("game.simulation")

from analysis.benchmark import (  # noqa: E402
    BenchmarkResult,
    experiments_to_samples,
    run_benchmark,
    summarize_table,
)
from game.constants import (  # noqa: E402
    DEFAULT_SEED,
    PROBABILITY_TOLERANCE,
    REFERENCE_GAME_VALUE,
)
from game.models import AgentType, MatchConfig  # noqa: E402
from game.simulation import run_match  # noqa: E402

ALL_AGENTS = list(AgentType)
ROUNDS = 100


@pytest.fixture(scope="module")
def full_benchmark() -> BenchmarkResult:
    """Benchmark complet 4 x 4 sur deux graines (partagé par les tests)."""
    return run_benchmark(ALL_AGENTS, ALL_AGENTS, num_rounds=ROUNDS, num_seeds=2)


def test_every_match_plays_exactly_the_requested_rounds(
    full_benchmark: BenchmarkResult,
) -> None:
    assert len(full_benchmark.experiments) == 4 * 4 * 2
    for experiment in full_benchmark.experiments:
        assert len(experiment.rounds) == ROUNDS
        assert [r.round_number for r in experiment.rounds] == list(range(1, ROUNDS + 1))


def test_every_round_is_zero_sum(full_benchmark: BenchmarkResult) -> None:
    for experiment in full_benchmark.experiments:
        for result in experiment.rounds:
            assert result.attacker_reward + result.goalkeeper_reward == 0


def test_same_seed_gives_same_rounds() -> None:
    config = MatchConfig(ROUNDS, AgentType.FICTITIOUS_PLAY, AgentType.MINIMAX)
    assert run_match(config).rounds == run_match(config).rounds


def test_benchmark_conditions_are_identical_for_all_pairs(
    full_benchmark: BenchmarkResult,
) -> None:
    expected = {DEFAULT_SEED, DEFAULT_SEED + 1}
    seeds: dict[tuple[object, object], set[int]] = {}
    for e in full_benchmark.experiments:
        key = (e.config.attacker_agent, e.config.goalkeeper_agent)
        seeds.setdefault(key, set()).add(e.config.seed)
        assert e.config.num_rounds == ROUNDS
        assert e.config.payoff_matrix is None
    assert len(seeds) == 16
    assert all(value == expected for value in seeds.values())


def test_metrics_are_within_their_bounds(full_benchmark: BenchmarkResult) -> None:
    samples = experiments_to_samples(full_benchmark.experiments)
    for rate in ("goal_rate", "save_rate", "equilibrium_gap", "predictability"):
        assert samples[rate].between(0.0, 1.0).all()
    assert (samples["exploitability"] >= 0.0).all()
    assert (samples["mean_decision_time"] >= 0.0).all()


def test_empirical_strategies_are_valid_distributions(
    full_benchmark: BenchmarkResult,
) -> None:
    for experiment in full_benchmark.experiments:
        for stats in (experiment.attacker_stats, experiment.goalkeeper_stats):
            assert all(p >= 0.0 for p in stats.empirical_strategy)
            assert abs(sum(stats.empirical_strategy) - 1.0) < PROBABILITY_TOLERANCE


def test_summary_table_covers_every_agent_role_and_opponent(
    full_benchmark: BenchmarkResult,
) -> None:
    table = summarize_table(experiments_to_samples(full_benchmark.experiments))
    assert len(table) == 4 * 2 * 4
    assert (table["n_seeds"] == 2).all()


def test_deterministic_goalkeeper_is_exploited_by_fictitious_play() -> None:
    experiment = run_match(
        MatchConfig(1_000, AgentType.FICTITIOUS_PLAY, AgentType.DETERMINISTIC)
    )
    assert experiment.attacker_stats.mean_reward > 0.9
    assert experiment.goalkeeper_stats.exploitability > 0.5


def test_minimax_attacker_secures_the_game_value() -> None:
    experiment = run_match(
        MatchConfig(2_000, AgentType.MINIMAX, AgentType.DETERMINISTIC)
    )
    assert experiment.attacker_stats.mean_reward == pytest.approx(
        REFERENCE_GAME_VALUE, abs=0.15
    )
    assert experiment.attacker_stats.exploitability < 0.1
    assert experiment.attacker_stats.equilibrium_gap < 0.1
