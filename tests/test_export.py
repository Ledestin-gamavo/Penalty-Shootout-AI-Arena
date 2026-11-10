"""Tests de ``analysis/export.py``."""

from pathlib import Path

import pandas as pd
import pytest

from analysis.benchmark import experiments_to_samples
from analysis.export import default_filename, export_csv, rounds_to_dataframe
from game.actions import Action
from game.models import AgentType, MatchConfig
from tests.fakes import fake_run_match


def test_rounds_to_dataframe_is_consistent() -> None:
    config = MatchConfig(20, AgentType.MINIMAX, AgentType.MINIMAX)
    rounds = fake_run_match(config).rounds
    table = rounds_to_dataframe(rounds)
    assert len(table) == 20
    assert list(table["Manche"]) == list(range(1, 21))
    assert set(table["Résultat"]) <= {"But", "Arrêt"}
    assert (table["Gain attaquant"] + table["Gain gardien"] == 0).all()
    assert set(table["Attaquant"]) <= {a.label for a in Action}


def test_export_csv_round_trip_keeps_parameters_and_metrics(tmp_path: Path) -> None:
    config = MatchConfig(30, AgentType.DETERMINISTIC, AgentType.MINIMAX, seed=7)
    samples = experiments_to_samples([fake_run_match(config)])
    path = export_csv(samples, "essai.csv", tmp_path)
    loaded = pd.read_csv(path, encoding="utf-8-sig")
    assert {"seed", "num_rounds", "goal_rate", "exploitability"} <= set(loaded.columns)
    assert list(loaded["seed"]) == [7, 7]
    assert list(loaded["num_rounds"]) == [30, 30]


def test_export_csv_keeps_accents(tmp_path: Path) -> None:
    config = MatchConfig(10, AgentType.MINIMAX, AgentType.MINIMAX)
    table = rounds_to_dataframe(fake_run_match(config).rounds)
    loaded = pd.read_csv(export_csv(table, "h.csv", tmp_path), encoding="utf-8-sig")
    assert "Résultat" in loaded.columns


def test_export_csv_creates_missing_directory(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b"
    path = export_csv(pd.DataFrame({"x": [1]}), "t.csv", target)
    assert path.exists()


@pytest.mark.parametrize("name", ["../evil.csv", "dossier/x.csv", "x.txt", ""])
def test_export_csv_rejects_unsafe_names(name: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        export_csv(pd.DataFrame({"x": [1]}), name, tmp_path)


def test_default_filename_format() -> None:
    name = default_filename("benchmark")
    assert name.startswith("benchmark_") and name.endswith("Z.csv")


def test_to_csv_bytes_round_trip() -> None:
    from io import BytesIO

    from analysis.export import to_csv_bytes

    frame = pd.DataFrame({"Résultat": ["But", "Arrêt"], "x": [1, 2]})
    loaded = pd.read_csv(BytesIO(to_csv_bytes(frame)), encoding="utf-8-sig")
    assert loaded.equals(frame)


def test_agent_stats_dataframe_matches_the_experiment() -> None:
    from analysis.export import METRIC_LABELS, agent_stats_dataframe

    experiment = fake_run_match(
        MatchConfig(60, AgentType.DETERMINISTIC, AgentType.MINIMAX, seed=3)
    )
    table = agent_stats_dataframe(experiment)
    assert list(table["Rôle"]) == ["Attaquant", "Gardien"]
    assert list(table["Agent"]) == ["Déterministe", "Minimax"]
    assert (table["Buts"] + table["Arrêts"] == 60).all()
    assert table.loc[0, METRIC_LABELS["mean_reward"]] == (
        experiment.attacker_stats.mean_reward
    )
    assert table.loc[0, METRIC_LABELS["predictability"]] == 1.0


def test_experiments_overview_has_one_row_per_experiment() -> None:
    from analysis.export import experiments_overview

    experiments = [
        fake_run_match(MatchConfig(30, AgentType.MINIMAX, AgentType.MINIMAX, seed=s))
        for s in (1, 2, 3)
    ]
    table = experiments_overview(experiments)
    assert list(table["Expérience"]) == [1, 2, 3]
    assert list(table["Graine"]) == [1, 2, 3]
    assert (table["Manches"] == 30).all()


def test_summary_for_display_formats_mean_and_interval() -> None:
    from analysis.benchmark import (
        experiments_to_samples,
        run_benchmark,
        summarize_table,
    )
    from analysis.export import METRIC_LABELS, summary_for_display

    result = run_benchmark(
        [AgentType.MINIMAX],
        [AgentType.FREQUENCY],
        num_rounds=40,
        num_seeds=3,
        runner=fake_run_match,
    )
    samples = experiments_to_samples(result.experiments)
    shown = summary_for_display(summarize_table(samples))
    assert set(shown["Rôle"]) == {"Attaquant", "Gardien"}
    assert set(shown["Adversaire"]) == {"Fréquences", "Minimax"}
    assert all("±" in value for value in shown[METRIC_LABELS["goal_rate"]])
    overall = summary_for_display(summarize_table(samples, by_opponent=False))
    assert set(overall["Adversaire"]) == {"Tous"}
