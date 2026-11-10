"""Tableaux d'historique et export CSV des expériences.

Ce module ne contient aucune logique métier : il met en forme des résultats
déjà calculés et les écrit dans ``data/exports/``.

Dépendances autorisées : ``game.*`` et pandas.
"""

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from analysis.benchmark import METRIC_COLUMNS
from game.constants import EXPORTS_DIR
from game.models import AgentType, Experiment, Role, RoundResult
from theory.benchmark_metrics import markov_predictability

# Conversion secondes -> millisecondes pour l'affichage des temps.
_SECONDS_TO_MS = 1_000.0

# Libellés français des métriques, dans l'ordre d'affichage.
METRIC_LABELS: dict[str, str] = {
    "goal_rate": "Taux de buts",
    "save_rate": "Taux d'arrêts",
    "mean_reward": "Gain moyen",
    "mean_decision_time": "Temps de décision (ms)",
    "equilibrium_gap": "Écart à l'équilibre",
    "exploitability": "Exploitabilité",
    "predictability": "Prévisibilité (ordre 1)",
}

# Encodage des CSV : avec BOM pour que Excel affiche correctement les accents.
_CSV_ENCODING = "utf-8-sig"


def rounds_to_dataframe(rounds: Sequence[RoundResult]) -> pd.DataFrame:
    """Construit l'historique compact d'une liste de manches (F08).

    Args:
        rounds: Manches jouées, dans l'ordre.

    Returns:
        Un tableau avec le numéro de manche, l'action de chaque joueur, le
        résultat (« But » ou « Arrêt ») et les gains des deux joueurs.
    """
    return pd.DataFrame(
        {
            "Manche": [r.round_number for r in rounds],
            "Attaquant": [r.attacker_action.label for r in rounds],
            "Gardien": [r.goalkeeper_action.label for r in rounds],
            "Résultat": ["But" if r.is_goal else "Arrêt" for r in rounds],
            "Gain attaquant": [r.attacker_reward for r in rounds],
            "Gain gardien": [r.goalkeeper_reward for r in rounds],
        }
    )


def default_filename(prefix: str) -> str:
    """Construit un nom de fichier CSV horodaté (UTC).

    Args:
        prefix: Début du nom, par exemple ``"benchmark"``.

    Returns:
        Un nom de la forme ``prefix_20261109T201500Z.csv``.
    """
    return f"{prefix}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.csv"


def export_csv(
    frame: pd.DataFrame,
    filename: str,
    directory: Path = EXPORTS_DIR,
) -> Path:
    """Écrit un tableau dans un fichier CSV du dossier d'export.

    Args:
        frame: Tableau à écrire.
        filename: Nom du fichier, sans chemin, terminé par ``.csv``.
        directory: Dossier d'export (créé si nécessaire).

    Returns:
        Le chemin du fichier écrit.

    Raises:
        ValueError: Si le nom contient un chemin ou n'est pas un ``.csv``.
    """
    if Path(filename).name != filename or not filename.endswith(".csv"):
        raise ValueError(f"Nom de fichier CSV invalide : {filename!r}.")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    frame.to_csv(path, index=False, encoding=_CSV_ENCODING)
    return path


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Sérialise un tableau en CSV (pour un bouton de téléchargement).

    Args:
        frame: Tableau à sérialiser.

    Returns:
        Le contenu CSV, encodé comme les fichiers de ``export_csv``.
    """
    return frame.to_csv(index=False).encode(_CSV_ENCODING)


def agent_stats_dataframe(experiment: Experiment) -> pd.DataFrame:
    """Construit le tableau des statistiques des deux agents d'une expérience.

    Args:
        experiment: Expérience à résumer.

    Returns:
        Une ligne par agent (attaquant puis gardien), avec des colonnes en
        français. Pour un gardien, « Buts » désigne les buts concédés.
    """
    rows: list[dict[str, object]] = []
    for role, stats in (
        (Role.ATTACKER, experiment.attacker_stats),
        (Role.GOALKEEPER, experiment.goalkeeper_stats),
    ):
        actions = [result.action_of(role) for result in experiment.rounds]
        predictability = (
            markov_predictability(actions) if len(actions) >= 2 else math.nan
        )
        rows.append(
            {
                "Rôle": role.label,
                "Agent": stats.agent_type.label,
                "Manches": stats.rounds,
                "Buts": stats.goals,
                "Arrêts": stats.saves,
                METRIC_LABELS["goal_rate"]: stats.goal_rate,
                METRIC_LABELS["save_rate"]: stats.save_rate,
                METRIC_LABELS["mean_reward"]: stats.mean_reward,
                METRIC_LABELS["mean_decision_time"]: (
                    stats.mean_decision_time * _SECONDS_TO_MS
                ),
                METRIC_LABELS["equilibrium_gap"]: stats.equilibrium_gap,
                METRIC_LABELS["exploitability"]: stats.exploitability,
                METRIC_LABELS["predictability"]: predictability,
            }
        )
    return pd.DataFrame(rows)


def experiments_overview(experiments: Sequence[Experiment]) -> pd.DataFrame:
    """Compare plusieurs expériences sur une ligne chacune.

    Args:
        experiments: Expériences à comparer, dans l'ordre de création.

    Returns:
        Un tableau avec la configuration (agents, manches, graine) et les
        principaux indicateurs des deux joueurs.
    """
    rows = []
    for index, experiment in enumerate(experiments, start=1):
        attacker = experiment.attacker_stats
        goalkeeper = experiment.goalkeeper_stats
        rows.append(
            {
                "Expérience": index,
                "Attaquant": attacker.agent_type.label,
                "Gardien": goalkeeper.agent_type.label,
                "Manches": experiment.config.num_rounds,
                "Graine": experiment.config.seed,
                "Gain moyen attaquant": attacker.mean_reward,
                "Taux de buts": attacker.goal_rate,
                "Écart attaquant": attacker.equilibrium_gap,
                "Écart gardien": goalkeeper.equilibrium_gap,
                "Exploitabilité attaquant": attacker.exploitability,
                "Exploitabilité gardien": goalkeeper.exploitability,
            }
        )
    return pd.DataFrame(rows)


def summary_for_display(summary: pd.DataFrame) -> pd.DataFrame:
    """Met en forme un résumé de benchmark pour l'affichage.

    Chaque métrique devient une colonne de texte « moyenne ± IC95 » (temps
    de décision en millisecondes).

    Args:
        summary: Tableau de ``analysis.benchmark.summarize_table``.

    Returns:
        Un tableau lisible, avec des libellés français.
    """
    rows: list[dict[str, object]] = []
    for _, line in summary.iterrows():
        opponent = str(line["opponent"])
        row: dict[str, object] = {
            "Agent": AgentType(str(line["agent"])).label,
            "Rôle": Role(str(line["role"])).label,
            "Adversaire": ("Tous" if opponent == "all" else AgentType(opponent).label),
            "Graines": int(line["n_seeds"]),
        }
        for metric in METRIC_COLUMNS:
            scale = _SECONDS_TO_MS if metric == "mean_decision_time" else 1.0
            mean = float(line[f"{metric}_mean"]) * scale
            ci95 = float(line[f"{metric}_ci95"]) * scale
            row[METRIC_LABELS[metric]] = f"{mean:.3f} ± {ci95:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)
