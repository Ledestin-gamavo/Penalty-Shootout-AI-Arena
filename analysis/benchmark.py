"""Exécution du benchmark et agrégation des résultats.

Le benchmark joue le produit cartésien attaquants x gardiens, avec les mêmes
graines ``base_seed + k`` pour toutes les paires, le même nombre de manches
et la même matrice : les agents sont donc comparés dans des conditions
identiques. Ce module n'appelle que ``game.simulation.run_match`` et les
métriques existantes ; il ne recalcule aucune règle de jeu.

Dépendances autorisées : ``game.*``, ``theory.*``, pandas et NumPy.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

import pandas as pd

from game.constants import (
    DEFAULT_BENCHMARK_NUM_SEEDS,
    DEFAULT_NUM_ROUNDS_SIMULATION,
    DEFAULT_SEED,
)
from game.models import (
    AgentStats,
    AgentType,
    Experiment,
    MatchConfig,
    PayoffMatrix,
    Role,
)
from theory.benchmark_metrics import markov_predictability, summarize_samples

# Fonction qui exécute un match IA contre IA (``game.simulation.run_match``).
MatchRunner: TypeAlias = Callable[[MatchConfig], Experiment]

# Callback de progression : ``(expériences terminées, expériences totales)``.
ProgressCallback: TypeAlias = Callable[[int, int], None]

# Colonnes de métriques d'un échantillon (une valeur par expérience et rôle).
METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "goal_rate",
    "save_rate",
    "mean_reward",
    "mean_decision_time",
    "equilibrium_gap",
    "exploitability",
    "predictability",
)

# Colonnes d'identification et de paramètres d'un échantillon.
KEY_COLUMNS: Final[tuple[str, ...]] = ("agent", "role", "opponent")
PARAMETER_COLUMNS: Final[tuple[str, ...]] = ("seed", "num_rounds")

# Valeur de la colonne ``opponent`` lorsque tous les adversaires sont agrégés.
ALL_OPPONENTS: Final[str] = "all"


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Résultat brut d'un benchmark.

    Attributes:
        num_rounds: Nombre de manches de chaque expérience.
        num_seeds: Nombre de graines par paire d'agents.
        base_seed: Première graine ; les suivantes sont ``base_seed + k``.
        payoff_matrix: Matrice personnalisée, ou ``None`` pour la référence.
        experiments: Toutes les expériences, dans l'ordre attaquant,
            gardien, graine.

    Raises:
        ValueError: Si ``experiments`` est vide.
    """

    num_rounds: int
    num_seeds: int
    base_seed: int
    payoff_matrix: PayoffMatrix | None
    experiments: tuple[Experiment, ...]

    def __post_init__(self) -> None:
        """Vérifie que le benchmark contient au moins une expérience.

        Raises:
            ValueError: Si aucune expérience n'est fournie.
        """
        if not self.experiments:
            raise ValueError("Un benchmark contient au moins une expérience.")


def run_benchmark(
    attackers: Sequence[AgentType],
    goalkeepers: Sequence[AgentType],
    *,
    num_rounds: int = DEFAULT_NUM_ROUNDS_SIMULATION,
    num_seeds: int = DEFAULT_BENCHMARK_NUM_SEEDS,
    base_seed: int = DEFAULT_SEED,
    payoff_matrix: PayoffMatrix | None = None,
    runner: MatchRunner | None = None,
    progress: ProgressCallback | None = None,
) -> BenchmarkResult:
    """Joue toutes les paires attaquant/gardien sur plusieurs graines.

    Args:
        attackers: Types d'agents jouant l'attaquant.
        goalkeepers: Types d'agents jouant le gardien.
        num_rounds: Nombre de manches par expérience.
        num_seeds: Nombre de graines par paire (au moins 1).
        base_seed: Première graine ; la graine ``k`` est ``base_seed + k``.
        payoff_matrix: Matrice personnalisée, ou ``None`` pour la référence.
        runner: Fonction d'exécution d'un match. Par défaut,
            ``game.simulation.run_match`` ; injectable pour les tests.
        progress: Callback appelé après chaque expérience.

    Returns:
        Le résultat brut du benchmark.

    Raises:
        ValueError: Si une liste d'agents est vide, si ``num_seeds`` est
            inférieur à 1 ou si une configuration est invalide.
    """
    if not attackers or not goalkeepers:
        raise ValueError("Il faut au moins un attaquant et un gardien.")
    if num_seeds < 1:
        raise ValueError(f"num_seeds doit être au moins 1, reçu : {num_seeds}.")
    if runner is None:
        # Import tardif : game.simulation dépend de agents/, livré par M2.
        from game.simulation import run_match

        runner = run_match
    total = len(attackers) * len(goalkeepers) * num_seeds
    experiments: list[Experiment] = []
    for attacker in attackers:
        for goalkeeper in goalkeepers:
            for offset in range(num_seeds):
                config = MatchConfig(
                    num_rounds=num_rounds,
                    attacker_agent=attacker,
                    goalkeeper_agent=goalkeeper,
                    seed=base_seed + offset,
                    payoff_matrix=payoff_matrix,
                )
                experiments.append(runner(config))
                if progress is not None:
                    progress(len(experiments), total)
    return BenchmarkResult(
        num_rounds=num_rounds,
        num_seeds=num_seeds,
        base_seed=base_seed,
        payoff_matrix=payoff_matrix,
        experiments=tuple(experiments),
    )


def _sample_row(
    experiment: Experiment, role: Role, stats: AgentStats, opponent: AgentType
) -> dict[str, object]:
    """Construit la ligne d'échantillon d'un agent dans une expérience.

    Args:
        experiment: Expérience analysée.
        role: Rôle de l'agent.
        stats: Statistiques agrégées de l'agent.
        opponent: Type de l'agent adverse.

    Returns:
        La ligne : identification, paramètres et métriques.
    """
    actions = [result.action_of(role) for result in experiment.rounds]
    predictability = markov_predictability(actions) if len(actions) >= 2 else math.nan
    return {
        "agent": stats.agent_type.value,
        "role": role.value,
        "opponent": opponent.value,
        "seed": experiment.config.seed,
        "num_rounds": experiment.config.num_rounds,
        "goal_rate": stats.goal_rate,
        "save_rate": stats.save_rate,
        "mean_reward": stats.mean_reward,
        "mean_decision_time": stats.mean_decision_time,
        "equilibrium_gap": stats.equilibrium_gap,
        "exploitability": stats.exploitability,
        "predictability": predictability,
    }


def experiments_to_samples(experiments: Sequence[Experiment]) -> pd.DataFrame:
    """Transforme des expériences en échantillons (deux lignes chacune).

    Pour chaque expérience, une ligne par agent (attaquant et gardien). Les
    colonnes sont ``agent``, ``role``, ``opponent``, ``seed``, ``num_rounds``
    puis les métriques de ``METRIC_COLUMNS``. ``goal_rate`` et ``save_rate``
    sont relatifs aux manches jouées par l'agent : pour un gardien,
    ``goal_rate`` est le taux de buts concédés.

    Args:
        experiments: Expériences à convertir.

    Returns:
        Un tableau avec une ligne par agent et par expérience.

    Raises:
        ValueError: Si aucune expérience n'est fournie.
    """
    if not experiments:
        raise ValueError("Au moins une expérience est nécessaire.")
    rows: list[dict[str, object]] = []
    for experiment in experiments:
        attacker = experiment.attacker_stats
        goalkeeper = experiment.goalkeeper_stats
        rows.append(
            _sample_row(experiment, Role.ATTACKER, attacker, goalkeeper.agent_type)
        )
        rows.append(
            _sample_row(experiment, Role.GOALKEEPER, goalkeeper, attacker.agent_type)
        )
    columns = [*KEY_COLUMNS, *PARAMETER_COLUMNS, *METRIC_COLUMNS]
    return pd.DataFrame(rows, columns=columns)


def _finite(values: Sequence[float]) -> list[float]:
    """Retire les valeurs non finies (NaN) d'une liste de mesures.

    Args:
        values: Mesures à filtrer.

    Returns:
        Les mesures finies, dans l'ordre.
    """
    return [value for value in values if math.isfinite(value)]


def summarize_table(samples: pd.DataFrame, *, by_opponent: bool = True) -> pd.DataFrame:
    """Résume les échantillons par agent, rôle et (option) adversaire.

    Pour chaque métrique ``m``, le tableau contient ``m_mean``, ``m_std`` et
    ``m_ci95`` (IC à 95 %, loi de Student). Avec ``by_opponent=False``, tous
    les adversaires sont agrégés (colonne ``opponent`` = ``"all"``) : les
    échantillons de toutes les paires et de toutes les graines sont mis en
    commun, ce qui suppose le même nombre de graines par adversaire (garanti
    par ``run_benchmark``).

    Args:
        samples: Tableau produit par ``experiments_to_samples``.
        by_opponent: Séparer ou non les adversaires.

    Returns:
        Un tableau avec une ligne par groupe et la colonne ``n_seeds``
        (nombre de mesures par groupe).

    Raises:
        ValueError: Si le tableau est vide ou s'il manque des colonnes.
    """
    missing = [c for c in (*KEY_COLUMNS, *METRIC_COLUMNS) if c not in samples.columns]
    if samples.empty or missing:
        raise ValueError(f"Échantillons vides ou colonnes manquantes : {missing}.")
    frame = samples if by_opponent else samples.assign(opponent=ALL_OPPONENTS)
    rows: list[dict[str, object]] = []
    for (agent, role, opponent), group in frame.groupby(list(KEY_COLUMNS), sort=False):
        row: dict[str, object] = {
            "agent": agent,
            "role": role,
            "opponent": opponent,
            "n_seeds": len(group),
        }
        for metric in METRIC_COLUMNS:
            values = _finite([float(v) for v in group[metric]])
            if values:
                summary = summarize_samples(values)
                row[f"{metric}_mean"] = summary.mean
                row[f"{metric}_std"] = summary.std
                row[f"{metric}_ci95"] = summary.ci95
            else:
                row[f"{metric}_mean"] = math.nan
                row[f"{metric}_std"] = math.nan
                row[f"{metric}_ci95"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)
