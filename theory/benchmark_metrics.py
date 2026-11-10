"""Métriques complémentaires pour le laboratoire et le benchmark.

Ce module complète ``theory.metrics`` (propriété de M1) sans le modifier :
prévisibilité séquentielle, évolution cumulée des fréquences et du gain,
résumé statistique d'un échantillon de graines et cases de meilleure réponse
pour la lecture de la matrice. Les définitions sont décrites dans
``docs/benchmark_protocol.md``.

Dépendances autorisées : ``game.actions``, ``game.constants``,
``game.models``, ``theory.validation`` et SciPy.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from game.actions import Action
from game.constants import N_ACTIONS, NUMERICAL_NOISE_TOLERANCE
from game.models import PayoffMatrix, Role, RoundResult, Strategy
from theory.validation import validate_payoff_matrix

# Niveau de confiance des intervalles de confiance (bilatéral, 95 %).
_CONFIDENCE_QUANTILE: float = 0.975


@dataclass(frozen=True, slots=True)
class SampleSummary:
    """Résumé statistique d'un échantillon de mesures (une par graine).

    Attributes:
        n: Nombre de mesures.
        mean: Moyenne de l'échantillon.
        std: Écart-type corrigé (``ddof=1``), 0 si ``n`` vaut 1.
        ci95: Demi-largeur de l'intervalle de confiance à 95 % de la moyenne
            (loi de Student), 0 si ``n`` vaut 1.
    """

    n: int
    mean: float
    std: float
    ci95: float


def summarize_samples(values: Sequence[float]) -> SampleSummary:
    """Résume un échantillon par sa moyenne, son écart-type et son IC à 95 %.

    Args:
        values: Mesures indépendantes (par exemple une par graine).

    Returns:
        Le résumé statistique de l'échantillon.

    Raises:
        ValueError: Si l'échantillon est vide.
    """
    if not values:
        raise ValueError("Au moins une mesure est nécessaire.")
    n = len(values)
    mean = float(np.mean(values))
    if n == 1:
        return SampleSummary(n=1, mean=mean, std=0.0, ci95=0.0)
    std = float(np.std(values, ddof=1))
    critical = float(stats.t.ppf(_CONFIDENCE_QUANTILE, n - 1))
    return SampleSummary(n=n, mean=mean, std=std, ci95=critical * std / math.sqrt(n))


def markov_predictability(actions: Sequence[Action]) -> float:
    """Mesure la prévisibilité séquentielle d'une suite d'actions.

    Un prédicteur d'ordre 1 apprend en ligne, sans voir le futur, quelle
    action suit le plus souvent chaque action. La métrique est la part des
    actions (à partir de la deuxième) qu'il devine correctement. En cas
    d'égalité, il prédit l'action d'indice le plus faible.

    Contrairement à l'exploitabilité, elle détecte une séquence répétée
    G, C, D, G, C, D... : elle vaut 1 pour une séquence parfaitement
    prévisible et environ ``1 / N_ACTIONS`` pour une suite uniforme et
    indépendante.

    Args:
        actions: Actions jouées par un joueur, dans l'ordre.

    Returns:
        La précision du prédicteur, entre 0 et 1.

    Raises:
        ValueError: Si moins de deux actions sont fournies.
    """
    if len(actions) < 2:
        raise ValueError("Au moins deux actions sont nécessaires.")
    counts = np.zeros((N_ACTIONS, N_ACTIONS), dtype=np.int64)
    hits = 0
    for previous, current in zip(actions[:-1], actions[1:], strict=True):
        if int(np.argmax(counts[previous])) == current:
            hits += 1
        counts[previous, current] += 1
    return hits / (len(actions) - 1)


def cumulative_frequencies(actions: Sequence[Action]) -> tuple[Strategy, ...]:
    """Calcule la stratégie empirique après chaque manche.

    L'élément ``k`` est la fréquence de chaque action sur les ``k + 1``
    premières manches. Le calcul est linéaire en nombre de manches.

    Args:
        actions: Actions jouées par un joueur, dans l'ordre.

    Returns:
        Une stratégie (somme égale à 1) par manche jouée.

    Raises:
        ValueError: Si aucune action n'est fournie.
    """
    if not actions:
        raise ValueError("Au moins une action est nécessaire.")
    indices = np.asarray([int(action) for action in actions], dtype=np.int64)
    one_hot = np.eye(N_ACTIONS, dtype=np.float64)[indices]
    played = np.arange(1, len(indices) + 1, dtype=np.float64).reshape(-1, 1)
    frequencies = np.cumsum(one_hot, axis=0) / played
    return tuple(tuple(float(value) for value in row) for row in frequencies)


def cumulative_mean_reward(
    rounds: Sequence[RoundResult], role: Role
) -> tuple[float, ...]:
    """Calcule le gain moyen d'un rôle après chaque manche.

    Args:
        rounds: Manches jouées, dans l'ordre.
        role: Rôle dont on suit le gain.

    Returns:
        Le gain moyen cumulé après chaque manche.

    Raises:
        ValueError: Si aucune manche n'est fournie.
    """
    if not rounds:
        raise ValueError("Au moins une manche est nécessaire.")
    rewards = np.asarray([result.reward_for(role) for result in rounds])
    played = np.arange(1, len(rewards) + 1, dtype=np.float64)
    return tuple(float(value) for value in np.cumsum(rewards) / played)


def best_response_cells(
    payoff_matrix: PayoffMatrix, role: Role
) -> tuple[tuple[Action, Action], ...]:
    """Liste les cases de meilleure réponse pure d'un rôle.

    - Attaquant : pour chaque action du gardien, les actions de l'attaquant
      qui maximisent son gain.
    - Gardien : pour chaque action de l'attaquant, les actions du gardien
      qui minimisent le gain de l'attaquant.

    Toutes les actions à égalité sont retournées (à
    ``NUMERICAL_NOISE_TOLERANCE`` près), contrairement à
    ``theory.equilibrium.best_response`` qui en choisit une seule.

    Args:
        payoff_matrix: Matrice des gains de l'attaquant.
        role: Rôle dont on cherche les meilleures réponses.

    Returns:
        Les couples ``(action attaquant, action gardien)`` concernés.

    Raises:
        ValueError: Si la matrice est invalide.
    """
    validate_payoff_matrix(payoff_matrix)
    matrix = np.asarray(payoff_matrix, dtype=np.float64)
    cells: list[tuple[Action, Action]] = []
    if role is Role.ATTACKER:
        best = matrix.max(axis=0)
        for row, column in zip(*np.nonzero(matrix >= best - NUMERICAL_NOISE_TOLERANCE)):
            cells.append((Action(int(row)), Action(int(column))))
    else:
        best = matrix.min(axis=1).reshape(-1, 1)
        for row, column in zip(*np.nonzero(matrix <= best + NUMERICAL_NOISE_TOLERANCE)):
            cells.append((Action(int(row)), Action(int(column))))
    return tuple(sorted(cells))
