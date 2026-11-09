"""Validation des propriétés mathématiques du modèle.

Ce module vérifie les distributions de probabilités et la forme de la
matrice des gains. Il ne corrige jamais silencieusement une donnée : toute
anomalie lève une ``ValueError``.

La somme nulle est garantie par construction : une seule matrice (celle de
l'attaquant) est stockée et le gain du gardien en est l'opposé. La validation
de la matrice rejette donc les valeurs non finies, seules capables de rendre
cette propriété dénuée de sens.

Dépendances autorisées : ``game.constants`` et ``game.models``.
"""

import math
from game.constants import N_ACTIONS, PROBABILITY_TOLERANCE
from game.models import PayoffMatrix, Strategy


def validate_strategy(strategy: Strategy) -> None:
    """Vérifie qu'une stratégie est une distribution de probabilités valide.

    Une stratégie valide contient exactement ``N_ACTIONS`` valeurs finies,
    positives et de somme égale à 1, à ``PROBABILITY_TOLERANCE`` près.

    Args:
        strategy: Distribution de probabilités indexée par ``Action``.

    Raises:
        ValueError: Si la taille est incorrecte, si une valeur n'est pas
            finie ou est négative, ou si la somme diffère de 1.
    """
    if len(strategy) != N_ACTIONS:
        raise ValueError(
            f"Une stratégie doit contenir {N_ACTIONS} probabilités, "
            f"reçu : {len(strategy)}."
        )
    if not all(math.isfinite(p) for p in strategy):
        raise ValueError(f"Les probabilités doivent être finies : {strategy}.")
    if any(p < -PROBABILITY_TOLERANCE for p in strategy):
        raise ValueError(f"Les probabilités doivent être positives : {strategy}.")
    total = math.fsum(strategy)
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise ValueError(
            f"Les probabilités doivent sommer à 1, somme obtenue : {total}."
        )


def validate_payoff_matrix(payoff_matrix: PayoffMatrix) -> None:
    """Vérifie qu'une matrice des gains est exploitable par le moteur.

    La matrice doit être de taille ``N_ACTIONS`` x ``N_ACTIONS`` et ne
    contenir que des valeurs finies. Elle représente les gains de
    l'attaquant ; ceux du gardien en sont l'opposé, le jeu est donc à somme
    nulle par construction.

    Args:
        payoff_matrix: Matrice des gains de l'attaquant,
            ``matrix[attaquant][gardien]``.

    Raises:
        ValueError: Si la matrice n'est pas carrée de taille ``N_ACTIONS``
            ou si elle contient une valeur non finie.
    """
    if len(payoff_matrix) != N_ACTIONS or any(
        len(row) != N_ACTIONS for row in payoff_matrix
    ):
        raise ValueError(
            f"La matrice des gains doit être de taille {N_ACTIONS}x{N_ACTIONS}."
        )
    if not all(math.isfinite(value) for row in payoff_matrix for value in row):
        raise ValueError("La matrice des gains doit contenir des valeurs finies.")