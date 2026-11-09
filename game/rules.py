"""Règles du jeu : validation des actions et résolution d'une manche.

Ce module applique les règles métier du moteur. Il ne connaît ni les agents,
ni l'interface. Il est l'unique endroit où un ``RoundResult`` est fabriqué.

Dépendances autorisées : ``game.actions``, ``game.models``, ``game.payoff``
et ``theory.validation``.
"""

from collections.abc import Sequence
from typing import Final
from game.actions import Action
from game.models import MatchConfig, PayoffMatrix, RoundResult
from game.payoff import attacker_reward
from theory.validation import validate_payoff_matrix

_VALID_ACTIONS: Final[tuple[Action, ...]] = tuple(Action)


def validate_action(value: object) -> Action:
    """Convertit une valeur en ``Action`` valide.

    Sont acceptés : un membre de ``Action`` ou l'indice entier
    correspondant (0, 1 ou 2). Les booléens sont refusés, car ``True`` et
    ``False`` seraient sinon pris pour les indices 1 et 0.

    Args:
        value: Valeur à convertir.

    Returns:
        L'action correspondante.

    Raises:
        ValueError: Si la valeur ne correspond à aucune action.
    """
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value in _VALID_ACTIONS
    ):
        return Action(value)
    raise ValueError(
        f"Action invalide : {value!r}. "
        f"Valeurs autorisées : {[action.name for action in Action]}."
    )


def validate_config(config: MatchConfig) -> None:
    """Vérifie les règles métier d'une configuration de match.

    Les invariants simples (nombre de manches, graine, présence d'au moins
    un agent) sont déjà garantis par ``MatchConfig``. Cette fonction
    vérifie la matrice personnalisée, si elle est fournie.

    Args:
        config: Configuration à vérifier.

    Raises:
        ValueError: Si la matrice des gains personnalisée est invalide.
    """
    if config.payoff_matrix is not None:
        validate_payoff_matrix(config.payoff_matrix)


def resolve_round(
    round_number: int,
    attacker_action: Action,
    goalkeeper_action: Action,
    payoff_matrix: PayoffMatrix,
) -> RoundResult:
    """Résout une manche à partir des deux actions révélées.

    Args:
        round_number: Numéro de la manche, à partir de 1.
        attacker_action: Zone choisie par l'attaquant.
        goalkeeper_action: Zone choisie par le gardien.
        payoff_matrix: Matrice des gains de l'attaquant.

    Returns:
        Le résultat de la manche.

    Raises:
        ValueError: Si le numéro de manche est inférieur à 1 ou si une des
            actions est invalide.
    """
    if round_number < 1:
        raise ValueError(
            f"round_number doit être au moins 1, reçu : {round_number}."
        )
    attacker = validate_action(attacker_action)
    goalkeeper = validate_action(goalkeeper_action)
    return RoundResult(
        round_number=round_number,
        attacker_action=attacker,
        goalkeeper_action=goalkeeper,
        attacker_reward=attacker_reward(attacker, goalkeeper, payoff_matrix),
    )


def compute_score(results: Sequence[RoundResult]) -> tuple[int, int]:
    """Calcule le score d'une suite de manches.

    Args:
        results: Résultats des manches.

    Returns:
        Un couple ``(buts, arrêts)``.
    """
    goals = sum(result.is_goal for result in results)
    return goals, len(results) - goals