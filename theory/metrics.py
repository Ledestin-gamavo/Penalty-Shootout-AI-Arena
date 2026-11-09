"""Indicateurs de performance des agents.

Les fonctions sont pures et indépendantes de l'interface : elles servent aux
simulations IA contre IA, au laboratoire et au benchmark. Les définitions
mathématiques sont décrites dans ``docs/game_theory.md``.

Dépendances autorisées : ``game.actions``, ``game.models``, ``game.rules``,
``theory.equilibrium`` et ``theory.validation``.
"""

import math
from collections.abc import Sequence
import numpy as np
from game.actions import Action
from game.models import (AgentStats, AgentType, PayoffMatrix, Role, RoundResult, Strategy)
from game.rules import compute_score
from theory.equilibrium import solve_equilibrium
from theory.validation import validate_strategy

def _require_rounds(rounds: Sequence[RoundResult]) -> None:
     """Vérifie qu'une métrique dispose d'au moins une manche.

     Args:
        rounds: Manches à analyser.

     Raises:
        ValueError: Si la séquence est vide.
     """
     if not rounds:
        raise ValueError("Au moins une manche est nécessaire pour cette métrique.")

def empirical_strategy(actions: Sequence[Action]) -> Strategy:
    """Calcule la stratégie empirique à partir d'actions jouées.

    Args:
        actions: Actions jouées par un joueur, dans l'ordre.

    Returns:
        La fréquence de chaque action, indexée par ``Action``.

    Raises:
        ValueError: Si aucune action n'est fournie.
    """
    if not actions:
        raise ValueError("Au moins une action est nécessaire.")
    return tuple(actions.count(action) / len(actions) for action in Action)

def goal_rate(rounds: Sequence[RoundResult]) -> float:
    """Calcule le taux de buts (buts / manches).

    Args:
        rounds: Manches à analyser.

    Returns:
        Le taux de buts, entre 0 et 1.

    Raises:
        ValueError: Si la séquence est vide.
    """
    _require_rounds(rounds)
    goals, _ = compute_score(rounds)
    return goals / len(rounds)

def save_rate(rounds: Sequence[RoundResult]) -> float:
    """Calcule le taux d'arrêts (arrêts / manches).

    Args:
        rounds: Manches à analyser.

    Returns:
        Le taux d'arrêts, entre 0 et 1.

    Raises:
        ValueError: Si la séquence est vide.
    """
    _require_rounds(rounds)
    _, saves = compute_score(rounds)
    return saves / len(rounds)

def average_reward(rounds: Sequence[RoundResult], role: Role) -> float:
    """Calcule le gain moyen par manche d'un rôle.

    Args:
        rounds: Manches à analyser.
        role: Rôle dont on mesure le gain.

    Returns:
        Le gain moyen du rôle par manche.

    Raises:
        ValueError: Si la séquence est vide.
    """
    _require_rounds(rounds)
    return sum(result.reward_for(role) for result in rounds) / len(rounds)

def distance_to_equilibrium(strategy: Strategy, reference: Strategy) -> float:
    """Calcule la distance en variation totale entre deux stratégies.

    La distance vaut ``0.5 * somme(|p_i - q_i|)``. Elle est comprise entre 0
    (stratégies identiques) et 1 (supports disjoints).

    Args:
        strategy: Stratégie évaluée (par exemple empirique).
        reference: Stratégie de référence (par exemple d'équilibre).

    Returns:
        La distance en variation totale.

    Raises:
        ValueError: Si les deux stratégies n'ont pas la même taille.
    """
    pairs = zip(strategy, reference, strict=True)
    return 0.5 * math.fsum(abs(p - q) for p, q in pairs)

def exploitability(payoff_matrix: PayoffMatrix, role: Role, strategy: Strategy
                                   ) -> float:
    """Calcule l'exploitabilité d'une stratégie.

    L'exploitabilité est le gain supplémentaire qu'une meilleure réponse de
    l'adversaire obtient par rapport à la valeur du jeu ``v`` :

    - attaquant : ``v - min_j (p^T A)_j`` ;
    - gardien : ``max_i (A q)_i - v``.

    Elle est positive ou nulle, et nulle pour une stratégie optimale. Un
    résultat légèrement négatif dû au bruit numérique est ramené à 0.

    Args:
        payoff_matrix: Matrice des gains de l'attaquant.
        role: Rôle qui joue la stratégie évaluée.
        strategy: Stratégie évaluée.

    Returns:
        L'exploitabilité de la stratégie.

    Raises:
        ValueError: Si la stratégie ou la matrice est invalide.
    """
    validate_strategy(strategy)
    value = solve_equilibrium(payoff_matrix).value
    matrix = np.asarray(payoff_matrix, dtype=np.float64)
    weights = np.asarray(strategy, dtype=np.float64)
    if role is Role.ATTACKER:
        gap = value - float(np.min(weights @ matrix))
    else:
        gap = float(np.max(matrix @ weights)) - value
    return max(0.0, gap)

def compute_agent_stats(agent_type: AgentType, role: Role, rounds: Sequence[RoundResult], mean_decision_time: float, payoff_matrix: PayoffMatrix,
                                   ) -> AgentStats:
    """Agrège les statistiques d'un agent sur un match.

    Args:
        agent_type: Type de l'agent évalué.
        role: Rôle tenu par l'agent.
        rounds: Manches jouées, dans l'ordre.
        mean_decision_time: Temps moyen de décision, en secondes.
        payoff_matrix: Matrice des gains de l'attaquant.

    Returns:
        Les statistiques agrégées de l'agent.

    Raises:
        ValueError: Si aucune manche n'est fournie.
    """
    goals, saves = compute_score(rounds)
    strategy = empirical_strategy([result.action_of(role) for result in rounds])
    reference = solve_equilibrium(payoff_matrix).strategy_of(role)
    return AgentStats(
        agent_type=agent_type,
        role=role,
        rounds=len(rounds),
        goals=goals,
        saves=saves,
        mean_reward=average_reward(rounds, role),
        mean_decision_time=mean_decision_time,
        empirical_strategy=strategy,
        equilibrium_gap=distance_to_equilibrium(strategy, reference),
        exploitability=exploitability(payoff_matrix, role, strategy),
    )


