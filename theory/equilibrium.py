"""Équilibre minimax, valeur du jeu et meilleure réponse.

Le même programme linéaire sert aux deux rôles : on le résout du point de
vue du rôle, dont la matrice est celle de l'attaquant (``A``) ou l'opposé de
sa transposée (``-A^T``) pour le gardien.

Dépendances autorisées : ``game.actions``, ``game.constants``,
``game.models`` et ``theory.validation``.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog
from game.actions import Action
from game.constants import N_ACTIONS, NUMERICAL_NOISE_TOLERANCE
from game.models import Equilibrium, PayoffMatrix, Role, Strategy
from theory.validation import validate_payoff_matrix, validate_strategy


def _role_matrix(payoff_matrix: PayoffMatrix, role: Role) -> NDArray[np.float64]:
    """Construit la matrice des gains du point de vue d'un rôle.

    Args:
        payoff_matrix: Matrice des gains de l'attaquant.
        role: Rôle dont on adopte le point de vue.

    Returns:
        Une matrice dont les lignes sont les actions du rôle et les colonnes
        celles de l'adversaire. Chaque rôle cherche à la maximiser.
    """
    matrix = np.asarray(payoff_matrix, dtype=np.float64)
    return matrix if role is Role.ATTACKER else -matrix.T


def _to_strategy(raw: NDArray[np.float64]) -> Strategy:
    """Transforme le vecteur brut de ``linprog`` en stratégie valide.

    Les valeurs comprises entre ``-NUMERICAL_NOISE_TOLERANCE`` et 0 sont
    ramenées à 0, puis le vecteur est renormalisé. Une valeur négative plus
    marquée n'est pas corrigée : ``validate_strategy`` la rejette.

    Args:
        raw: Probabilités brutes retournées par le solveur.

    Returns:
        La stratégie nettoyée et validée.

    Raises:
        ValueError: Si le résultat n'est pas une distribution valide.
    """
    cleaned = np.where((raw < 0.0) & (raw >= -NUMERICAL_NOISE_TOLERANCE), 0.0, raw)
    normalized = cleaned / cleaned.sum()
    strategy = tuple(float(probability) for probability in normalized)
    validate_strategy(strategy)
    return strategy


def _solve_for_role(payoff_matrix: PayoffMatrix, role: Role) -> tuple[Strategy, float]:
    """Calcule la stratégie maximin d'un rôle par programmation linéaire.

    Le programme est : maximiser ``v`` sous les contraintes
    ``sum_i p_i * M[i][k] >= v`` pour chaque colonne ``k``, ``sum_i p_i = 1``
    et ``p_i >= 0``, où ``M`` est la matrice du point de vue du rôle.

    Args:
        payoff_matrix: Matrice des gains de l'attaquant.
        role: Rôle dont on cherche la stratégie optimale.

    Returns:
        Un couple ``(stratégie optimale, valeur du jeu pour ce rôle)``.

    Raises:
        RuntimeError: Si le solveur ne trouve pas de solution.
        ValueError: Si la solution n'est pas une distribution valide.
    """
    matrix = _role_matrix(payoff_matrix, role)
    # Variables : (p_1, ..., p_n, v). On minimise -v.
    cost = np.zeros(N_ACTIONS + 1)
    cost[-1] = -1.0
    inequality_matrix = np.hstack([-matrix.T, np.ones((N_ACTIONS, 1))])
    equality_matrix = np.append(np.ones(N_ACTIONS), 0.0).reshape(1, -1)
    result = linprog(
        cost,
        A_ub=inequality_matrix,
        b_ub=np.zeros(N_ACTIONS),
        A_eq=equality_matrix,
        b_eq=[1.0],
        bounds=[(0.0, None)] * N_ACTIONS + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            f"Échec de linprog pour le rôle {role.value} : {result.message}"
        )
    return _to_strategy(result.x[:N_ACTIONS]), float(result.x[N_ACTIONS])


def solve_equilibrium(payoff_matrix: PayoffMatrix) -> Equilibrium:
    """Calcule les stratégies minimax et la valeur du jeu.

    Args:
        payoff_matrix: Matrice des gains de l'attaquant.

    Returns:
        L'équilibre : stratégies optimales des deux rôles et valeur du jeu
        du point de vue de l'attaquant.

    Raises:
        ValueError: Si la matrice est invalide.
        RuntimeError: Si le solveur échoue.
    """
    validate_payoff_matrix(payoff_matrix)
    attacker_strategy, value = _solve_for_role(payoff_matrix, Role.ATTACKER)
    goalkeeper_strategy, _ = _solve_for_role(payoff_matrix, Role.GOALKEEPER)
    return Equilibrium(
        attacker_strategy=attacker_strategy,
        goalkeeper_strategy=goalkeeper_strategy,
        value=value,
    )


def best_response(payoff_matrix: PayoffMatrix, role: Role, opponent_strategy: Strategy,) -> Action:
    """Retourne la meilleure réponse pure à une stratégie adverse.

    En cas d'égalité (à ``NUMERICAL_NOISE_TOLERANCE`` près), l'action
    d'indice le plus faible est retenue : le comportement est déterministe
    et reproductible.

    Args:
        payoff_matrix: Matrice des gains de l'attaquant, déjà validée.
        role: Rôle qui répond.
        opponent_strategy: Stratégie (mixte ou empirique) de l'adversaire.

    Returns:
        L'action maximisant le gain espéré du rôle.

    Raises:
        ValueError: Si la stratégie adverse n'est pas une distribution valide.
    """
    validate_strategy(opponent_strategy)
    expected = _role_matrix(payoff_matrix, role) @ np.asarray(opponent_strategy)
    candidates = np.flatnonzero(expected >= expected.max() - NUMERICAL_NOISE_TOLERANCE)
    return Action(int(candidates[0]))