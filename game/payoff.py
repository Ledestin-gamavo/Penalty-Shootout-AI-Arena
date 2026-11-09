"""Matrice des gains et lecture des récompenses.

Ce module est la source unique concernant les gains. Les gains sont
exprimés du point de vue de l'attaquant ; celui du gardien est l'opposé
(``RoundResult.goalkeeper_reward``), ce qui garantit la somme nulle.

Dépendances autorisées : ``game.actions``, ``game.constants`` et
``game.models``.
"""

from game.actions import Action
from game.constants import REWARD_GOAL, REWARD_SAVE
from game.models import PayoffMatrix

def build_reference_matrix() -> PayoffMatrix:
     """Construit la matrice de référence 3x3 des gains de l'attaquant.

     Le ballon est arrêté (``REWARD_SAVE``) lorsque le gardien plonge dans la
     zone du tir ; sinon, un but est marqué (``REWARD_GOAL``). Les lignes
     correspondent à l'action de l'attaquant et les colonnes à celle du
     gardien, dans l'ordre ``LEFT``, ``CENTER``, ``RIGHT``.

     Returns:
        La matrice des gains de l'attaquant, indexée par
        ``matrix[attaquant][gardien]``.
     """
     save, goal = REWARD_SAVE, REWARD_GOAL
     return (
          (save, goal, goal),
          (goal, save, goal),
          (goal, goal, save),
     )

def resolve_payoff_matrix(payoff_matrix: PayoffMatrix | None) -> PayoffMatrix:
   """Retourne la matrice des gains effectivement utilisée.

    Cette fonction centralise la règle « ``None`` signifie matrice de
    référence » (voir ``MatchConfig.payoff_matrix``) pour que le moteur,
    l'interface et le benchmark ne la réécrivent pas.

    Args:
        payoff_matrix: Matrice personnalisée, ou ``None``.

    Returns:
        La matrice fournie, ou la matrice de référence si l'argument est
        ``None``.
    """
   if payoff_matrix is None:
      return build_reference_matrix()
   return payoff_matrix

def attacker_reward(attacker_action: Action,
                     goalkeeper_action: Action,
                     payoff_matrix: PayoffMatrix,) -> float:
     """Retourne le gain de l'attaquant pour un couple d'actions.

     Cette fonction ne valide pas ses entrées : la validation des actions
     relève de ``game.rules`` et celle de la matrice de
     ``theory.validation``.

     Args:
        attacker_action: Zone choisie par l'attaquant (ligne de la matrice).
        goalkeeper_action: Zone choisie par le gardien (colonne de la
            matrice).
        payoff_matrix: Matrice des gains de l'attaquant.

     Returns:
        Le gain de l'attaquant. Celui du gardien est son opposé.
     """
     return payoff_matrix[attacker_action][goalkeeper_action]
     