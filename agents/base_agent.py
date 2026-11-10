"""Classe abstraite des agents (contrat d'API, §7.7)."""

from abc import ABC, abstractmethod
import numpy as np
from game.actions import Action
from game.models import Observation, PayoffMatrix
from theory.validation import validate_payoff_matrix


class BaseAgent(ABC):
    """Classe mère des agents.

    L'agent ne connaît que la matrice des gains (information publique), son
    historique et l'observation autorisée. Son rôle est lu dans
    ``observation.role`` à chaque décision, jamais stocké.

    Attributes:
        payoff_matrix: Matrice des gains de l'attaquant.
        seed: Graine du générateur aléatoire de l'agent.
        rng: Générateur aléatoire propre à l'agent.
        t: Nombre de manches terminées vues par l'agent.
    """

    def __init__(self, payoff_matrix: PayoffMatrix, seed: int) -> None:
        """Initialise l'agent.

        Args:
            payoff_matrix: Matrice des gains de l'attaquant.
            seed: Graine du générateur aléatoire.

        Raises:
            ValueError: Si la matrice est invalide.
        """
        validate_payoff_matrix(payoff_matrix)
        self.payoff_matrix: PayoffMatrix = payoff_matrix
        self.seed: int = seed
        self.reset()

    @abstractmethod
    def choose_action(self, observation: Observation) -> Action:
        """Choisit une action. Reçoit UNIQUEMENT l'observation autorisée.

        Args:
            observation: Rôle et numéro de manche.

        Returns:
            L'action choisie.
        """

    def update(
        self,
        observation: Observation,
        own_action: Action,
        opponent_action: Action,
        reward: float,
    ) -> None:
        """Met à jour l'agent APRÈS la révélation des deux actions.

        Args:
            observation: Observation de la manche terminée.
            own_action: Action jouée par l'agent.
            opponent_action: Action jouée par l'adversaire.
            reward: Gain de l'agent pour cette manche.

        Raises:
            ValueError: Si une des actions est invalide.
        """
        self._observe(Action(own_action), Action(opponent_action))
        self.t += 1

    def reset(self) -> None:
        """Réinitialise l'état interne ET le générateur (graine d'origine)."""
        self.rng: np.random.Generator = np.random.default_rng(self.seed)
        self.t: int = 0

    def _observe(self, own_action: Action, opponent_action: Action) -> None:
        """Mémorise une manche terminée (à surcharger si besoin).

        Args:
            own_action: Action jouée par l'agent.
            opponent_action: Action jouée par l'adversaire.
        """