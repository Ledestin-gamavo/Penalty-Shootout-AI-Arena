"""Agent Fictitious Play : meilleure réponse à la stratégie empirique cumulée."""

import numpy as np
from numpy.typing import NDArray
from agents.base_agent import BaseAgent
from game.actions import Action
from game.models import Observation, PayoffMatrix, Strategy
from theory.equilibrium import best_response


class FictitiousPlayAgent(BaseAgent):
    """Compte toutes les actions adverses depuis le début (mémoire complète),
    en déduit la stratégie empirique, puis joue la meilleure réponse.

    Différence avec l'agent Fréquences : ici le comptage est cumulatif, là-bas
    il porte sur une fenêtre glissante.
    """

    def __init__(
        self, payoff_matrix: PayoffMatrix, seed: int, prior: float = 1.0
    ) -> None:
        """Initialise l'agent.

        Args:
            payoff_matrix: Matrice des gains de l'attaquant.
            seed: Graine du générateur aléatoire.
            prior: Pseudo-comptes initiaux par action (1 = croyance uniforme).

        Raises:
            ValueError: Si ``prior`` n'est pas strictement positif.
        """
        if prior <= 0:
            raise ValueError("prior doit être > 0")
        self.prior: float = prior
        super().__init__(payoff_matrix, seed)

    def reset(self) -> None:
        """Réinitialise les comptes et le générateur."""
        super().reset()
        self.counts: NDArray[np.float64] = np.full(
            len(Action), self.prior, dtype=np.float64
        )

    def _observe(self, own_action: Action, opponent_action: Action) -> None:
        self.counts[opponent_action] += 1.0

    def empirical_opponent(self) -> Strategy:
        """Retourne la stratégie empirique cumulée de l'adversaire."""
        freq = self.counts / self.counts.sum()
        return tuple(float(p) for p in freq)

    def choose_action(self, observation: Observation) -> Action:
        """Joue la meilleure réponse à la stratégie empirique.

        Args:
            observation: Observation autorisée ; son rôle fixe l'orientation.

        Returns:
            La meilleure réponse pure du rôle courant.
        """
        return best_response(
            self.payoff_matrix, observation.role, self.empirical_opponent()
        )