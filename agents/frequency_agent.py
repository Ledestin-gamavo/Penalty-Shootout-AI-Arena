"""Agent « fréquences » : heuristique adaptative à mémoire courte."""
from collections import deque

import numpy as np

from agents.base_agent import BaseAgent
from game.actions import Action
from game.models import Observation, PayoffMatrix, Strategy
from theory.equilibrium import best_response


class FrequencyAgent(BaseAgent):
    """Estime la distribution adverse sur les ``window`` dernières manches,
    avec un lissage de Laplace, puis joue la meilleure réponse.

    Différence avec Fictitious Play : mémoire courte (fenêtre), donc réaction
    rapide aux changements de style de l'adversaire.
    """

    def __init__(
        self,
        payoff_matrix: PayoffMatrix,
        seed: int,
        window: int = 10,
        smoothing: float = 0.5,
    ) -> None:
        """Initialise l'agent.

        Args:
            payoff_matrix: Matrice des gains de l'attaquant.
            seed: Graine du générateur aléatoire.
            window: Taille de la fenêtre glissante (>= 1).
            smoothing: Pseudo-comptes de lissage par action (> 0).

        Raises:
            ValueError: Si ``window`` ou ``smoothing`` est invalide.
        """
        if window < 1:
            raise ValueError("window doit être >= 1")
        if smoothing <= 0:
            raise ValueError("smoothing doit être > 0")
        self.window: int = window
        self.smoothing: float = smoothing
        super().__init__(payoff_matrix, seed)

    def reset(self) -> None:
        """Vide la fenêtre et réinitialise le générateur."""
        super().reset()
        self.recent: deque[Action] = deque(maxlen=self.window)

    def _observe(self, own_action: Action, opponent_action: Action) -> None:
        self.recent.append(opponent_action)

    def estimated_opponent(self) -> Strategy:
        """Retourne les fréquences lissées de l'adversaire dans la fenêtre."""
        counts = np.full(len(Action), self.smoothing, dtype=np.float64)
        for action in self.recent:
            counts[action] += 1.0
        return tuple(float(p) for p in counts / counts.sum())

    def choose_action(self, observation: Observation) -> Action:
        """Joue la meilleure réponse aux fréquences estimées.

        Args:
            observation: Observation autorisée ; son rôle fixe l'orientation.

        Returns:
            La meilleure réponse pure du rôle courant.
        """
        return best_response(
            self.payoff_matrix, observation.role, self.estimated_opponent()
        )