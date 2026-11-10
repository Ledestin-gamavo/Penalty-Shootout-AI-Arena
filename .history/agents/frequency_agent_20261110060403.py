"""Agent « fréquences » : heuristique adaptative à mémoire courte."""
from collections import deque
import numpy as np
from .base_agent import BaseAgent, own_utility, best_response_policy


class FrequencyAgent(BaseAgent):
    """Estime la distribution de l'adversaire sur ses `window` DERNIÈRES actions
    (fenêtre glissante), avec un lissage de Laplace `smoothing` (pseudo-comptes
    qui évitent les fréquences nulles), puis joue la meilleure réponse.

    Différence avec Fictitious Play (documentée comme l'exige le cahier) :
      * ici : mémoire COURTE (fenêtre) -> réagit vite aux changements de style ;
      * Fictitious Play : mémoire COMPLÈTE (comptes cumulés depuis le début).
    """
    name = "frequences"

    def __init__(self, role, window=10, smoothing=0.5, seed=None):
        if window < 1:
            raise ValueError("window doit être >= 1")
        self.window = window
        self.smoothing = smoothing
        super().__init__(role, seed)

    def reset(self):
        super().reset()
        self.recent = deque(maxlen=self.window)   # dernières actions adverses

    def _observe(self, own_action, opponent_action):
        self.recent.append(opponent_action)

    def estimated_opponent(self):
        """Fréquences lissées de l'adversaire dans la fenêtre (vecteur de somme 1)."""
        counts = np.full(3, float(self.smoothing))
        for a in self.recent:
            counts[a] += 1.0
        return counts / counts.sum()

    def policy(self, observation):
        return best_response_policy(own_utility(observation.payoff, self.role), self.estimated_opponent())
