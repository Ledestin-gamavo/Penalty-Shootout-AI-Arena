"""Agent Fictitious Play : meilleure réponse à la stratégie empirique cumulée."""
import numpy as np
from .base_agent import BaseAgent, own_utility, best_response_policy


class FictitiousPlayAgent(BaseAgent):
    """Procédure classique (Brown 1951, Robinson 1951) :
      1. compter toutes les actions de l'adversaire depuis le début (comptes cumulés) ;
      2. en déduire sa stratégie empirique (comptes / total) ;
      3. jouer la meilleure réponse à cette stratégie.
    Dans un jeu à somme nulle, les fréquences empiriques convergent vers un
    équilibre : on observe la convergence des FRÉQUENCES, pas de chaque coup.

    `prior` = pseudo-comptes initiaux (1 par action : on part d'une croyance uniforme).
    """
    name = "fictitious_play"

    def __init__(self, role, prior=1.0, seed=None):
        self.prior = prior
        super().__init__(role, seed)

    def reset(self):
        super().reset()
        self.counts = np.full(3, float(self.prior))   # comptes cumulés de l'adversaire

    def _observe(self, own_action, opponent_action):
        self.counts[opponent_action] += 1.0

    def empirical_opponent(self):
        return self.counts / self.counts.sum()

    def policy(self, observation):
        return best_response_policy(own_utility(observation.payoff, self.role), self.empirical_opponent())
