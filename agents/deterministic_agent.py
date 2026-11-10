"""Agent déterministe : suit une séquence fixe, sans jamais s'adapter."""
import numpy as np
from .base_agent import BaseAgent


class DeterministicAgent(BaseAgent):
    """Rejoue une séquence d'actions en boucle, par exemple (0,) = toujours à gauche,
    ou (0, 1, 2) = gauche, centre, droite, gauche, ...

    Sert de référence : une stratégie pure répétée est prévisible, donc
    exploitable par une meilleure réponse (niveau 1 du cahier des charges).
    """
    name = "deterministe"

    def __init__(self, role, pattern=(0,), seed=None):
        if len(pattern) == 0 or any(a not in (0, 1, 2) for a in pattern):
            raise ValueError("pattern : séquence non vide d'actions 0, 1 ou 2")
        self.pattern = tuple(pattern)
        super().__init__(role, seed)

    def policy(self, observation):
        p = np.zeros(3)
        p[self.pattern[self.t % len(self.pattern)]] = 1.0   # t = nombre de manches terminées
        return p
