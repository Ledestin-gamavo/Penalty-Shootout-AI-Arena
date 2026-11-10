"""Agent Minimax : stratégie mixte optimale calculée par programmation linéaire."""
import numpy as np
from scipy.optimize import linprog
from .base_agent import BaseAgent, own_utility


def solve_maximin(utility):
    """Résout max_x min_j par programmation linéaire.

    Variables (x_1..x_m, v). On maximise vsous condition  pour
    chaque colonne j, sum_i x_i = 1, x >= 0, v libre (c'est le programme de J1 du cours).
    Retourne (x, v) : stratégie optimale et valeur garantie.
    """
    U = np.asarray(utility, dtype=float)
    m, n = U.shape
    c = np.r_[np.zeros(m), -1.0]                       # maximiser v = minimiser -v
    A_ub = np.hstack([-U.T, np.ones((n, 1))])          # v - U^T x <= 0
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n),
                  A_eq=[np.r_[np.ones(m), 0.0]], b_eq=[1.0],
                  bounds=[(0, None)] * m + [(None, None)], method="highs")
    if not res.success:
        raise RuntimeError(f"programmation linéaire non résolue : {res.message}")
    x = np.clip(res.x[:m], 0.0, None)
    return x / x.sum(), float(res.x[m])


class MinimaxAgent(BaseAgent):
    """Joue sa stratégie de sécurité (maximin) pour la matrice reçue. N'apprend rien :
    sa politique ne dépend pas de l'historique. Sur la matrice de base elle est
    uniforme (1/3, 1/3, 1/3) et la valeur est 1/3 pour l'attaquant.

    Le tirage au sort dans choose_action est l'exécution de la stratégie mixte
    calculée, pas un agent aléatoire.
    """
    name = "minimax"

    def reset(self):
        super().reset()
        self._cache = {}

    def _solve(self, observation):
        U = own_utility(observation.payoff, self.role)
        key = (self.role, U.tobytes())
        if key not in self._cache:
            self._cache[key] = solve_maximin(U)
        return self._cache[key]

    def policy(self, observation):
        return self._solve(observation)[0]

    def game_value(self, observation):
        """Valeur du jeu vue par l'agent  (attaquant : v ; gardien : -v)."""
        return self._solve(observation)[1]
