"""Agent Minimax : stratégie mixte optimale issue de solve_equilibrium."""
from agents.base_agent import BaseAgent
from game.actions import Action
from game.models import Equilibrium, Observation, PayoffMatrix
from theory.equilibrium import solve_equilibrium


class MinimaxAgent(BaseAgent):
    """Joue la stratégie d'équilibre de son rôle, sans rien apprendre.

    Le tirage dans ``choose_action`` exécute la stratégie mixte calculée ;
    ce n'est pas un agent aléatoire.
    """

    def __init__(self, payoff_matrix: PayoffMatrix, seed: int) -> None:
        """Calcule l'équilibre une seule fois.

        Args:
            payoff_matrix: Matrice des gains de l'attaquant.
            seed: Graine du générateur aléatoire.

        Raises:
            ValueError: Si la matrice est invalide.
            RuntimeError: Si le solveur échoue.
        """
        super().__init__(payoff_matrix, seed)
        self._equilibrium: Equilibrium = solve_equilibrium(payoff_matrix)

    @property
    def game_value(self) -> float:
        """Valeur du jeu du point de vue de l'attaquant."""
        return self._equilibrium.value

    def choose_action(self, observation: Observation) -> Action:
        """Tire une action selon la stratégie d'équilibre du rôle courant.

        Args:
            observation: Observation autorisée ; son rôle choisit la stratégie.

        Returns:
            L'action tirée.
        """
        strategy = self._equilibrium.strategy_of(observation.role)
        return Action(int(self.rng.choice(len(strategy), p=strategy)))