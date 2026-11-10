"""Agent déterministe : suit une séquence fixe, sans jamais s'adapter."""

from collections.abc import Sequence
from agents.base_agent import BaseAgent
from game.actions import Action
from game.models import Observation, PayoffMatrix


class DeterministicAgent(BaseAgent):
    """Rejoue une séquence d'actions en boucle.

    Sert de référence : une stratégie pure répétée est exploitable par une
    meilleure réponse.
    """

    def __init__(self, payoff_matrix: PayoffMatrix, seed: int, pattern: Sequence[Action] = (Action.LEFT,),
                                    ) -> None:
        """Initialise l'agent.

        Args:
            payoff_matrix: Matrice des gains de l'attaquant.
            seed: Graine du générateur aléatoire.
            pattern: Séquence non vide d'actions rejouée en boucle.

        Raises:
            ValueError: Si la séquence est vide ou contient une action invalide.
        """
        if len(pattern) == 0:
            raise ValueError("pattern : séquence non vide requise")
        self.pattern: tuple[Action, ...] = tuple(Action(a) for a in pattern)
        super().__init__(payoff_matrix, seed)

    def choose_action(self, observation: Observation) -> Action:
        """Retourne l'action suivante de la séquence.

        Args:
            observation: Observation autorisée (non utilisée).

        Returns:
            L'action de la séquence correspondant à la manche courante.
        """
        return self.pattern[self.t % len(self.pattern)]