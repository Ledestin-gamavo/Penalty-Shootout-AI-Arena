"""Définition de l'énumération des actions du jeu.

Ce module ne dépend d'aucun autre module du projet. Il est importé par tous
les autres modules qui manipulent des actions.
"""

from enum import IntEnum
from typing import Final

class Action(IntEnum):
     """Zone choisie par un joueur lors d'une manche.

    L'attaquant tire dans une zone, le gardien plonge dans une zone. Les
    valeurs entières servent directement d'indices dans la matrice des gains
    et dans les vecteurs de probabilités (``matrix[attaquant][gardien]``).

    Remarque:
        ``str(Action.LEFT)`` renvoie ``"0"`` (comportement d'``IntEnum``).
        Utiliser ``Action.name`` pour un identifiant technique et
        ``Action.label`` pour un affichage destiné à l'utilisateur.

    Attributes:
        LEFT: Zone de gauche (indice 0).
        CENTER: Zone centrale (indice 1).
        RIGHT: Zone de droite (indice 2).
    """

     LEFT = 0
     CENTER = 1
     RIGHT = 2

     @property
     def label(self) -> str:
          """Retourne le libellé français de l'action.

          Returns:
            Le libellé destiné à l'interface et aux graphiques.
          """
          return _ACTION_LABELS[self]

_ACTION_LABELS: Final[dict[Action, str]] = {
     Action.LEFT: "Gauche",
     Action.CENTER: "Centre",
     Action.RIGHT: "Droite"
}
          