"""Constantes visuelles communes aux figures Plotly."""

from typing import Final

from game.actions import Action
from game.models import AgentType

# Hauteur par défaut des figures, en pixels.
FIGURE_HEIGHT: Final[int] = 420

# Nombre maximal de points tracés par courbe (sous-échantillonnage au-delà).
MAX_PLOT_POINTS: Final[int] = 2_000

# Couleur de la stratégie de référence (équilibre).
REFERENCE_COLOR: Final[str] = "#7F7F7F"

ACTION_COLORS: Final[dict[Action, str]] = {
    Action.LEFT: "#4C78A8",
    Action.CENTER: "#F58518",
    Action.RIGHT: "#54A24B",
}

AGENT_COLORS: Final[dict[AgentType, str]] = {
    AgentType.DETERMINISTIC: "#4C78A8",
    AgentType.FREQUENCY: "#F58518",
    AgentType.MINIMAX: "#54A24B",
    AgentType.FICTITIOUS_PLAY: "#B279A2",
}
