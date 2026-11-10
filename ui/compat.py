"""Couche de branchement entre l'interface et les modules ``game/`` et ``agents/``.

L'interface manipule des actions et des rôles sous forme de texte
(``"gauche"``, ``"attaquant"``...) ; le moteur utilise ``Action`` et ``Role``.
Ce module traduit dans les deux sens et garde l'API historique
(``ACTIONS``, ``LEVELS``, ``make_agent``, ``resolve_round``...), de sorte que
``match.py`` et les pages n'ont pas à connaître le moteur.

Il n'y a plus d'agent factice : si un agent ne peut pas être créé, l'erreur
remonte au lieu d'être masquée.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol, TypeAlias, TypedDict

from agents import create_agent
from agents.base_agent import BaseAgent
from game.actions import Action
from game.models import AgentType, Observation, Role
from game.payoff import resolve_payoff_matrix
from game.rules import resolve_round as _resolve_round

# Actions de l'interface, dans l'ordre des indices de ``Action``.
ACTIONS: tuple[str, ...] = ("gauche", "centre", "droite")

# Niveaux de difficulté (section 5 du cahier des charges).
LEVELS: dict[str, str] = {
    "1 — Découverte": "deterministe",
    "2 — Adaptatif": "frequences",
    "3 — Stratégique": "fictitious_play",
    "4 — Optimal": "minimax",
}
LEVEL_HELP: dict[str, tuple[str, str]] = {
    "1 — Découverte": (
        "Déterministe",
        "Comportement fixe et prévisible : apprends à exploiter une stratégie "
        "pure répétée.",
    ),
    "2 — Adaptatif": ("Fréquences", "Observe tes habitudes récentes et les contre."),
    "3 — Stratégique": (
        "Fictitious Play",
        "Meilleure réponse à tes fréquences cumulées depuis le début.",
    ),
    "4 — Optimal": (
        "Minimax",
        "Stratégie mixte optimale (1/3, 1/3, 1/3) : impossible à exploiter.",
    ),
}

# Nom d'agent de l'interface -> type d'agent du moteur.
_AGENT_TYPES: dict[str, AgentType] = {
    "deterministe": AgentType.DETERMINISTIC,
    "frequences": AgentType.FREQUENCY,
    "fictitious_play": AgentType.FICTITIOUS_PLAY,
    "minimax": AgentType.MINIMAX,
}
AGENT_LABELS: dict[str, str] = {
    name: agent_type.label for name, agent_type in _AGENT_TYPES.items()
}

# Rôle de l'interface -> rôle du moteur.
_ROLES: dict[str, Role] = {"attaquant": Role.ATTACKER, "gardien": Role.GOALKEEPER}

_TO_ACTION: dict[str, Action] = {name: Action(i) for i, name in enumerate(ACTIONS)}

# Conservé pour compatibilité : il n'y a plus d'agent factice.
AGENT_SOURCE: dict[str, str] = {name: "réel" for name in _AGENT_TYPES}

# Décision d'un agent : une action, ou une politique (dict ou 3 probabilités).
UiChoice: TypeAlias = str | Mapping[str, float] | Sequence[float]


class UiObservation(TypedDict):
    """Observation transmise aux agents par ``ui.match.Match``."""

    role: str
    round: int
    total_rounds: int
    history: list[tuple[str, str]]


class UiAgent(Protocol):
    """Agent tel que l'interface le manipule (actions sous forme de texte)."""

    def reset(self) -> None:
        """Réinitialise l'agent."""

    def choose_action(self, observation: UiObservation) -> UiChoice:
        """Choisit une action à partir de l'observation seule."""

    def update(
        self,
        observation: UiObservation,
        own_action: str,
        opponent_action: str,
        reward: float,
    ) -> None:
        """Met à jour l'agent après révélation des deux actions."""


def _action(name: object) -> Action:
    """Convertit un nom d'action de l'interface en ``Action``.

    Args:
        name: Nom d'action (``"gauche"``, ``"centre"`` ou ``"droite"``).

    Returns:
        L'action correspondante.

    Raises:
        ValueError: Si le nom ne correspond à aucune action.
    """
    if isinstance(name, str) and name in _TO_ACTION:
        return _TO_ACTION[name]
    raise ValueError(f"Action invalide : {name!r}")


def _observation(observation: UiObservation) -> Observation:
    """Convertit l'observation de l'interface en ``Observation`` du moteur.

    Args:
        observation: Observation de l'interface.

    Returns:
        L'observation du moteur (rôle et numéro de manche).
    """
    return Observation(
        role=_ROLES[observation["role"]], round_number=observation["round"]
    )


class _AgentAdapter:
    """Fait parler un agent du moteur dans le vocabulaire de l'interface."""

    def __init__(self, agent: BaseAgent) -> None:
        self._agent = agent

    @property
    def engine_agent(self) -> BaseAgent:
        """L'agent du moteur sous-jacent."""
        return self._agent

    def reset(self) -> None:
        """Réinitialise l'agent."""
        self._agent.reset()

    def choose_action(self, observation: UiObservation) -> str:
        """Retourne le nom de l'action choisie.

        Args:
            observation: Observation de l'interface.

        Returns:
            ``"gauche"``, ``"centre"`` ou ``"droite"``.
        """
        return ACTIONS[int(self._agent.choose_action(_observation(observation)))]

    def update(
        self,
        observation: UiObservation,
        own_action: str,
        opponent_action: str,
        reward: float,
    ) -> None:
        """Transmet le résultat de la manche à l'agent.

        Args:
            observation: Observation donnée avant la décision.
            own_action: Action jouée par l'agent.
            opponent_action: Action jouée par l'adversaire.
            reward: Gain de l'agent pour cette manche.
        """
        self._agent.update(
            _observation(observation),
            _action(own_action),
            _action(opponent_action),
            float(reward),
        )


def to_agent_type(name: str) -> AgentType:
    """Convertit un nom d'agent de l'interface en type d'agent du moteur.

    Args:
        name: Nom d'agent (``"minimax"``, ``"frequences"``...).

    Returns:
        Le type d'agent correspondant.

    Raises:
        ValueError: Si le nom d'agent est inconnu.
    """
    try:
        return _AGENT_TYPES[name]
    except KeyError:
        raise ValueError(f"Agent inconnu : {name!r}") from None


def make_agent(name: str, seed: int = 0) -> UiAgent:
    """Crée un agent du moteur, adapté au vocabulaire de l'interface.

    Args:
        name: Nom d'agent de l'interface.
        seed: Graine de l'agent.

    Returns:
        L'agent, prêt à jouer.

    Raises:
        ValueError: Si le nom d'agent est inconnu.
    """
    agent = create_agent(to_agent_type(name), resolve_payoff_matrix(None), seed)
    return _AgentAdapter(agent)


def resolve_round(action_attaquant: str, action_gardien: str) -> tuple[str, int, int]:
    """Résout une manche avec les règles du moteur (matrice de référence).

    Args:
        action_attaquant: Zone choisie par l'attaquant.
        action_gardien: Zone choisie par le gardien.

    Returns:
        ``(résultat, gain attaquant, gain gardien)`` : le résultat vaut
        ``"but"`` ou ``"arret"`` et les gains sont entiers (±1).

    Raises:
        ValueError: Si une action est invalide.
    """
    result = _resolve_round(
        1,
        _action(action_attaquant),
        _action(action_gardien),
        resolve_payoff_matrix(None),
    )
    gain = int(result.attacker_reward)
    return ("but" if result.is_goal else "arret"), gain, -gain
