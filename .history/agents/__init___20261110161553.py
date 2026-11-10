"""Agents du projet et fabrique unique `create_agent` (contrat d'API, §7.7)."""
from agents.base_agent import BaseAgent
from agents.deterministic_agent import DeterministicAgent
from agents.fictitious_play_agent import FictitiousPlayAgent
from agents.frequency_agent import FrequencyAgent
from agents.minimax_agent import MinimaxAgent
from game.models import AgentType, PayoffMatrix

_AGENT_CLASSES: dict[AgentType, type[BaseAgent]] = {
    AgentType.DETERMINISTIC: DeterministicAgent,
    AgentType.FREQUENCY: FrequencyAgent,
    AgentType.MINIMAX: MinimaxAgent,
    AgentType.FICTITIOUS_PLAY: FictitiousPlayAgent,
}


def create_agent(
    agent_type: AgentType, payoff_matrix: PayoffMatrix, seed: int
) -> BaseAgent:
    """Fabrique unique d'agents.

    Args:
        agent_type: Type d'agent à créer.
        payoff_matrix: Matrice des gains de l'attaquant.
        seed: Graine du générateur aléatoire de l'agent.

    Returns:
        Un agent prêt à jouer ; son rôle est fourni par chaque observation.

    Raises:
        ValueError: Si le type d'agent ou la matrice est invalide.
    """
    try:
        agent_class = _AGENT_CLASSES[AgentType(agent_type)]
    except (KeyError, ValueError):
        raise ValueError(f"type d'agent inconnu : {agent_type!r}") from None
    return agent_class(payoff_matrix, seed)


__all__ = [
    "BaseAgent",
    "DeterministicAgent",
    "FictitiousPlayAgent",
    "FrequencyAgent",
    "MinimaxAgent",
    "create_agent",
]