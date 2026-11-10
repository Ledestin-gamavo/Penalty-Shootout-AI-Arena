"""Faux moteur de match, indépendant de ``agents/`` et ``game.simulation``.

Permet de tester le benchmark, les tableaux et les figures avant la livraison
des agents. Les politiques sont volontairement simplistes : déterministe
(toujours gauche), cyclique (gauche, centre, droite...) ou uniforme.
"""

from datetime import UTC, datetime

import numpy as np

from game.actions import Action
from game.models import AgentType, Experiment, MatchConfig, Role, RoundResult
from game.payoff import attacker_reward, resolve_payoff_matrix
from theory.metrics import compute_agent_stats

# Temps de décision fictif, en secondes.
FAKE_DECISION_TIME = 1e-5


def _fake_actions(
    agent: AgentType | None, rng: np.random.Generator, count: int
) -> list[Action]:
    """Génère les actions d'un agent factice.

    Args:
        agent: Type de l'agent (jamais ``None`` dans un match IA contre IA).
        rng: Générateur aléatoire propre à l'agent.
        count: Nombre de manches.

    Returns:
        La suite d'actions.
    """
    if agent is AgentType.DETERMINISTIC:
        return [Action.LEFT] * count
    if agent is AgentType.FREQUENCY:
        return [Action(i % 3) for i in range(count)]
    return [Action(int(i)) for i in rng.integers(0, 3, count)]


def fake_run_match(config: MatchConfig) -> Experiment:
    """Exécute un faux match IA contre IA, reproductible par graine.

    Args:
        config: Configuration du match.

    Returns:
        Une expérience complète, avec de vraies statistiques.
    """
    matrix = resolve_payoff_matrix(config.payoff_matrix)
    seed_a, seed_g = np.random.SeedSequence(config.seed).spawn(2)
    attacker_actions = _fake_actions(
        config.attacker_agent, np.random.default_rng(seed_a), config.num_rounds
    )
    goalkeeper_actions = _fake_actions(
        config.goalkeeper_agent, np.random.default_rng(seed_g), config.num_rounds
    )
    rounds = tuple(
        RoundResult(n, a, g, attacker_reward(a, g, matrix))
        for n, (a, g) in enumerate(zip(attacker_actions, goalkeeper_actions), 1)
    )
    assert config.attacker_agent is not None  # noqa: S101 - aide au typage
    assert config.goalkeeper_agent is not None  # noqa: S101 - aide au typage
    return Experiment(
        config=config,
        created_at=datetime.now(UTC),
        rounds=rounds,
        attacker_stats=compute_agent_stats(
            config.attacker_agent, Role.ATTACKER, rounds, FAKE_DECISION_TIME, matrix
        ),
        goalkeeper_stats=compute_agent_stats(
            config.goalkeeper_agent, Role.GOALKEEPER, rounds, FAKE_DECISION_TIME, matrix
        ),
    )
