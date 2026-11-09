"""Orchestration des manches et des matchs.

Ce module impose l'ordre qui garantit la simultanéité des décisions :
1. chaque agent choisit son action à partir d'une ``Observation`` seule ;
2. la manche est résolue une fois les deux actions fixées ;
3. seulement ensuite, chaque agent reçoit le résultat via ``update``.

Aucun agent n'a accès à l'action courante de l'adversaire avant l'étape 3.
Le calcul des métriques est délégué à ``theory.metrics``.

Dépendances autorisées : ``agents``, ``game.*`` et ``theory.*``.
"""

from datetime import UTC, datetime
from time import perf_counter
from numpy.random import SeedSequence
from agents import create_agent
from agents.base_agent import BaseAgent
from game.actions import Action
from game.models import (
    Experiment,
    MatchConfig,
    Observation,
    PayoffMatrix,
    Role,
    RoundResult,
)
from game.payoff import resolve_payoff_matrix
from game.rules import resolve_round, validate_action, validate_config
from theory.metrics import compute_agent_stats


def _timed_choice(agent: BaseAgent, observation: Observation) -> tuple[Action, float]:
    """Fait décider un agent et mesure son temps de décision.

    Args:
        agent: Agent qui doit choisir.
        observation: Information autorisée donnée à l'agent.

    Returns:
        Un couple ``(action choisie, temps de décision en secondes)``.
    """
    start = perf_counter()
    action = agent.choose_action(observation)
    return action, perf_counter() - start


def _update_agent(agent: BaseAgent, observation: Observation, result: RoundResult) -> None:
    """Transmet à un agent le résultat de la manche terminée.

    Args:
        agent: Agent à mettre à jour.
        observation: Observation donnée à l'agent avant sa décision.
        result: Résultat de la manche, actions révélées.
    """
    role = observation.role
    agent.update(
        observation,
        result.action_of(role),
        result.action_of(role.opponent),
        result.reward_for(role),
    )


def _play_ai_round(attacker: BaseAgent, goalkeeper: BaseAgent, round_number: int, payoff_matrix: PayoffMatrix,) -> tuple[RoundResult, float, float]:
    """Joue une manche IA contre IA avec décisions simultanées.

    Args:
        attacker: Agent attaquant.
        goalkeeper: Agent gardien.
        round_number: Numéro de la manche, à partir de 1.
        payoff_matrix: Matrice des gains de l'attaquant.

    Returns:
        Un couple ``(résultat, temps de décision de l'attaquant, temps de
        décision du gardien)``, les temps étant exprimés en secondes.
    """
    attacker_obs = Observation(role=Role.ATTACKER, round_number=round_number)
    goalkeeper_obs = Observation(role=Role.GOALKEEPER, round_number=round_number)
    attacker_action, attacker_time = _timed_choice(attacker, attacker_obs)
    goalkeeper_action, goalkeeper_time = _timed_choice(goalkeeper, goalkeeper_obs)
    result = resolve_round(
        round_number, attacker_action, goalkeeper_action, payoff_matrix
    )
    _update_agent(attacker, attacker_obs, result)
    _update_agent(goalkeeper, goalkeeper_obs, result)
    return result, attacker_time, goalkeeper_time


def run_match(config: MatchConfig) -> Experiment:
    """Exécute un match IA contre IA complet.

    Deux graines indépendantes, une par agent, sont dérivées de
    ``config.seed`` avec ``SeedSequence``. Une même configuration produit
    donc les mêmes manches.

    Args:
        config: Configuration du match. Les deux joueurs doivent être des
            agents.

    Returns:
        L'expérience complète : configuration, manches et statistiques.

    Raises:
        ValueError: Si un des joueurs est humain ou si la matrice des gains
            personnalisée est invalide.
    """
    attacker_type = config.attacker_agent
    goalkeeper_type = config.goalkeeper_agent
    if attacker_type is None or goalkeeper_type is None:
        raise ValueError(
            "run_match exige deux agents : un joueur humain ne peut pas être "
            "simulé."
        )
    validate_config(config)
    matrix = resolve_payoff_matrix(config.payoff_matrix)
    attacker_seed, goalkeeper_seed = (
        int(seed) for seed in SeedSequence(config.seed).generate_state(2)
    )
    attacker = create_agent(attacker_type, matrix, attacker_seed)
    goalkeeper = create_agent(goalkeeper_type, matrix, goalkeeper_seed)

    rounds: list[RoundResult] = []
    attacker_total_time = 0.0
    goalkeeper_total_time = 0.0
    for round_number in range(1, config.num_rounds + 1):
        result, attacker_time, goalkeeper_time = _play_ai_round(
            attacker, goalkeeper, round_number, matrix
        )
        rounds.append(result)
        attacker_total_time += attacker_time
        goalkeeper_total_time += goalkeeper_time

    results = tuple(rounds)
    return Experiment(
        config=config,
        created_at=datetime.now(UTC),
        rounds=results,
        attacker_stats=compute_agent_stats(
            attacker_type,
            Role.ATTACKER,
            results,
            attacker_total_time / config.num_rounds,
            matrix,
        ),
        goalkeeper_stats=compute_agent_stats(
            goalkeeper_type,
            Role.GOALKEEPER,
            results,
            goalkeeper_total_time / config.num_rounds,
            matrix,
        ),
    )


def play_human_round(agent: BaseAgent, human_role: Role, human_action: Action, round_number: int, payoff_matrix: PayoffMatrix,) -> RoundResult:
    """Joue une manche humain contre IA.

    L'action humaine est validée avant la décision de l'agent. L'agent ne
    reçoit qu'une ``Observation`` ; l'action humaine ne lui est transmise
    qu'après résolution, via ``update``. L'interface ne doit afficher
    l'action de l'agent qu'après l'appel à cette fonction.

    Args:
        agent: Agent adverse, qui tient le rôle opposé à celui de l'humain.
        human_role: Rôle tenu par l'humain.
        human_action: Zone choisie par l'humain.
        round_number: Numéro de la manche, à partir de 1.
        payoff_matrix: Matrice des gains de l'attaquant, déjà validée par
            l'appelant.

    Returns:
        Le résultat de la manche.

    Raises:
        ValueError: Si l'action humaine ou le numéro de manche est invalide.
    """
    human_action = validate_action(human_action)
    agent_role = human_role.opponent
    observation = Observation(role=agent_role, round_number=round_number)
    agent_action = agent.choose_action(observation)
    actions = {human_role: human_action, agent_role: agent_action}
    result = resolve_round(
        round_number,
        actions[Role.ATTACKER],
        actions[Role.GOALKEEPER],
        payoff_matrix,
    )
    _update_agent(agent, observation, result)
    return result