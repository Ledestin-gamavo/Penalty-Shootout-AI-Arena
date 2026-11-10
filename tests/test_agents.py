"""Tests des agents : conformité au contrat d'API."""
import inspect

import pytest

from agents import (
    BaseAgent,
    DeterministicAgent,
    FictitiousPlayAgent,
    FrequencyAgent,
    MinimaxAgent,
    create_agent,
)
from game.actions import Action
from game.models import AgentType, Observation, Role
from game.payoff import build_reference_matrix

MATRIX = build_reference_matrix()


def _obs(role: Role, n: int = 1) -> Observation:
    return Observation(role=role, round_number=n)


def _play(agent: BaseAgent, role: Role, opp: Action, rounds: int) -> None:
    for n in range(1, rounds + 1):
        obs = _obs(role, n)
        own = agent.choose_action(obs)
        agent.update(obs, own, opp, 0.0)


@pytest.mark.parametrize(
    ("agent_type", "agent_class"),
    [
        (AgentType.DETERMINISTIC, DeterministicAgent),
        (AgentType.FREQUENCY, FrequencyAgent),
        (AgentType.MINIMAX, MinimaxAgent),
        (AgentType.FICTITIOUS_PLAY, FictitiousPlayAgent),
    ],
)
def test_create_agent_tous_les_types(
    agent_type: AgentType, agent_class: type[BaseAgent]
) -> None:
    agent = create_agent(agent_type, MATRIX, 0)
    assert isinstance(agent, agent_class)
    assert isinstance(agent.choose_action(_obs(Role.ATTACKER)), Action)


def test_create_agent_accepte_la_valeur_chaine() -> None:
    assert create_agent("frequency", MATRIX, 0).__class__.__name__ == "FrequencyAgent"  # type: ignore[arg-type]


def test_create_agent_type_inconnu() -> None:
    with pytest.raises(ValueError):
        create_agent("inconnu", MATRIX, 0)  # type: ignore[arg-type]


def test_signature_choose_action_observation_seule() -> None:
    params = list(inspect.signature(BaseAgent.choose_action).parameters)
    assert params == ["self", "observation"]


def test_role_lu_a_chaque_decision() -> None:
    agent = create_agent(AgentType.FICTITIOUS_PLAY, MATRIX, 0)
    _play(agent, Role.ATTACKER, Action.LEFT, 5)
    assert agent.choose_action(_obs(Role.ATTACKER)) == Action.CENTER
    assert agent.choose_action(_obs(Role.GOALKEEPER)) == Action.LEFT


@pytest.mark.parametrize("agent_type", [AgentType.FREQUENCY, AgentType.FICTITIOUS_PLAY])
def test_meilleure_reponse_contre_toujours_gauche(agent_type: AgentType) -> None:
    att = create_agent(agent_type, MATRIX, 0)
    gk = create_agent(agent_type, MATRIX, 0)
    _play(att, Role.ATTACKER, Action.LEFT, 10)
    _play(gk, Role.GOALKEEPER, Action.LEFT, 10)
    assert att.choose_action(_obs(Role.ATTACKER, 11)) == Action.CENTER
    assert gk.choose_action(_obs(Role.GOALKEEPER, 11)) == Action.LEFT


def test_croyance_uniforme_donne_left() -> None:
    for agent_type in (AgentType.FREQUENCY, AgentType.FICTITIOUS_PLAY):
        agent = create_agent(agent_type, MATRIX, 0)
        assert agent.choose_action(_obs(Role.ATTACKER)) == Action.LEFT


def test_deterministe_rejoue_la_sequence() -> None:
    agent = DeterministicAgent(MATRIX, 0, pattern=(Action.LEFT, Action.RIGHT))
    seen = []
    for n in range(1, 5):
        obs = _obs(Role.ATTACKER, n)
        action = agent.choose_action(obs)
        seen.append(action)
        agent.update(obs, action, Action.CENTER, 0.0)
    assert seen == [Action.LEFT, Action.RIGHT, Action.LEFT, Action.RIGHT]


def test_deterministe_avance_uniquement_apres_update() -> None:
    agent = DeterministicAgent(MATRIX, 0, pattern=(Action.LEFT, Action.RIGHT))
    obs = _obs(Role.ATTACKER)

    assert agent.choose_action(obs) == Action.LEFT
    assert agent.choose_action(obs) == Action.LEFT
    agent.update(obs, Action.LEFT, Action.CENTER, 0.0)
    assert agent.choose_action(obs) == Action.RIGHT


def test_frequence_fenetre_glissante() -> None:
    agent = FrequencyAgent(MATRIX, 0, window=3)
    _play(agent, Role.ATTACKER, Action.LEFT, 10)
    _play(agent, Role.ATTACKER, Action.RIGHT, 3)       # la fenêtre ne garde que ça
    assert agent.estimated_opponent()[Action.RIGHT] > 0.7


def test_frequence_reset_efface_la_fenetre() -> None:
    agent = FrequencyAgent(MATRIX, 0, window=3)
    _play(agent, Role.ATTACKER, Action.RIGHT, 3)

    agent.reset()

    assert agent.t == 0
    assert agent.estimated_opponent() == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_fictitious_play_conserve_tout_historique_adverse() -> None:
    agent = FictitiousPlayAgent(MATRIX, 0, prior=1.0)
    for n in range(1, 13):
        obs = _obs(Role.ATTACKER, n)
        agent.update(obs, Action.CENTER, Action.LEFT, 0.0)
    for n in range(13, 16):
        obs = _obs(Role.ATTACKER, n)
        agent.update(obs, Action.CENTER, Action.RIGHT, 0.0)

    assert agent.empirical_opponent() == pytest.approx((13 / 18, 1 / 18, 4 / 18))
    assert agent.t == 15


def test_meme_graine_meme_historique_meme_action() -> None:
    a1 = create_agent(AgentType.MINIMAX, MATRIX, 7)
    a2 = create_agent(AgentType.MINIMAX, MATRIX, 7)
    s1 = [a1.choose_action(_obs(Role.GOALKEEPER, n)) for n in range(1, 30)]
    s2 = [a2.choose_action(_obs(Role.GOALKEEPER, n)) for n in range(1, 30)]
    assert s1 == s2


def test_reset_restaure_etat_et_sequence_aleatoire() -> None:
    agent = create_agent(AgentType.MINIMAX, MATRIX, 3)
    first = [agent.choose_action(_obs(Role.ATTACKER, n)) for n in range(1, 20)]
    agent.reset()
    again = [agent.choose_action(_obs(Role.ATTACKER, n)) for n in range(1, 20)]
    assert first == again
    fp = create_agent(AgentType.FICTITIOUS_PLAY, MATRIX, 0)
    _play(fp, Role.ATTACKER, Action.RIGHT, 5)
    fp.reset()
    assert fp.t == 0 and fp.choose_action(_obs(Role.ATTACKER)) == Action.LEFT


def test_minimax_utilise_lequilibre_de_reference() -> None:
    from agents import MinimaxAgent

    agent = MinimaxAgent(MATRIX, 0)
    assert agent.game_value == pytest.approx(1 / 3)
    draws = [agent.choose_action(_obs(Role.ATTACKER, n)) for n in range(1, 301)]
    for action in Action:
        assert 60 < draws.count(action) < 140


def test_update_invalide() -> None:
    agent = create_agent(AgentType.FREQUENCY, MATRIX, 0)
    with pytest.raises(ValueError):
        agent.update(_obs(Role.ATTACKER), Action.LEFT, 5, 0.0)  # type: ignore[arg-type]


def test_matrice_invalide() -> None:
    with pytest.raises(ValueError):
        create_agent(AgentType.MINIMAX, ((1.0, 2.0),), 0)