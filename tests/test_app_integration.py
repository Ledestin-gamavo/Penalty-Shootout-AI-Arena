"""Tests d'intégration de l'application complète (``app.py``) avec le vrai moteur."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from game.models import AgentType, Experiment
from ui import compat
from ui.lab_state import EXPERIMENTS_KEY
from ui.match import (
    ATTAQUANT,
    GARDIEN,
    MODE_HUMAIN_IA,
    Match,
    MatchConfig,
    autre_role,
)
from ui.state import HUMAN_MATCH, SIM_MATCH

TIMEOUT = 90
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["Jouer", "Laboratoire", "Analyse", "Benchmark"]


def _app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=TIMEOUT).run()


def _go(app: AppTest, page: str) -> AppTest:
    app.sidebar.radio[0].set_value(page).run()
    return app


def _button(app: AppTest, label: str):  # type: ignore[no-untyped-def]
    return next(b for b in app.button if b.label == label)


def test_app_starts_without_error_or_fake_agent_warning() -> None:
    app = _app()
    assert not app.exception
    assert not app.sidebar.warning


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page: str) -> None:
    app = _go(_app(), page)
    assert not app.exception
    assert not app.error


def test_compat_uses_the_real_agents() -> None:
    from agents import (
        DeterministicAgent,
        FictitiousPlayAgent,
        FrequencyAgent,
        MinimaxAgent,
    )

    expected = {
        "deterministe": DeterministicAgent,
        "frequences": FrequencyAgent,
        "fictitious_play": FictitiousPlayAgent,
        "minimax": MinimaxAgent,
    }
    for name, agent_class in expected.items():
        adapter = compat.make_agent(name, seed=3)
        assert isinstance(adapter._agent, agent_class)
        observation = {"role": ATTAQUANT, "round": 1}
        action = adapter.choose_action(observation)
        assert action in compat.ACTIONS
        adapter.update(observation, action, "gauche", 0)
    assert compat.AGENT_LABELS == {
        name: AgentType(value).label
        for name, value in zip(
            expected, ("deterministic", "frequency", "fictitious_play", "minimax")
        )
    }


def test_compat_rejects_unknown_agent_and_invalid_action() -> None:
    with pytest.raises(ValueError):
        compat.make_agent("inconnu")
    with pytest.raises(ValueError):
        compat.resolve_round("haut", "gauche")


@pytest.mark.parametrize("attaquant", compat.ACTIONS)
@pytest.mark.parametrize("gardien", compat.ACTIONS)
def test_resolve_round_follows_the_game_rules(attaquant: str, gardien: str) -> None:
    result, gain_attaquant, gain_gardien = compat.resolve_round(attaquant, gardien)
    assert (result, gain_attaquant, gain_gardien) == (
        ("arret", -1, 1) if attaquant == gardien else ("but", 1, -1)
    )
    assert isinstance(gain_attaquant, int)


@pytest.mark.parametrize("level", list(compat.LEVELS))
@pytest.mark.parametrize("role", [ATTAQUANT, GARDIEN])
def test_human_game_plays_to_the_end_against_every_level(level: str, role: str) -> None:
    agents = {ATTAQUANT: None, GARDIEN: None}
    agents[autre_role(role)] = compat.LEVELS[level]
    config = MatchConfig(MODE_HUMAIN_IA, 5, agents, 42, role_humain=role)
    app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    app.session_state[HUMAN_MATCH] = Match(config)
    app.run()
    for action in ("gauche", "centre", "droite", "gauche", "centre"):
        app.button(key=f"zone_{action}").click().run()
        assert not app.exception
    match = app.session_state[HUMAN_MATCH]
    assert match.finished and len(match.history) == 5
    assert all(r.gain_attaquant == -r.gain_gardien for r in match.history)
    assert any(s.value == "Fin de partie" for s in app.subheader)


def test_human_game_is_reproducible_for_a_given_seed() -> None:
    def play() -> list:
        config = MatchConfig(
            MODE_HUMAIN_IA,
            8,
            {ATTAQUANT: None, GARDIEN: "fictitious_play"},
            7,
            role_humain=ATTAQUANT,
        )
        match = Match(config)
        for action in ("gauche", "centre", "droite", "centre") * 2:
            match.play_round(action)
        return match.history

    assert play() == play()


def test_human_game_can_be_configured_from_the_form() -> None:
    app = _go(_app(), "Jouer")
    next(s for s in app.selectbox if s.label == "Niveau de l'adversaire").set_value(
        "4 — Optimal"
    )
    _button(app, "Lancer la partie").click().run()
    assert not app.exception
    match = app.session_state[HUMAN_MATCH]
    assert match.config.agents[GARDIEN] == "minimax" or (
        match.config.agents[ATTAQUANT] == "minimax"
    )


def _simulate_from_play_page(rounds: int, seed: int) -> Experiment:
    app = _go(_app(), "Jouer")
    next(r for r in app.radio if r.label == "Mode de jeu").set_value("IA contre IA")
    app.run()
    next(n for n in app.number_input if n.label == "Nombre de manches").set_value(
        rounds
    )
    next(n for n in app.number_input if "Graine" in n.label).set_value(seed)
    _button(app, "Lancer la simulation").click().run()
    assert not app.exception
    assert not app.error
    experiment, _ = app.session_state[SIM_MATCH]
    return experiment  # type: ignore[no-any-return]


def test_ai_vs_ai_simulation_from_the_play_page() -> None:
    experiment = _simulate_from_play_page(300, 42)
    assert isinstance(experiment, Experiment)
    assert experiment.config.num_rounds == 300
    assert len(experiment.rounds) == 300


def test_play_page_and_laboratory_give_the_same_rounds_for_a_seed() -> None:
    """Une seule boucle de simulation : même graine, mêmes manches."""
    from_play = _simulate_from_play_page(300, 42)  # déterministe vs Minimax

    app = _go(_app(), "Laboratoire")
    app.selectbox(key="lab_attacker").set_value(AgentType.DETERMINISTIC)
    app.selectbox(key="lab_goalkeeper").set_value(AgentType.MINIMAX)
    app.number_input(key="lab_rounds").set_value(300)
    app.number_input(key="lab_seed").set_value(42)
    app.button(key="lab_run").click().run()
    assert not app.exception
    from_lab = app.session_state[EXPERIMENTS_KEY][0]
    assert from_play.rounds == from_lab.rounds


def test_laboratory_and_analysis_work_inside_the_full_app() -> None:
    app = _go(_app(), "Laboratoire")
    app.number_input(key="lab_rounds").set_value(200)
    app.button(key="lab_run").click().run()
    assert not app.exception
    assert len(app.session_state[EXPERIMENTS_KEY]) == 1
    _go(app, "Analyse")
    assert not app.exception
    assert len(app.get("plotly_chart")) == 2
