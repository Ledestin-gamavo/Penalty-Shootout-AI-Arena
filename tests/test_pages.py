"""Tests des pages Streamlit avec ``streamlit.testing.v1.AppTest``."""

from streamlit.testing.v1 import AppTest

from game.models import AgentType, MatchConfig
from tests.fakes import fake_run_match
from ui.lab_state import EXPERIMENTS_KEY

TIMEOUT = 60


def _app(page: str) -> AppTest:
    script = f"from ui.{page} import render\nrender()"
    return AppTest.from_string(script, default_timeout=TIMEOUT)


def _count(app: AppTest, element: str) -> int:
    return len(app.get(element))


def test_laboratory_starts_empty() -> None:
    app = _app("laboratory_page").run()
    assert not app.exception
    assert len(app.info) == 1
    assert _count(app, "plotly_chart") == 0


def test_laboratory_runs_an_experiment() -> None:
    app = _app("laboratory_page").run()
    app.number_input(key="lab_rounds").set_value(200)
    app.number_input(key="lab_seed").set_value(7)
    app.button(key="lab_run").click().run()
    assert not app.exception
    assert not app.error
    experiments = app.session_state[EXPERIMENTS_KEY]
    assert len(experiments) == 1
    assert experiments[0].config.num_rounds == 200
    assert experiments[0].config.seed == 7
    assert "graine 7" in app.success[0].value
    assert _count(app, "plotly_chart") == 3  # 2 barres + matrice
    assert not app.info


def test_laboratory_reproducible_from_the_page() -> None:
    first = _app("laboratory_page").run()
    first.number_input(key="lab_rounds").set_value(100)
    first.button(key="lab_run").click().run()
    second = _app("laboratory_page").run()
    second.number_input(key="lab_rounds").set_value(100)
    second.button(key="lab_run").click().run()
    assert (
        first.session_state[EXPERIMENTS_KEY][0].rounds
        == second.session_state[EXPERIMENTS_KEY][0].rounds
    )


def test_analysis_without_experiment_invites_to_run_one() -> None:
    app = _app("analysis_page").run()
    assert not app.exception
    assert len(app.info) == 1


def test_analysis_shows_curves_and_comparison() -> None:
    app = _app("analysis_page")
    app.session_state[EXPERIMENTS_KEY] = [
        fake_run_match(MatchConfig(120, AgentType.MINIMAX, AgentType.FREQUENCY)),
        fake_run_match(MatchConfig(80, AgentType.DETERMINISTIC, AgentType.MINIMAX)),
    ]
    app.run()
    assert not app.exception
    assert _count(app, "plotly_chart") == 2
    assert len(app.dataframe) == 1
    assert len(app.dataframe[0].value) == 2
    app.radio(key="analysis_role").set_value(app.radio(key="analysis_role").options[1])
    app.run()
    assert not app.exception


def test_analysis_clear_button_empties_the_session() -> None:
    app = _app("analysis_page")
    app.session_state[EXPERIMENTS_KEY] = [
        fake_run_match(MatchConfig(50, AgentType.MINIMAX, AgentType.MINIMAX))
    ]
    app.run()
    app.button(key="analysis_clear").click().run()
    assert not app.exception
    assert len(app.session_state[EXPERIMENTS_KEY]) == 0
    assert len(app.info) == 1


def test_benchmark_runs_and_displays_results() -> None:
    app = _app("benchmark_page").run()
    assert not app.exception
    assert len(app.info) == 1
    app.number_input(key="bench_rounds").set_value(40)
    app.number_input(key="bench_seeds").set_value(2)
    app.button(key="bench_run").click().run()
    assert not app.exception
    assert not app.error
    assert "32 matchs joués" in app.success[0].value
    assert _count(app, "plotly_chart") == 4  # performance + matrice (onglets) etc.
    assert len(app.dataframe) == 2


def test_benchmark_rejects_empty_selection() -> None:
    app = _app("benchmark_page").run()
    app.multiselect(key="bench_attackers").set_value([])
    app.button(key="bench_run").click().run()
    assert not app.exception
    assert len(app.error) == 1
