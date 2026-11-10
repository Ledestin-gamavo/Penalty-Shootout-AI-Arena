"""Page « Benchmark » : comparer les agents dans des conditions identiques.

La page lance ``analysis.benchmark.run_benchmark`` (toutes les paires
attaquant/gardien, mêmes graines, même nombre de manches) puis affiche
tableaux, graphiques et exports. Elle ne calcule aucune métrique.

Dépendances autorisées : ``analysis``, ``game.*`` et ``visualization``.
"""

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from analysis.benchmark import (
    BenchmarkResult,
    experiments_to_samples,
    run_benchmark,
    summarize_table,
)
from analysis.export import default_filename, summary_for_display, to_csv_bytes
from game.constants import (
    DEFAULT_BENCHMARK_NUM_SEEDS,
    DEFAULT_NUM_ROUNDS_SIMULATION,
    DEFAULT_SEED,
    MAX_NUM_ROUNDS,
    MIN_NUM_ROUNDS,
)
from game.models import AgentType, Role
from visualization.benchmark_plots import (
    plot_decision_time,
    plot_matchup_heatmap,
    plot_performance_overview,
    plot_strategy_quality,
)

# Clé de session du dernier benchmark.
_RESULT_KEY = "benchmark_result"

# Nombre maximal de graines proposé.
_MAX_SEEDS = 50

# Pas du champ « nombre de manches ».
_ROUNDS_STEP = 100

# Métriques de la carte des confrontations (colonne -> libellé).
_MATCHUP_METRICS = {
    "mean_reward": "Gain moyen de l'attaquant",
    "goal_rate": "Taux de buts",
    "exploitability": "Exploitabilité de l'attaquant",
    "predictability": "Prévisibilité de l'attaquant",
}

_PROTOCOL_HELP = """
- **Conditions identiques** : toutes les paires d'agents jouent le même nombre
  de manches, avec les mêmes graines et la matrice de référence.
- **Barres d'erreur** : intervalle de confiance à 95 % de la moyenne sur les
  graines. Avec une seule graine, il n'est pas estimable (barre nulle).
- **Lire plusieurs indicateurs** : un fort gain moyen contre un adversaire
  faible ne prouve pas qu'une stratégie est bonne. L'exploitabilité mesure ce
  qu'un adversaire optimal pourrait prendre ; la prévisibilité détecte les
  séquences répétées que l'exploitabilité ne voit pas.
- **Gardien** : « taux de buts » désigne les buts concédés.
"""


@dataclass(frozen=True, slots=True)
class _BenchmarkView:
    """Résultat d'un benchmark et tableaux déjà agrégés.

    Attributes:
        result: Résultat brut.
        samples: Une ligne par agent et par expérience.
        by_opponent: Résumé par agent, rôle et adversaire.
        overall: Résumé par agent et rôle, tous adversaires confondus.
    """

    result: BenchmarkResult
    samples: pd.DataFrame
    by_opponent: pd.DataFrame
    overall: pd.DataFrame


def _launch(
    attackers: list[AgentType],
    goalkeepers: list[AgentType],
    num_rounds: int,
    num_seeds: int,
    base_seed: int,
) -> None:
    """Lance le benchmark avec une barre de progression et le conserve.

    Args:
        attackers: Agents jouant l'attaquant.
        goalkeepers: Agents jouant le gardien.
        num_rounds: Nombre de manches par match.
        num_seeds: Nombre de graines par paire.
        base_seed: Première graine.
    """
    bar = st.progress(0.0, text="Benchmark en cours...")

    def on_progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Match {done} / {total}")

    try:
        result = run_benchmark(
            attackers,
            goalkeepers,
            num_rounds=num_rounds,
            num_seeds=num_seeds,
            base_seed=base_seed,
            progress=on_progress,
        )
    except (ValueError, RuntimeError) as error:
        bar.empty()
        st.error(f"Impossible de lancer le benchmark : {error}")
        return
    bar.empty()
    samples = experiments_to_samples(result.experiments)
    st.session_state[_RESULT_KEY] = _BenchmarkView(
        result=result,
        samples=samples,
        by_opponent=summarize_table(samples),
        overall=summarize_table(samples, by_opponent=False),
    )


def _render_form() -> None:
    """Affiche le formulaire de configuration du benchmark."""
    agents = list(AgentType)
    with st.form("benchmark_form"):
        attackers = st.multiselect(
            "Agents attaquants",
            agents,
            default=agents,
            format_func=lambda agent: agent.label,
            key="bench_attackers",
        )
        goalkeepers = st.multiselect(
            "Agents gardiens",
            agents,
            default=agents,
            format_func=lambda agent: agent.label,
            key="bench_goalkeepers",
        )
        rounds_col, seeds_col, seed_col = st.columns(3)
        num_rounds = rounds_col.number_input(
            "Manches par match",
            min_value=MIN_NUM_ROUNDS,
            max_value=MAX_NUM_ROUNDS,
            value=DEFAULT_NUM_ROUNDS_SIMULATION,
            step=_ROUNDS_STEP,
            key="bench_rounds",
        )
        num_seeds = seeds_col.number_input(
            "Graines par paire",
            min_value=1,
            max_value=_MAX_SEEDS,
            value=DEFAULT_BENCHMARK_NUM_SEEDS,
            step=1,
            key="bench_seeds",
        )
        base_seed = seed_col.number_input(
            "Première graine",
            min_value=0,
            value=DEFAULT_SEED,
            step=1,
            key="bench_base_seed",
        )
        submitted = st.form_submit_button("Lancer le benchmark", key="bench_run")
    matches = len(attackers) * len(goalkeepers) * int(num_seeds)
    st.caption(
        f"{matches} matchs de {int(num_rounds)} manches seront joués. "
        "Réduisez les graines ou les manches si le calcul est trop long."
    )
    if not submitted:
        return
    if not attackers or not goalkeepers:
        st.error("Choisissez au moins un attaquant et un gardien.")
        return
    _launch(attackers, goalkeepers, int(num_rounds), int(num_seeds), int(base_seed))


def _render_role_charts(view: _BenchmarkView) -> None:
    """Affiche les graphiques par rôle.

    Args:
        view: Benchmark à afficher.
    """
    role = st.radio(
        "Rôle comparé",
        list(Role),
        format_func=lambda item: item.label,
        horizontal=True,
        key="bench_role",
    )
    if role is None:
        return
    performance, quality, timing = st.tabs(
        ["Performance", "Qualité stratégique", "Temps de décision"]
    )
    with performance:
        st.plotly_chart(plot_performance_overview(view.by_opponent, role))
    with quality:
        st.plotly_chart(plot_strategy_quality(view.by_opponent, role))
    with timing:
        log_scale = st.checkbox("Échelle logarithmique", key="bench_log")
        st.plotly_chart(plot_decision_time(view.by_opponent, role, log_y=log_scale))


def _render_tables(view: _BenchmarkView) -> None:
    """Affiche les tableaux et les boutons d'export.

    Args:
        view: Benchmark à afficher.
    """
    st.markdown("**Tous adversaires confondus** (moyenne ± IC à 95 %)")
    st.dataframe(summary_for_display(view.overall), hide_index=True)
    st.markdown("**Par adversaire**")
    st.dataframe(summary_for_display(view.by_opponent), hide_index=True)
    left, right = st.columns(2)
    left.download_button(
        "Télécharger les échantillons (CSV)",
        to_csv_bytes(view.samples),
        file_name=default_filename("benchmark_echantillons"),
        mime="text/csv",
        key="bench_export_samples",
    )
    right.download_button(
        "Télécharger le résumé (CSV)",
        to_csv_bytes(view.by_opponent),
        file_name=default_filename("benchmark_resume"),
        mime="text/csv",
        key="bench_export_summary",
    )


def _render_view(view: _BenchmarkView) -> None:
    """Affiche le dernier benchmark.

    Args:
        view: Benchmark à afficher.
    """
    result = view.result
    st.success(
        f"{len(result.experiments)} matchs joués : {result.num_rounds} manches, "
        f"{result.num_seeds} graines à partir de {result.base_seed}, matrice "
        "de référence."
    )
    _render_role_charts(view)
    st.subheader("Confrontations")
    metric = st.selectbox(
        "Indicateur de l'attaquant",
        list(_MATCHUP_METRICS),
        format_func=lambda key: _MATCHUP_METRICS[key],
        key="bench_matchup_metric",
    )
    if metric is not None:
        st.plotly_chart(plot_matchup_heatmap(view.by_opponent, metric))
    st.subheader("Tableaux et export")
    _render_tables(view)
    with st.expander("Protocole et limites"):
        st.markdown(_PROTOCOL_HELP)


def render() -> None:
    """Affiche la page « Benchmark »."""
    st.header("Benchmark")
    st.write(
        "Comparez les agents sur toutes les paires attaquant/gardien, avec "
        "les mêmes graines et le même nombre de manches."
    )
    _render_form()
    view = st.session_state.get(_RESULT_KEY)
    if view is None:
        st.info("Aucun benchmark pour l'instant : lancez-en un ci-dessus.")
        return
    _render_view(view)


# Nom utilisé par les pages provisoires du dépôt.
render_benchmark_page = render
