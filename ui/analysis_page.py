"""Page « Analyse » : évolution et comparaison des expériences du laboratoire.

Dépendances autorisées : ``analysis``, ``game.*``, ``theory.*``,
``visualization`` et Streamlit.
"""

import streamlit as st

from analysis.export import experiments_overview
from game.models import Experiment, Role
from game.payoff import resolve_payoff_matrix
from theory.equilibrium import solve_equilibrium
from ui.lab_state import clear_experiments, get_experiments
from visualization.strategy_plots import (
    plot_frequency_evolution,
    plot_reward_evolution,
)


def _experiment_label(index: int) -> str:
    """Retourne le numéro affiché d'une expérience (à partir de 1).

    Args:
        index: Position de l'expérience dans la liste (à partir de 0).

    Returns:
        Le libellé court de l'expérience.
    """
    return f"Expérience {index + 1}"


def _describe(experiment: Experiment) -> str:
    """Décrit une expérience en une ligne.

    Args:
        experiment: Expérience décrite.

    Returns:
        Agents, nombre de manches et graine.
    """
    config = experiment.config
    return (
        f"{experiment.attacker_stats.agent_type.label} contre "
        f"{experiment.goalkeeper_stats.agent_type.label}, "
        f"{config.num_rounds} manches, graine {config.seed}"
    )


def _render_evolution(experiment: Experiment) -> None:
    """Affiche les courbes d'évolution d'un rôle.

    Args:
        experiment: Expérience analysée.
    """
    role = st.radio(
        "Rôle étudié",
        list(Role),
        format_func=lambda item: item.label,
        horizontal=True,
        key="analysis_role",
    )
    if role is None:
        return
    matrix = resolve_payoff_matrix(experiment.config.payoff_matrix)
    equilibrium = solve_equilibrium(matrix)
    value = equilibrium.value if role is Role.ATTACKER else -equilibrium.value
    stats = (
        experiment.attacker_stats
        if role is Role.ATTACKER
        else experiment.goalkeeper_stats
    )
    cols = st.columns(3)
    cols[0].metric("Agent", stats.agent_type.label)
    cols[1].metric("Écart à l'équilibre", f"{stats.equilibrium_gap:.3f}")
    cols[2].metric("Exploitabilité", f"{stats.exploitability:.3f}")
    st.plotly_chart(
        plot_frequency_evolution(experiment.rounds, role, equilibrium.strategy_of(role))
    )
    st.plotly_chart(plot_reward_evolution(experiment.rounds, role, value))
    st.caption(
        "Les pointillés indiquent l'équilibre : fréquences optimales et "
        "valeur du jeu. Un agent qui s'adapte voit ses fréquences bouger "
        "au fil des manches ; un agent qui joue l'équilibre reste proche "
        "des pointillés."
    )


def render() -> None:
    """Affiche la page « Analyse »."""
    st.header("Analyse")
    experiments = get_experiments()
    if not experiments:
        st.info(
            "Aucune expérience à analyser : lancez d'abord un duel dans la "
            "page « Laboratoire »."
        )
        return
    index = st.selectbox(
        "Expérience à analyser",
        list(range(len(experiments))),
        index=len(experiments) - 1,
        format_func=lambda i: f"{_experiment_label(i)} — {_describe(experiments[i])}",
        key="analysis_experiment",
    )
    if index is not None:
        _render_evolution(experiments[index])
    st.subheader("Comparer les expériences")
    st.dataframe(experiments_overview(experiments), hide_index=True)
    if st.button("Effacer les expériences", key="analysis_clear"):
        clear_experiments()
        st.rerun()


# Nom utilisé par les pages provisoires du dépôt.
render_analysis_page = render
