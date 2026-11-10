"""Page « Laboratoire » : lancer et lire une expérience IA contre IA.

La page saisit une configuration, appelle ``game.simulation.run_match`` et
affiche métriques, stratégies, matrice et historique. Elle ne calcule aucune
métrique : tout vient de ``Experiment`` et de ``theory``.

Dépendances autorisées : ``analysis``, ``game.*``, ``theory.*``,
``visualization`` et Streamlit.
"""

import streamlit as st

from analysis.benchmark import experiments_to_samples
from analysis.export import (
    agent_stats_dataframe,
    default_filename,
    rounds_to_dataframe,
    to_csv_bytes,
)
from game.actions import Action
from game.constants import (
    DEFAULT_NUM_ROUNDS_SIMULATION,
    DEFAULT_SEED,
    MAX_NUM_ROUNDS,
    MIN_NUM_ROUNDS,
)
from game.models import AgentType, Experiment, MatchConfig, Role
from game.payoff import resolve_payoff_matrix
from game.simulation import run_match
from theory.benchmark_metrics import best_response_cells
from theory.equilibrium import solve_equilibrium
from ui.lab_state import HISTORY_DISPLAY_LIMIT, add_experiment, get_experiments
from visualization.strategy_plots import plot_payoff_heatmap, plot_strategy_bars

# Agents proposés par défaut dans le formulaire.
_DEFAULT_ATTACKER = AgentType.FICTITIOUS_PLAY
_DEFAULT_GOALKEEPER = AgentType.MINIMAX

# Pas du champ « nombre de manches ».
_ROUNDS_STEP = 100

_METRICS_HELP = """
- **Taux de buts / d'arrêts** : part des manches terminées par un but ou un
  arrêt. Pour le gardien, les buts sont des buts *concédés*.
- **Gain moyen** : gain moyen par manche du point de vue de l'agent (+1 par
  but pour l'attaquant, -1 par arrêt).
- **Écart à l'équilibre** : distance entre les fréquences jouées par l'agent
  et la stratégie optimale (0 = identiques, 1 = totalement différentes).
- **Exploitabilité** : ce qu'un adversaire optimal gagnerait de plus contre
  les fréquences de l'agent. 0 = stratégie optimale.
- **Prévisibilité (ordre 1)** : part des actions devinées par un prédicteur
  qui apprend l'enchaînement des coups. Environ 33 % pour un jeu imprévisible,
  100 % pour une séquence répétée.
- L'exploitabilité ne voit pas l'ordre des coups : un agent qui joue
  gauche, centre, droite en boucle a une exploitabilité nulle mais une
  prévisibilité de 100 %. Lisez toujours ces indicateurs ensemble.
"""


def _render_form() -> None:
    """Affiche le formulaire de configuration et lance l'expérience."""
    agents = list(AgentType)
    with st.form("lab_form"):
        left, right = st.columns(2)
        attacker = left.selectbox(
            "Agent attaquant",
            agents,
            index=agents.index(_DEFAULT_ATTACKER),
            format_func=lambda agent: agent.label,
            key="lab_attacker",
        )
        goalkeeper = right.selectbox(
            "Agent gardien",
            agents,
            index=agents.index(_DEFAULT_GOALKEEPER),
            format_func=lambda agent: agent.label,
            key="lab_goalkeeper",
        )
        rounds_col, seed_col = st.columns(2)
        num_rounds = rounds_col.number_input(
            "Nombre de manches",
            min_value=MIN_NUM_ROUNDS,
            max_value=MAX_NUM_ROUNDS,
            value=DEFAULT_NUM_ROUNDS_SIMULATION,
            step=_ROUNDS_STEP,
            key="lab_rounds",
        )
        seed = seed_col.number_input(
            "Graine (reproductibilité)",
            min_value=0,
            value=DEFAULT_SEED,
            step=1,
            key="lab_seed",
        )
        submitted = st.form_submit_button("Lancer l'expérience", key="lab_run")
    if not submitted or attacker is None or goalkeeper is None:
        return
    try:
        config = MatchConfig(
            num_rounds=int(num_rounds),
            attacker_agent=attacker,
            goalkeeper_agent=goalkeeper,
            seed=int(seed),
        )
        with st.spinner("Simulation en cours..."):
            experiment = run_match(config)
    except (ValueError, RuntimeError) as error:
        st.error(f"Impossible de lancer l'expérience : {error}")
        return
    add_experiment(experiment)


def _render_metrics(experiment: Experiment) -> None:
    """Affiche les indicateurs clés et le tableau complet.

    Args:
        experiment: Expérience affichée.
    """
    attacker = experiment.attacker_stats
    goalkeeper = experiment.goalkeeper_stats
    cols = st.columns(4)
    cols[0].metric("Gain moyen de l'attaquant", f"{attacker.mean_reward:+.3f}")
    cols[1].metric("Taux de buts", f"{attacker.goal_rate:.1%}")
    cols[2].metric("Exploitabilité attaquant", f"{attacker.exploitability:.3f}")
    cols[3].metric("Exploitabilité gardien", f"{goalkeeper.exploitability:.3f}")
    st.dataframe(agent_stats_dataframe(experiment), hide_index=True)
    with st.expander("Comment lire ces indicateurs ?"):
        st.markdown(_METRICS_HELP)


def _render_strategies(experiment: Experiment) -> None:
    """Affiche les stratégies empiriques face à l'équilibre.

    Args:
        experiment: Expérience affichée.
    """
    matrix = resolve_payoff_matrix(experiment.config.payoff_matrix)
    equilibrium = solve_equilibrium(matrix)
    left, right = st.columns(2)
    for column, role, stats in (
        (left, Role.ATTACKER, experiment.attacker_stats),
        (right, Role.GOALKEEPER, experiment.goalkeeper_stats),
    ):
        figure = plot_strategy_bars(
            stats.empirical_strategy,
            equilibrium.strategy_of(role),
            title=f"{role.label} — {stats.agent_type.label}",
            strategy_name="Fréquences jouées",
        )
        column.plotly_chart(figure)


def _render_matrix(experiment: Experiment) -> None:
    """Affiche la matrice des gains et ses meilleures réponses.

    Args:
        experiment: Expérience affichée.
    """
    matrix = resolve_payoff_matrix(experiment.config.payoff_matrix)
    choice = st.radio(
        "Entourer les meilleures réponses",
        ["Aucune", Role.ATTACKER.label, Role.GOALKEEPER.label],
        horizontal=True,
        key="lab_best_response",
    )
    cells: tuple[tuple[Action, Action], ...] = ()
    if choice == Role.ATTACKER.label:
        cells = best_response_cells(matrix, Role.ATTACKER)
    elif choice == Role.GOALKEEPER.label:
        cells = best_response_cells(matrix, Role.GOALKEEPER)
    st.plotly_chart(plot_payoff_heatmap(matrix, cells))
    st.caption(
        "Les cases entourées indiquent, pour chaque action de l'adversaire, "
        "la ou les réponses qui rapportent le plus au joueur choisi."
    )


def _render_history(experiment: Experiment) -> None:
    """Affiche l'historique compact des dernières manches.

    Args:
        experiment: Expérience affichée.
    """
    rounds = experiment.rounds[-HISTORY_DISPLAY_LIMIT:]
    st.dataframe(rounds_to_dataframe(rounds), hide_index=True)
    if len(rounds) < len(experiment.rounds):
        st.caption(
            f"Les {HISTORY_DISPLAY_LIMIT} dernières manches sont affichées ; "
            "l'export CSV contient toutes les manches."
        )


def _render_exports(experiment: Experiment) -> None:
    """Affiche les boutons d'export CSV.

    Args:
        experiment: Expérience exportée.
    """
    left, right = st.columns(2)
    left.download_button(
        "Télécharger les métriques (CSV)",
        to_csv_bytes(experiments_to_samples([experiment])),
        file_name=default_filename("experience_metriques"),
        mime="text/csv",
        key="lab_export_metrics",
    )
    right.download_button(
        "Télécharger toutes les manches (CSV)",
        to_csv_bytes(rounds_to_dataframe(experiment.rounds)),
        file_name=default_filename("experience_manches"),
        mime="text/csv",
        key="lab_export_rounds",
    )


def _render_experiment(experiment: Experiment) -> None:
    """Affiche la dernière expérience sous forme d'onglets.

    Args:
        experiment: Expérience affichée.
    """
    config = experiment.config
    st.success(
        f"{experiment.attacker_stats.agent_type.label} (attaquant) contre "
        f"{experiment.goalkeeper_stats.agent_type.label} (gardien) : "
        f"{config.num_rounds} manches, graine {config.seed}, matrice de "
        "référence. Relancer avec la même graine redonne les mêmes manches."
    )
    metrics, strategies, matrix, history = st.tabs(
        ["Métriques", "Stratégies", "Matrice", "Historique"]
    )
    with metrics:
        _render_metrics(experiment)
    with strategies:
        _render_strategies(experiment)
    with matrix:
        _render_matrix(experiment)
    with history:
        _render_history(experiment)
    _render_exports(experiment)


def render() -> None:
    """Affiche la page « Laboratoire »."""
    st.header("Laboratoire")
    st.write(
        "Choisissez deux agents, un nombre de manches et une graine, puis "
        "lancez un duel automatique. La graine rend l'expérience "
        "reproductible."
    )
    _render_form()
    experiments = get_experiments()
    if not experiments:
        st.info("Aucune expérience pour l'instant : lancez-en une ci-dessus.")
        return
    _render_experiment(experiments[-1])


# Nom utilisé par les pages provisoires du dépôt.
render_laboratory_page = render
