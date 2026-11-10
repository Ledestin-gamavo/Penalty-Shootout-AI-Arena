"""Figures Plotly d'une stratégie, d'une matrice ou d'une expérience.

Une fonction produit une figure. Les calculs viennent de
``theory.benchmark_metrics`` ; ce module ne fait que les mettre en forme.

Dépendances autorisées : ``game.*``, ``theory.benchmark_metrics``, Plotly.
"""

import math
from collections.abc import Sequence

import plotly.graph_objects as go

from game.actions import Action
from game.constants import N_ACTIONS
from game.models import PayoffMatrix, Role, RoundResult, Strategy
from theory.benchmark_metrics import cumulative_frequencies, cumulative_mean_reward
from visualization.theme import (
    ACTION_COLORS,
    FIGURE_HEIGHT,
    MAX_PLOT_POINTS,
    REFERENCE_COLOR,
)


def _action_labels() -> list[str]:
    """Retourne les libellés des actions dans l'ordre des indices.

    Returns:
        Les libellés français de ``Action``.
    """
    return [action.label for action in Action]


def _sampled_indices(count: int) -> list[int]:
    """Choisit les indices à tracer pour limiter le nombre de points.

    Le dernier indice est toujours conservé.

    Args:
        count: Nombre de points disponibles.

    Returns:
        Au plus ``MAX_PLOT_POINTS`` indices croissants.
    """
    step = max(1, math.ceil(count / MAX_PLOT_POINTS))
    indices = list(range(0, count, step))
    if indices[-1] != count - 1:
        indices.append(count - 1)
    return indices


def plot_strategy_bars(
    strategy: Strategy,
    reference: Strategy | None = None,
    *,
    title: str = "Stratégie",
    strategy_name: str = "Agent",
    reference_name: str = "Équilibre",
) -> go.Figure:
    """Trace les probabilités (ou fréquences) gauche / centre / droite.

    Args:
        strategy: Stratégie à afficher (par exemple empirique).
        reference: Stratégie de référence affichée à côté, ou ``None``.
        title: Titre de la figure.
        strategy_name: Légende de la stratégie.
        reference_name: Légende de la référence.

    Returns:
        La figure en barres groupées.

    Raises:
        ValueError: Si une stratégie n'a pas ``N_ACTIONS`` probabilités.
    """
    for candidate in (strategy, reference):
        if candidate is not None and len(candidate) != N_ACTIONS:
            raise ValueError(f"Une stratégie doit avoir {N_ACTIONS} probabilités.")
    labels = _action_labels()
    figure = go.Figure()
    figure.add_bar(
        x=labels,
        y=list(strategy),
        name=strategy_name,
        text=[f"{p:.1%}" for p in strategy],
        textposition="outside",
    )
    if reference is not None:
        figure.add_bar(
            x=labels,
            y=list(reference),
            name=reference_name,
            marker_color=REFERENCE_COLOR,
            text=[f"{p:.1%}" for p in reference],
            textposition="outside",
        )
    figure.update_layout(
        title=title,
        barmode="group",
        height=FIGURE_HEIGHT,
        yaxis={"title": "Probabilité", "range": [0, 1.1], "tickformat": ".0%"},
    )
    return figure


def plot_payoff_heatmap(
    payoff_matrix: PayoffMatrix,
    highlight_cells: Sequence[tuple[Action, Action]] = (),
    *,
    title: str = "Matrice des gains de l'attaquant",
) -> go.Figure:
    """Trace la matrice des gains sous forme de carte thermique.

    Les lignes sont les actions de l'attaquant, les colonnes celles du
    gardien. Les cases de ``highlight_cells`` sont entourées (par exemple les
    meilleures réponses, voir ``theory.benchmark_metrics.best_response_cells``).

    Args:
        payoff_matrix: Matrice des gains de l'attaquant.
        highlight_cells: Couples ``(action attaquant, action gardien)``.
        title: Titre de la figure.

    Returns:
        La carte thermique annotée.

    Raises:
        ValueError: Si la matrice n'est pas ``N_ACTIONS`` x ``N_ACTIONS``.
    """
    if len(payoff_matrix) != N_ACTIONS or any(
        len(row) != N_ACTIONS for row in payoff_matrix
    ):
        raise ValueError(f"La matrice doit être {N_ACTIONS} x {N_ACTIONS}.")
    labels = _action_labels()
    figure = go.Figure(
        go.Heatmap(
            z=[list(row) for row in payoff_matrix],
            x=labels,
            y=labels,
            text=[[f"{value:+g}" for value in row] for row in payoff_matrix],
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0,
            colorbar={"title": "Gain attaquant"},
            hovertemplate="Tir : %{y}<br>Plongeon : %{x}<br>Gain : %{z}<extra></extra>",
        )
    )
    for attacker_action, goalkeeper_action in highlight_cells:
        figure.add_shape(
            type="rect",
            x0=int(goalkeeper_action) - 0.5,
            x1=int(goalkeeper_action) + 0.5,
            y0=int(attacker_action) - 0.5,
            y1=int(attacker_action) + 0.5,
            line={"width": 3},
        )
    figure.update_layout(
        title=title,
        height=FIGURE_HEIGHT,
        xaxis={"title": "Plongeon du gardien", "side": "top"},
        yaxis={"title": "Tir de l'attaquant", "autorange": "reversed"},
    )
    return figure


def plot_frequency_evolution(
    rounds: Sequence[RoundResult],
    role: Role,
    reference: Strategy | None = None,
    *,
    title: str | None = None,
) -> go.Figure:
    """Trace l'évolution des fréquences empiriques d'un rôle.

    Args:
        rounds: Manches jouées, dans l'ordre.
        role: Rôle dont on suit les fréquences.
        reference: Stratégie de référence (lignes en pointillés), ou ``None``.
        title: Titre de la figure ; un titre par défaut est généré.

    Returns:
        Une courbe par action (fréquence cumulée en fonction de la manche).

    Raises:
        ValueError: Si aucune manche n'est fournie.
    """
    frequencies = cumulative_frequencies([r.action_of(role) for r in rounds])
    indices = _sampled_indices(len(frequencies))
    x_values = [rounds[i].round_number for i in indices]
    figure = go.Figure()
    for action in Action:
        color = ACTION_COLORS[action]
        figure.add_scatter(
            x=x_values,
            y=[frequencies[i][action] for i in indices],
            mode="lines",
            name=action.label,
            line={"color": color},
        )
        if reference is not None:
            figure.add_hline(
                y=reference[action],
                line={"color": color, "dash": "dot", "width": 1},
            )
    figure.update_layout(
        title=title or f"Évolution des fréquences — {role.label}",
        height=FIGURE_HEIGHT,
        xaxis={"title": "Manche"},
        yaxis={"title": "Fréquence cumulée", "range": [0, 1], "tickformat": ".0%"},
    )
    return figure


def plot_reward_evolution(
    rounds: Sequence[RoundResult],
    role: Role,
    reference_value: float | None = None,
    *,
    title: str | None = None,
) -> go.Figure:
    """Trace l'évolution du gain moyen cumulé d'un rôle.

    Args:
        rounds: Manches jouées, dans l'ordre.
        role: Rôle dont on suit le gain.
        reference_value: Gain de référence en pointillés (valeur du jeu du
            point de vue du rôle), ou ``None``.
        title: Titre de la figure ; un titre par défaut est généré.

    Returns:
        La courbe du gain moyen après chaque manche.

    Raises:
        ValueError: Si aucune manche n'est fournie.
    """
    means = cumulative_mean_reward(rounds, role)
    indices = _sampled_indices(len(means))
    figure = go.Figure(
        go.Scatter(
            x=[rounds[i].round_number for i in indices],
            y=[means[i] for i in indices],
            mode="lines",
            name="Gain moyen cumulé",
        )
    )
    if reference_value is not None:
        figure.add_hline(
            y=reference_value,
            line={"color": REFERENCE_COLOR, "dash": "dot"},
            annotation_text="Valeur du jeu",
        )
    figure.update_layout(
        title=title or f"Gain moyen cumulé — {role.label}",
        height=FIGURE_HEIGHT,
        xaxis={"title": "Manche"},
        yaxis={"title": "Gain moyen"},
    )
    return figure
