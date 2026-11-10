"""Figures Plotly du benchmark.

Les fonctions reçoivent le tableau produit par
``analysis.benchmark.summarize_table`` (colonnes ``agent``, ``role``,
``opponent``, ``<métrique>_mean`` et ``<métrique>_ci95``) et ne recalculent
rien.

Dépendances autorisées : ``game.models``, ``visualization.theme``, pandas,
Plotly.
"""

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from game.models import AgentType, Role
from visualization.theme import AGENT_COLORS, FIGURE_HEIGHT

# Conversion secondes -> millisecondes pour l'affichage des temps.
_SECONDS_TO_MS = 1_000.0

# Métriques du graphique de performance : (colonne, libellé attaquant, gardien).
_PERFORMANCE_PANELS: tuple[tuple[str, str, str], ...] = (
    ("goal_rate", "Taux de buts", "Taux de buts concédés"),
    ("save_rate", "Taux d'arrêts (tirs ratés)", "Taux d'arrêts"),
    ("mean_reward", "Gain moyen", "Gain moyen"),
)

# Métriques du graphique de qualité stratégique.
_STRATEGY_PANELS: tuple[tuple[str, str, str], ...] = (
    ("equilibrium_gap", "Écart à l'équilibre", "Écart à l'équilibre"),
    ("exploitability", "Exploitabilité", "Exploitabilité"),
    ("predictability", "Prédictibilité d'ordre 1", "Prédictibilité d'ordre 1"),
)

# Métriques exprimées en proportion (axe en pourcentage).
_RATE_METRICS = frozenset({"goal_rate", "save_rate", "predictability"})


def _require_columns(summary: pd.DataFrame, metric: str) -> None:
    """Vérifie la présence des colonnes nécessaires à une métrique.

    Args:
        summary: Tableau de résumé.
        metric: Nom de la métrique (sans suffixe).

    Raises:
        ValueError: Si une colonne est absente.
    """
    needed = ["agent", "role", "opponent", f"{metric}_mean", f"{metric}_ci95"]
    missing = [column for column in needed if column not in summary.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le résumé : {missing}.")


def _ordered(values: Sequence[str]) -> list[AgentType]:
    """Trie des types d'agents dans l'ordre de ``AgentType``, sans doublon.

    Args:
        values: Valeurs (``AgentType.value``) présentes dans le tableau.

    Returns:
        Les types d'agents correspondants, dans l'ordre de l'énumération.
    """
    present = {AgentType(value) for value in values}
    return [agent for agent in AgentType if agent in present]


def _role_rows(summary: pd.DataFrame, role: Role) -> pd.DataFrame:
    """Filtre les lignes d'un rôle.

    Args:
        summary: Tableau de résumé.
        role: Rôle souhaité.

    Returns:
        Les lignes du rôle.

    Raises:
        ValueError: Si aucune ligne ne correspond au rôle.
    """
    rows = summary[summary["role"] == role.value]
    if rows.empty:
        raise ValueError(f"Aucune ligne pour le rôle {role.value}.")
    return rows


def _add_metric_bars(
    figure: go.Figure,
    rows: pd.DataFrame,
    metric: str,
    *,
    scale: float = 1.0,
    row: int | None = None,
    col: int | None = None,
    show_legend: bool = True,
) -> None:
    """Ajoute une barre par adversaire (avec barres d'erreur IC 95 %).

    Args:
        figure: Figure à compléter.
        rows: Lignes d'un seul rôle.
        metric: Nom de la métrique (sans suffixe).
        scale: Facteur multiplicatif d'affichage (unités).
        row: Ligne du sous-graphique, ou ``None``.
        col: Colonne du sous-graphique, ou ``None``.
        show_legend: Afficher les légendes de cette série.
    """
    agents = _ordered(list(rows["agent"]))
    for opponent in _ordered(list(rows["opponent"])):
        subset = rows[rows["opponent"] == opponent.value]
        names = list(subset["agent"])
        means = dict(zip(names, [float(v) for v in subset[f"{metric}_mean"]]))
        errors = dict(zip(names, [float(v) for v in subset[f"{metric}_ci95"]]))
        present = [a for a in agents if a.value in means]
        trace = go.Bar(
            x=[a.label for a in present],
            y=[means[a.value] * scale for a in present],
            error_y={
                "type": "data",
                "array": [errors[a.value] * scale for a in present],
            },
            name=f"vs {opponent.label}",
            legendgroup=opponent.value,
            marker_color=AGENT_COLORS[opponent],
            showlegend=show_legend,
        )
        if row is None or col is None:
            figure.add_trace(trace)
        else:
            figure.add_trace(trace, row=row, col=col)


def plot_metric_comparison(
    summary: pd.DataFrame,
    metric: str,
    role: Role,
    *,
    title: str | None = None,
    y_title: str | None = None,
    scale: float = 1.0,
    log_y: bool = False,
) -> go.Figure:
    """Compare une métrique entre les agents d'un rôle, par adversaire.

    Args:
        summary: Tableau de ``analysis.benchmark.summarize_table``.
        metric: Nom de la métrique (``goal_rate``, ``mean_reward``...).
        role: Rôle dont on compare les agents.
        title: Titre de la figure ; un titre par défaut est généré.
        y_title: Titre de l'axe vertical.
        scale: Facteur d'affichage (par exemple 1000 pour des millisecondes).
        log_y: Utiliser une échelle logarithmique.

    Returns:
        Un groupe de barres par agent, une série par adversaire, avec les
        intervalles de confiance à 95 %.

    Raises:
        ValueError: Si des colonnes manquent ou si le rôle est absent.
    """
    _require_columns(summary, metric)
    rows = _role_rows(summary, role)
    figure = go.Figure()
    _add_metric_bars(figure, rows, metric, scale=scale)
    figure.update_layout(
        title=title or f"{metric} — {role.label}",
        barmode="group",
        height=FIGURE_HEIGHT,
        xaxis={"title": f"Agent ({role.label})"},
        yaxis={"title": y_title or metric, "type": "log" if log_y else "linear"},
    )
    if metric in _RATE_METRICS and not log_y:
        figure.update_yaxes(tickformat=".0%")
    return figure


def _panels_figure(
    summary: pd.DataFrame,
    role: Role,
    panels: tuple[tuple[str, str, str], ...],
    title: str,
) -> go.Figure:
    """Construit une figure à plusieurs panneaux, un par métrique.

    Args:
        summary: Tableau de résumé.
        role: Rôle étudié.
        panels: Triplets ``(colonne, libellé attaquant, libellé gardien)``.
        title: Titre de la figure.

    Returns:
        La figure à sous-graphiques.

    Raises:
        ValueError: Si des colonnes manquent ou si le rôle est absent.
    """
    for metric, _, _ in panels:
        _require_columns(summary, metric)
    rows = _role_rows(summary, role)
    names = [a if role is Role.ATTACKER else g for _, a, g in panels]
    figure = make_subplots(rows=1, cols=len(panels), subplot_titles=names)
    for index, (metric, _, _) in enumerate(panels, start=1):
        _add_metric_bars(figure, rows, metric, row=1, col=index, show_legend=index == 1)
        if metric in _RATE_METRICS:
            figure.update_yaxes(tickformat=".0%", row=1, col=index)
    figure.update_layout(
        title=f"{title} — {role.label}", barmode="group", height=FIGURE_HEIGHT
    )
    return figure


def plot_performance_overview(summary: pd.DataFrame, role: Role) -> go.Figure:
    """Compare taux de buts, taux d'arrêts et gain moyen des agents.

    Args:
        summary: Tableau de ``analysis.benchmark.summarize_table``.
        role: Rôle dont on compare les agents.

    Returns:
        Trois panneaux de barres avec intervalles de confiance à 95 %.

    Raises:
        ValueError: Si des colonnes manquent ou si le rôle est absent.
    """
    return _panels_figure(summary, role, _PERFORMANCE_PANELS, "Performance")


def plot_strategy_quality(summary: pd.DataFrame, role: Role) -> go.Figure:
    """Compare écart à l'équilibre, exploitabilité et prédictibilité.

    Args:
        summary: Tableau de ``analysis.benchmark.summarize_table``.
        role: Rôle dont on compare les agents.

    Returns:
        Trois panneaux de barres avec intervalles de confiance à 95 %.

    Raises:
        ValueError: Si des colonnes manquent ou si le rôle est absent.
    """
    return _panels_figure(summary, role, _STRATEGY_PANELS, "Qualité stratégique")


def plot_decision_time(
    summary: pd.DataFrame, role: Role, *, log_y: bool = False
) -> go.Figure:
    """Compare le temps de décision moyen des agents, en millisecondes.

    Args:
        summary: Tableau de ``analysis.benchmark.summarize_table``.
        role: Rôle dont on compare les agents.
        log_y: Utiliser une échelle logarithmique.

    Returns:
        Les barres du temps de décision moyen par agent.

    Raises:
        ValueError: Si des colonnes manquent ou si le rôle est absent.
    """
    return plot_metric_comparison(
        summary,
        "mean_decision_time",
        role,
        title=f"Temps de décision moyen — {role.label}",
        y_title="Temps de décision (ms)",
        scale=_SECONDS_TO_MS,
        log_y=log_y,
    )


def plot_matchup_heatmap(
    summary: pd.DataFrame, metric: str = "mean_reward"
) -> go.Figure:
    """Trace la carte des confrontations attaquant x gardien.

    Chaque case donne la moyenne de la métrique de l'agent attaquant (ligne)
    contre l'agent gardien (colonne). Avec ``mean_reward``, une valeur
    positive est favorable à l'attaquant.

    Args:
        summary: Tableau de ``analysis.benchmark.summarize_table`` par
            adversaire.
        metric: Métrique de l'attaquant à afficher.

    Returns:
        La carte thermique des confrontations.

    Raises:
        ValueError: Si des colonnes manquent ou si aucun attaquant n'existe.
    """
    _require_columns(summary, metric)
    rows = _role_rows(summary, Role.ATTACKER)
    attackers = _ordered(list(rows["agent"]))
    goalkeepers = _ordered(list(rows["opponent"]))
    column = f"{metric}_mean"
    values = {
        (agent, opponent): float(value)
        for agent, opponent, value in zip(rows["agent"], rows["opponent"], rows[column])
    }
    z_values: list[list[float | None]] = [
        [values.get((a.value, g.value)) for g in goalkeepers] for a in attackers
    ]
    figure = go.Figure(
        go.Heatmap(
            z=z_values,
            x=[g.label for g in goalkeepers],
            y=[a.label for a in attackers],
            text=[["" if v is None else f"{v:.3f}" for v in line] for line in z_values],
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0 if metric == "mean_reward" else None,
            colorbar={"title": metric},
        )
    )
    figure.update_layout(
        title=f"Confrontations — {metric} de l'attaquant",
        height=FIGURE_HEIGHT,
        xaxis={"title": "Agent gardien", "side": "top"},
        yaxis={"title": "Agent attaquant", "autorange": "reversed"},
    )
    return figure
