"""Page « Jouer » : Humain contre IA et IA contre IA (Membre 3).

Les parties humain contre IA utilisent ``ui.match.Match``. Les simulations
IA contre IA appellent ``game.simulation.run_match`` : une même graine donne
donc exactement les mêmes manches que dans la page « Laboratoire ».
"""

from collections.abc import Mapping

import pandas as pd
import streamlit as st

from analysis.export import rounds_to_dataframe
from game.models import Experiment
from game.models import MatchConfig as EngineConfig
from game.simulation import run_match

from . import compat, state
from .match import (
    ACTIONS,
    ATTAQUANT,
    GARDIEN,
    MODE_HUMAIN_IA,
    Match,
    MatchConfig,
    RoundResult,
    autre_role,
)

ZONES = {"gauche": "Gauche", "centre": "Centre", "droite": "Droite"}

# Nombre de manches affichées dans une simulation accélérée.
_FAST_HISTORY_ROWS = 20


def _frequency_frame(columns: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Met des fréquences par zone sous forme de tableau pour un graphique.

    Args:
        columns: Pour chaque série, la fréquence de chaque zone.

    Returns:
        Un tableau indexé par zone (libellés français).
    """
    return pd.DataFrame(
        {name: [freq[a] for a in ACTIONS] for name, freq in columns.items()},
        index=[ZONES[a] for a in ACTIONS],
    )


def render_play_page() -> None:
    """Affiche la page « Jouer »."""
    st.header("Jouer")
    mode = st.radio(
        "Mode de jeu", ["Humain contre IA", "IA contre IA"], horizontal=True
    )
    if mode == "Humain contre IA":
        _render_human()
    else:
        _render_ia_vs_ia()


# Alias au nom générique utilisé par les autres pages.
render = render_play_page


# Humain contre IA
def _render_human() -> None:
    """Affiche la configuration ou la partie en cours."""
    match = st.session_state.get(state.HUMAN_MATCH)
    if match is None:
        _human_config()
    else:
        _human_game(match)


def _human_config() -> None:
    """Affiche le formulaire de configuration d'une partie humain contre IA."""
    with st.expander("Que fait chaque niveau ?"):
        for level, (agent, text) in compat.LEVEL_HELP.items():
            st.markdown(f"**{level}** ({agent}) : {text}")
    with st.form("config_humain"):
        role = st.radio(
            "Ton rôle",
            [ATTAQUANT, GARDIEN],
            format_func=str.capitalize,
            horizontal=True,
        )
        niveau = st.selectbox("Niveau de l'adversaire", list(compat.LEVELS))
        n = st.number_input("Nombre de manches", min_value=1, max_value=100, value=5)
        graine = st.number_input(
            "Graine (reproductibilité)", min_value=0, max_value=10**6, value=42
        )
        lancer = st.form_submit_button("Lancer la partie", type="primary")
    if lancer and role is not None and niveau is not None:
        agents: dict[str, str | None] = {ATTAQUANT: None, GARDIEN: None}
        agents[autre_role(role)] = compat.LEVELS[niveau]
        cfg = MatchConfig(MODE_HUMAIN_IA, int(n), agents, int(graine), role_humain=role)
        st.session_state[state.HUMAN_MATCH] = Match(cfg)
        st.rerun()


def _jouer(action: str) -> None:
    """Joue une manche avec l'action de l'humain (appelé par les boutons).

    Args:
        action: Zone choisie par l'humain.
    """
    st.session_state[state.HUMAN_MATCH].play_round(action)


def _message(r: RoundResult, role: str) -> tuple[str, bool]:
    """Construit le message de résultat d'une manche pour l'humain.

    Args:
        r: Résultat de la manche.
        role: Rôle tenu par l'humain.

    Returns:
        Le texte et un booléen indiquant si le résultat est favorable.
    """
    but = r.resultat == "but"
    if role == ATTAQUANT:
        return (" But !", True) if but else ("Arrêté par le gardien.", False)
    return (" Arrêt !", True) if not but else ("Le tireur a marqué.", False)


def _human_game(match: Match) -> None:
    """Affiche une partie humain contre IA en cours ou terminée.

    Args:
        match: Partie en cours.
    """
    cfg = match.config
    role = cfg.role_humain
    if role is None:
        return
    adversaire = cfg.agents[autre_role(role)]
    if adversaire is None:
        return
    n_done = len(match.history)
    st.caption(f"Tu es **{role}** — adversaire : **{compat.AGENT_LABELS[adversaire]}**")
    st.progress(
        n_done / cfg.nombre_manches,
        text=f"Manche {min(n_done + 1, cfg.nombre_manches)} / {cfg.nombre_manches}",
    )

    s = match.summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Buts", int(s["buts"]))
    c2.metric("Arrêts", int(s["arrets"]))
    c3.metric("Ton gain cumulé", f"{match.human_gain():+d}")

    if match.history:
        last = match.history[-1]
        text, bon = _message(last, role)
        detail = (
            f"{text}  (attaquant : {last.action_attaquant}, "
            f"gardien : {last.action_gardien})"
        )
        (st.success if bon else st.error)(detail)

    if not match.finished:
        verbe = "Où tires-tu ?" if role == ATTAQUANT else "De quel côté plonges-tu ?"
        st.subheader(verbe)
        cols = st.columns(3)
        for col, action in zip(cols, ACTIONS, strict=True):
            col.button(
                ZONES[action],
                key=f"zone_{action}",
                on_click=_jouer,
                args=(action,),
            )
    else:
        _human_bilan(match)

    if match.history:
        with st.expander("Historique de la session", expanded=match.finished):

            st.dataframe(
                pd.DataFrame(match.history_rows()), hide_index=True
            )

    label = (
        "Nouvelle configuration"
        if match.finished
        else "Abandonner / changer les réglages"
    )
    if st.button(label):
        state.reset_human()
        st.rerun()


def _human_bilan(match: Match) -> None:
    """Affiche le bilan de fin de partie.

    Args:
        match: Partie terminée.
    """
    cfg = match.config
    role = cfg.role_humain
    if role is None:
        return
    st.subheader("Fin de partie")
    gain = match.human_gain()
    if gain > 0:
        st.success(f"Victoire ! Gain total : {gain:+g}")
    elif gain < 0:
        st.error(f"Défaite. Gain total : {gain:+g}")
    else:
        st.info("Égalité (gain total : 0).")
    st.markdown("**Tendances observées** (part de chaque zone jouée) :")
    ia_role = autre_role(role)
    st.bar_chart(
        _frequency_frame(
            {"Toi": match.frequencies(role), "IA": match.frequencies(ia_role)}
        )
    )
    st.caption("Référence théorique à l'équilibre : 1/3 par zone.")
    if st.button("Rejouer avec les mêmes réglages", type="primary"):
        st.session_state[state.HUMAN_MATCH] = Match(cfg)
        st.rerun()


# IA contre IA
def _render_ia_vs_ia() -> None:
    """Affiche le formulaire de simulation et le dernier résultat."""
    with st.form("config_ia"):
        noms = list(compat.AGENT_LABELS)
        c1, c2 = st.columns(2)
        att = c1.selectbox(
            "Agent attaquant", noms, format_func=lambda name: compat.AGENT_LABELS[name]
        )
        gar = c2.selectbox(
            "Agent gardien",
            noms,
            index=3,
            format_func=lambda name: compat.AGENT_LABELS[name],
        )
        n = st.number_input(
            "Nombre de manches", min_value=1, max_value=100_000, value=1000
        )
        graine = st.number_input(
            "Graine (reproductibilité)", min_value=0, max_value=10**6, value=42
        )
        detail = st.radio("Simulation", ["Accélérée", "Détaillée"], horizontal=True)
        lancer = st.form_submit_button("Lancer la simulation", type="primary")
    if lancer and att is not None and gar is not None and detail is not None:
        try:
            config = EngineConfig(
                num_rounds=int(n),
                attacker_agent=compat.to_agent_type(att),
                goalkeeper_agent=compat.to_agent_type(gar),
                seed=int(graine),
            )
            with st.spinner("Simulation en cours..."):
                experiment = run_match(config)
        except (ValueError, RuntimeError) as error:
            st.error(f"Impossible de lancer la simulation : {error}")
        else:
            st.session_state[state.SIM_MATCH] = (experiment, detail)

    stored = st.session_state.get(state.SIM_MATCH)
    if stored:
        _sim_result(*stored)


def _sim_result(experiment: Experiment, detail: str) -> None:
    """Affiche le résultat d'une simulation IA contre IA.

    Args:
        experiment: Expérience produite par ``run_match``.
        detail: Mode d'affichage (« Accélérée » ou « Détaillée »).
    """
    attacker = experiment.attacker_stats
    goalkeeper = experiment.goalkeeper_stats
    st.subheader(
        f"{attacker.agent_type.label} (attaque) vs "
        f"{goalkeeper.agent_type.label} (garde)"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Manches", experiment.config.num_rounds)
    c2.metric("Taux de buts", f"{attacker.goal_rate:.1%}")
    c3.metric("Taux d'arrêts", f"{attacker.save_rate:.1%}")
    c4.metric("Gain moyen attaquant", f"{attacker.mean_reward:+.3f}")
    st.caption(
        "Valeur théorique du jeu pour l'attaquant : +1/3 ≈ 0,333 "
        "(taux de buts d'équilibre : 2/3)."
    )

    st.markdown("**Stratégies empiriques** (part de chaque zone jouée) :")
    st.bar_chart(
        pd.DataFrame(
            {
                "Attaquant": list(attacker.empirical_strategy),
                "Gardien": list(goalkeeper.empirical_strategy),
            },
            index=[ZONES[a] for a in ACTIONS],
        )
    )

    rounds = experiment.rounds
    with st.expander("Historique", expanded=(detail == "Détaillée")):
        shown = rounds if detail == "Détaillée" else rounds[-_FAST_HISTORY_ROWS:]
        if detail != "Détaillée" and len(rounds) > _FAST_HISTORY_ROWS:
            st.caption(
                "Simulation accélérée : seules les "
                f"{_FAST_HISTORY_ROWS} dernières manches sont affichées."
            )
        st.dataframe(rounds_to_dataframe(shown), hide_index=True)
