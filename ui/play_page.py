"""Page « Jouer » : Humain contre IA et IA contre IA (Membre 3)."""
import pandas as pd
import streamlit as st

from . import compat, state
from .match import (ACTIONS, ATTAQUANT, GARDIEN, MODE_HUMAIN_IA, MODE_IA_IA,
                    Match, MatchConfig, autre_role)

ZONES = {"gauche": "Gauche", "centre": "Centre", "droite": "Droite"}


def render_play_page():
    st.header("Jouer")
    mode = st.radio("Mode de jeu", ["Humain contre IA", "IA contre IA"], horizontal=True)
    if mode == "Humain contre IA":
        _render_human()
    else:
        _render_ia_vs_ia()


# Humain contre IA
def _render_human():
    match = st.session_state.get(state.HUMAN_MATCH)
    if match is None:
        _human_config()
    else:
        _human_game(match)


def _human_config():
    with st.expander("Que fait chaque niveau ?"):
        for level, (agent, text) in compat.LEVEL_HELP.items():
            st.markdown(f"**{level}** ({agent}) : {text}")
    with st.form("config_humain"):
        role = st.radio("Ton rôle", [ATTAQUANT, GARDIEN], format_func=str.capitalize, horizontal=True)
        niveau = st.selectbox("Niveau de l'adversaire", list(compat.LEVELS))
        n = st.number_input("Nombre de manches", min_value=1, max_value=100, value=5)
        graine = st.number_input("Graine (reproductibilité)", min_value=0, max_value=10**6, value=42)
        lancer = st.form_submit_button("Lancer la partie", type="primary")
    if lancer:
        agents = {ATTAQUANT: None, GARDIEN: None}
        agents[autre_role(role)] = compat.LEVELS[niveau]
        cfg = MatchConfig(MODE_HUMAIN_IA, int(n), agents, int(graine), role_humain=role)
        st.session_state[state.HUMAN_MATCH] = Match(cfg)
        st.rerun()


def _jouer(action):
    st.session_state[state.HUMAN_MATCH].play_round(action)


def _message(r, role):
    but = r.resultat == "but"
    if role == ATTAQUANT:
        return (" But !", True) if but else ("Arrêté par le gardien.", False)
    return (" Arrêt !", True) if not but else ("Le tireur a marqué.", False)


def _human_game(match):
    cfg = match.config
    role = cfg.role_humain
    adversaire = cfg.agents[autre_role(role)]
    n_done = len(match.history)
    st.caption(f"Tu es **{role}** — adversaire : **{compat.AGENT_LABELS[adversaire]}**")
    st.progress(n_done / cfg.nombre_manches, text=f"Manche {min(n_done + 1, cfg.nombre_manches)} / {cfg.nombre_manches}")

    s = match.summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Buts", s["buts"])
    c2.metric("Arrêts", s["arrets"])
    c3.metric("Ton gain cumulé", f"{match.human_gain():+d}")

    if match.history:
        last = match.history[-1]
        text, bon = _message(last, role)
        detail = f"{text}  (attaquant : {last.action_attaquant}, gardien : {last.action_gardien})"
        (st.success if bon else st.error)(detail)

    if not match.finished:
        verbe = "Où tires-tu ?" if role == ATTAQUANT else "De quel côté plonges-tu ?"
        st.subheader(verbe)
        cols = st.columns(3)
        for col, action in zip(cols, ACTIONS):
            col.button(ZONES[action], key=f"zone_{action}", on_click=_jouer, args=(action,),
                       width="stretch")
    else:
        _human_bilan(match)

    if match.history:
        with st.expander("Historique de la session", expanded=match.finished):
            st.dataframe(pd.DataFrame(match.history_rows()), hide_index=True, width="stretch")

    if st.button("Abandonner / changer les réglages" if not match.finished else "Nouvelle configuration"):
        state.reset_human()
        st.rerun()


def _human_bilan(match):
    cfg = match.config
    role = cfg.role_humain
    st.subheader("Fin de partie")
    gain = match.human_gain()
    if gain > 0:
        st.success(f"Victoire ! Gain total : {gain:+d}")
    elif gain < 0:
        st.error(f"Défaite. Gain total : {gain:+d}")
    else:
        st.info("Égalité (gain total : 0).")
    st.markdown("**Tendances observées** (part de chaque zone jouée) :")
    ia_role = autre_role(role)
    df = pd.DataFrame({"Toi": match.frequencies(role), "IA": match.frequencies(ia_role)})
    df.index = [ZONES[a] for a in df.index]
    st.bar_chart(df)
    st.caption("Référence théorique à l'équilibre : 1/3 par zone.")
    if st.button("Rejouer avec les mêmes réglages", type="primary"):
        st.session_state[state.HUMAN_MATCH] = Match(cfg)
        st.rerun()


# IA contre IA
def _render_ia_vs_ia():
    with st.form("config_ia"):
        noms = list(compat.AGENT_LABELS)
        c1, c2 = st.columns(2)
        att = c1.selectbox("Agent attaquant", noms, format_func=compat.AGENT_LABELS.get)
        gar = c2.selectbox("Agent gardien", noms, index=3, format_func=compat.AGENT_LABELS.get)
        n = st.number_input("Nombre de manches", min_value=1, max_value=100_000, value=1000)
        graine = st.number_input("Graine (reproductibilité)", min_value=0, max_value=10**6, value=42)
        detail = st.radio("Simulation", ["Accélérée", "Détaillée"], horizontal=True)
        lancer = st.form_submit_button("Lancer la simulation", type="primary")
    if lancer:
        cfg = MatchConfig(MODE_IA_IA, int(n), {ATTAQUANT: att, GARDIEN: gar}, int(graine))
        sim = Match(cfg)
        sim.run_all()
        st.session_state[state.SIM_MATCH] = (sim, detail)

    stored = st.session_state.get(state.SIM_MATCH)
    if stored:
        _sim_result(*stored)


def _sim_result(sim, detail):
    cfg = sim.config
    s = sim.summary()
    st.subheader(f"{compat.AGENT_LABELS[cfg.agents[ATTAQUANT]]} (attaque) vs "
                 f"{compat.AGENT_LABELS[cfg.agents[GARDIEN]]} (garde)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Manches", s["manches"])
    c2.metric("Taux de buts", f"{s['taux_buts']:.1%}")
    c3.metric("Taux d'arrêts", f"{s['taux_arrets']:.1%}")
    c4.metric("Gain moyen attaquant", f"{s['gain_moyen_attaquant']:+.3f}")
    st.caption("Valeur théorique du jeu pour l'attaquant : +1/3 ≈ 0,333 (taux de buts d'équilibre : 2/3).")

    st.markdown("**Stratégies empiriques** (part de chaque zone jouée) :")
    df = pd.DataFrame({"Attaquant": sim.frequencies(ATTAQUANT), "Gardien": sim.frequencies(GARDIEN)})
    df.index = [ZONES[a] for a in df.index]
    st.bar_chart(df)

    rows = sim.history_rows()
    with st.expander("Historique", expanded=(detail == "Détaillée")):
        shown = rows if detail == "Détaillée" else rows[-20:]
        if detail != "Détaillée" and len(rows) > 20:
            st.caption("Simulation accélérée : seules les 20 dernières manches sont affichées.")
        st.dataframe(pd.DataFrame(shown), hide_index=True, width="stretch")
