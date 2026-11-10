"""Clés de ``st.session_state`` : tout l'état vit ici.

Streamlit relance le script à chaque clic, donc l'état de la session est
centralisé dans ce module.
"""

import streamlit as st

# Partie humain contre IA en cours (``ui.match.Match``).
HUMAN_MATCH = "match_humain"

# Dernière simulation IA contre IA : couple ``(Experiment, mode d'affichage)``.
SIM_MATCH = "match_ia"


def reset_human() -> None:
    """Abandonne la partie humain contre IA en cours."""
    st.session_state.pop(HUMAN_MATCH, None)


def reset_sim() -> None:
    """Efface la dernière simulation IA contre IA."""
    st.session_state.pop(SIM_MATCH, None)
