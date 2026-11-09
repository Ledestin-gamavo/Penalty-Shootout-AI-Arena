"""Clés de st.session_state : tout l'état vit ici car Streamlit relance le script à chaque clic."""
import streamlit as st

HUMAN_MATCH = "match_humain"
SIM_MATCH = "match_ia"


def reset_human():
    st.session_state.pop(HUMAN_MATCH, None)


def reset_sim():
    st.session_state.pop(SIM_MATCH, None)
