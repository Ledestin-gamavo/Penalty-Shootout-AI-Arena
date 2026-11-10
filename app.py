"""Point d'entrée : streamlit run app.py"""
import streamlit as st

from ui.analysis_page import render_analysis_page
from ui.benchmark_page import render_benchmark_page
from ui.laboratory_page import render_laboratory_page
from ui.play_page import render_play_page

st.set_page_config(page_title="Penalty Shootout AI Arena", page_icon="⚽", layout="wide")

PAGES = {
    "Jouer": render_play_page,
    "Laboratoire": render_laboratory_page,
    "Analyse": render_analysis_page,
    "Benchmark": render_benchmark_page,
}

st.sidebar.title("⚽ Penalty Shootout AI Arena")
choix = st.sidebar.radio("Navigation", list(PAGES))
with st.sidebar.expander("Règles rapides"):
    st.markdown(
        "Attaquant et gardien choisissent **en même temps** gauche, centre ou droite. "
        "Même zone = **arrêt** ; sinon = **but**. Jeu à somme nulle."
    )
PAGES[choix]()
