"""État partagé du laboratoire dans ``st.session_state``.

Les expériences lancées dans la page « Laboratoire » sont conservées pour la
session afin que la page « Analyse » puisse les consulter et les comparer.

Dépendances autorisées : ``game.models`` et Streamlit.
"""

from typing import Final

import streamlit as st

from game.models import Experiment

# Clé de session de la liste des expériences du laboratoire.
EXPERIMENTS_KEY: Final[str] = "lab_experiments"

# Nombre maximal d'expériences conservées (les plus anciennes sont retirées).
MAX_STORED_EXPERIMENTS: Final[int] = 20

# Nombre maximal de manches affichées dans les tableaux d'historique.
HISTORY_DISPLAY_LIMIT: Final[int] = 500


def get_experiments() -> list[Experiment]:
    """Retourne les expériences de la session, de la plus ancienne à la plus récente.

    Returns:
        La liste (éventuellement vide) des expériences conservées.
    """
    experiments: list[Experiment] = st.session_state.setdefault(EXPERIMENTS_KEY, [])
    return experiments


def add_experiment(experiment: Experiment) -> None:
    """Ajoute une expérience à la session, en retirant les plus anciennes.

    Args:
        experiment: Expérience à conserver.
    """
    experiments = get_experiments()
    experiments.append(experiment)
    del experiments[:-MAX_STORED_EXPERIMENTS]


def clear_experiments() -> None:
    """Efface toutes les expériences conservées."""
    get_experiments().clear()
