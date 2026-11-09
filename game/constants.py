"""Constantes métier du projet.

Ce module centralise les récompenses, les tolérances numériques, les valeurs
par défaut et les paramètres des agents. Aucune valeur métier ne doit être
écrite « en dur » ailleurs dans le code.

Dépendance autorisée : uniquement ``game.actions``.
"""

from pathlib import Path
from typing import Final
from game.actions import Action

# Nombre d'actions disponibles pour chaque joueur.
N_ACTIONS: Final[int] = len(Action)

# Gain de l'attaquant lorsqu'un but est marqué.
REWARD_GOAL: Final[float] = 1.0

# Gain de l'attaquant lorsqu'un tir est arrêté.
REWARD_SAVE: Final[float] = -1.0

# Probabilité de chaque action dans la stratégie uniforme.
UNIFORM_PROBABILITY: Final[float] = 1.0 / N_ACTIONS

# Stratégie d'équilibre du jeu de référence (uniforme)
REFERENCE_EQUILIBRIUM_STRATEGY: Final[tuple[float, ...]] = (UNIFORM_PROBABILITY,) * N_ACTIONS

# Valeur du jeu de référence pour l'attaquant : (-1 + 1 + 1 + 1 - 1 + 1 + 1 + 1 -1) / 9 = 1/3
REFERENCE_GAME_VALUE: Final[float] = ((N_ACTIONS - 1) * REWARD_GOAL + REWARD_SAVE) / N_ACTIONS

# Tolérance pour vérifier qu'une distribution somme à 1 et reste positive.
PROBABILITY_TOLERANCE: Final[float] = 1e-9

# Bruit numérique : valeurs négatives ramenées à 0 après un calcul de programmation linéaire, et écarts ignorés pour départager deux actions de gain espéré identique.
NUMERICAL_NOISE_TOLERANCE: Final[float] = 1e-12

# Tolérance pour comparer un résultat de programmation linéaire à la valeur théorique de l'équilibre.
EQUILIBRIUM_TOLERANCE: Final[float] = 1e-6

# Nombre minimal de manches d'un match.
MIN_NUM_ROUNDS: Final[int] = 1

# Nombre maximal de manches d'un match.
MAX_NUM_ROUNDS: Final[int] = 100_000

# Nombre de manches par défaut pour une partie humain contre IA.
DEFAULT_NUM_ROUNDS_HUMAN: Final[int] = 10

# Nombre de manches par défaut pour une simulation IA contre IA.
DEFAULT_NUM_ROUNDS_SIMULATION: Final[int] = 1_000

# Graine par défaut des expériences reproductibles.
DEFAULT_SEED: Final[int] = 42

# Nombre de graines par défaut pour répéter une expérience de benchmark.
DEFAULT_BENCHMARK_NUM_SEEDS: Final[int] = 10

# Cycle d'actions de l'agent déterministe (une seule action : toujours gauche).
DETERMINISTIC_ACTION_CYCLE: Final[tuple[Action, ...]] = (Action.LEFT,)

# Taille de la fenêtre glissante de l'agent basé sur les fréquences.
FREQUENCY_WINDOW_SIZE: Final[int] = 10

# Pseudo-comptage de lissage de l'agent basé sur les fréquences (évite les fréquences nulles).
FREQUENCY_SMOOTHING: Final[float] = 0.5

# Pseudo-comptage initial par action de Fictitious Play (croyance initiale uniforme).
FICTITIOUS_PLAY_INITIAL_COUNT: Final[float] = 1.0


# ---- Chemins  de données -----

# Racine du dépôt (ce fichier se trouve dans ``game/``).
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# Dossier des résultats bruts d'expériences.
EXPERIMENTS_DIR: Final[Path] = PROJECT_ROOT / "data" / "experiments"

# Dossier des exports CSV.
EXPORTS_DIR: Final[Path] = PROJECT_ROOT / "data" / "exports"