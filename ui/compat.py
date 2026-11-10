"""Couche de branchement entre l'interface et les modules game/ (M1) et agents/ (M2).

L'interface parle en français (« gauche », « attaquant »…) ; ce fichier traduit vers
les conventions du moteur et des agents (indices 0/1/2, rôles « attacker » /
« goalkeeper ») et inversement. Aucune règle de jeu n'est réécrite ici.
"""
import numpy as np
from numpy.random import SeedSequence

from agents import ACTION_NAMES, AGENTS
from agents.base_agent import Observation
from game.actions import Action
from game.payoff import resolve_payoff_matrix
from game.rules import resolve_round as _resolve_round

# Actions affichées/stockées par l'interface : "gauche", "centre", "droite"
ACTIONS = tuple(ACTION_NAMES)

# Rôles de l'interface -> rôles des agents
AGENT_ROLE = {"attaquant": "attacker", "gardien": "goalkeeper"}

# Niveaux de difficulté (section 5 du cahier des charges) -> clé dans agents.AGENTS
LEVELS = {
    "1 — Découverte": "deterministe",
    "2 — Adaptatif": "frequences",
    "3 — Stratégique": "fictitious_play",
    "4 — Optimal": "minimax",
}
LEVEL_HELP = {
    "1 — Découverte": ("Déterministe", "Comportement fixe et prévisible : apprends à exploiter une stratégie pure répétée."),
    "2 — Adaptatif": ("Fréquences", "Observe tes habitudes récentes et les contre."),
    "3 — Stratégique": ("Fictitious Play", "Meilleure réponse à tes fréquences cumulées depuis le début."),
    "4 — Optimal": ("Minimax", "Stratégie mixte optimale (1/3, 1/3, 1/3) : impossible à exploiter."),
}
AGENT_LABELS = {
    "deterministe": "Déterministe",
    "frequences": "Fréquences",
    "fictitious_play": "Fictitious Play",
    "minimax": "Minimax",
}
assert set(LEVELS.values()) == set(AGENTS) == set(AGENT_LABELS), "agents/ et ui/ ne sont plus alignés"


def reference_matrix():
    """Matrice des gains de l'attaquant (M1 : game.payoff)."""
    return resolve_payoff_matrix(None)


def derive_seeds(seed):
    """Deux graines indépendantes (attaquant, gardien), comme game.simulation.run_match."""
    a, g = (int(s) for s in SeedSequence(seed).generate_state(2))
    return a, g


def make_agent(name, role, seed):
    """Crée l'agent `name` pour le rôle d'interface `role` ("attaquant" / "gardien")."""
    return AGENTS[name](AGENT_ROLE[role], seed=seed)


def make_observation(matrix, round_number):
    """Observation autorisée (matrice + numéro de manche, jamais l'action adverse)."""
    return Observation(payoff=np.array(matrix, dtype=float), round_index=round_number)


def action_index(name):
    return ACTIONS.index(name)


def action_name(value):
    """Index (ou Action) -> nom d'interface ; accepte aussi un nom déjà valide."""
    if isinstance(value, str):
        if value not in ACTIONS:
            raise ValueError(f"Action invalide : {value!r}")
        return value
    return ACTIONS[int(value)]


def resolve_round(round_number, action_attaquant, action_gardien, matrix):
    """Délègue à game.rules.resolve_round. Retourne (résultat, gain_attaquant, gain_gardien)."""
    res = _resolve_round(round_number, Action(action_index(action_attaquant)),
                         Action(action_index(action_gardien)), matrix)
    return ("but" if res.is_goal else "arret"), res.attacker_reward, res.goalkeeper_reward
