"""Couche de branchement entre l'interface et les modules des Membres 1 et 2.

C'est LE SEUL fichier à modifier quand game/ et agents/ seront livrés.
Si un vrai agent est introuvable ou ne s'instancie pas, on retombe sur l'agent factice.
"""
import importlib

from .stub_agents import STUBS

ACTIONS = ("gauche", "centre", "droite")

# Niveaux de difficulté (section 5 du cahier des charges)
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

# TODO Membre 2 : confirmer les noms de modules/classes et la signature du constructeur.
_REAL_AGENTS = {
    "deterministe": ("agents.deterministic_agent", "DeterministicAgent"),
    "frequences": ("agents.frequency_agent", "FrequencyAgent"),
    "minimax": ("agents.minimax_agent", "MinimaxAgent"),
    "fictitious_play": ("agents.fictitious_play_agent", "FictitiousPlayAgent"),
}

AGENT_SOURCE = {}  # nom -> "réel" ou "factice" (affiché dans la barre latérale)


def make_agent(name, seed=0):
    """Retourne un agent respectant le contrat BaseAgent."""
    module_name, class_name = _REAL_AGENTS[name]
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
        agent = cls(seed=seed)
        AGENT_SOURCE[name] = "réel"
        return agent
    except (ImportError, AttributeError, TypeError):
        AGENT_SOURCE[name] = "factice"
        return STUBS[name](seed=seed)


def resolve_round(action_attaquant, action_gardien):
    """Règle de base (section 3.2). Retourne (résultat, gain_attaquant, gain_gardien).

    TODO Membre 1 : remplacer par game.rules / game.payoff une fois livrés.
    """
    for a in (action_attaquant, action_gardien):
        if a not in ACTIONS:
            raise ValueError(f"Action invalide : {a!r}")
    if action_attaquant == action_gardien:
        return "arret", -1, 1
    return "but", 1, -1
