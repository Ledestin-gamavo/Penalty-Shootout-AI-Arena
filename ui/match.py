"""Logique d'une partie — sans Streamlit, donc testable.

Garantie centrale (EF04, décisions simultanées) : dans play_round, chaque agent choisit
à partir de l'historique PUBLIC des manches terminées uniquement ; les actions sont ensuite
révélées ensemble, puis seulement, update() est appelé.
"""
import random
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from . import compat

ACTIONS = compat.ACTIONS
ATTAQUANT = "attaquant"
GARDIEN = "gardien"
ROLES = (ATTAQUANT, GARDIEN)
MODE_HUMAIN_IA = "humain_vs_ia"
MODE_IA_IA = "ia_vs_ia"


def autre_role(role):
    return GARDIEN if role == ATTAQUANT else ATTAQUANT


@dataclass(frozen=True)
class RoundResult:
    numero: int
    action_attaquant: str
    action_gardien: str
    resultat: str  # "but" ou "arret"
    gain_attaquant: int
    gain_gardien: int


@dataclass
class MatchConfig:
    mode: str
    nombre_manches: int
    agents: dict  # {"attaquant": nom|None, "gardien": nom|None} ; None = rôle tenu par l'humain
    graine: int = 0
    role_humain: Optional[str] = None

    def __post_init__(self):
        if self.nombre_manches < 1:
            raise ValueError("Il faut au moins une manche.")
        if self.mode == MODE_HUMAIN_IA and self.role_humain not in ROLES:
            raise ValueError("role_humain requis en mode Humain contre IA.")


class Match:
    def __init__(self, config, agents=None):
        """`agents` permet d'injecter des agents (tests) ; sinon ils sont créés via compat."""
        self.config = config
        self.history = []
        self._rng = random.Random(config.graine)
        if agents is not None:
            self.agents = dict(agents)
        else:
            self.agents = {}
            for i, role in enumerate(ROLES):
                name = config.agents.get(role)
                if name is not None:
                    self.agents[role] = compat.make_agent(name, seed=config.graine + i)
        for agent in self.agents.values():
            agent.reset()

    @property
    def finished(self):
        return len(self.history) >= self.config.nombre_manches

    def _observation(self, role):
        return {
            "role": role,
            "round": len(self.history) + 1,
            "total_rounds": self.config.nombre_manches,
            "history": [(r.action_attaquant, r.action_gardien) for r in self.history],
        }

    def _to_action(self, choice):
        """Accepte une action, ou une politique (dict ou séquence de 3 probabilités)."""
        if isinstance(choice, str):
            action = choice
        else:
            weights = [choice[a] for a in ACTIONS] if isinstance(choice, dict) else list(choice)
            action = self._rng.choices(ACTIONS, weights=weights)[0]
        if action not in ACTIONS:
            raise ValueError(f"Action invalide : {action!r}")
        return action

    def play_round(self, human_action=None):
        if self.finished:
            raise RuntimeError("La partie est terminée.")
        role_h = self.config.role_humain if self.config.mode == MODE_HUMAIN_IA else None

        # 1. Décisions simultanées : historique public uniquement
        observations, actions = {}, {}
        for role, agent in self.agents.items():
            observations[role] = self._observation(role)
            actions[role] = self._to_action(agent.choose_action(observations[role]))
        if role_h is not None:
            if human_action is None:
                raise ValueError("Une action humaine est requise.")
            if human_action not in ACTIONS:
                raise ValueError(f"Action invalide : {human_action!r}")
            actions[role_h] = human_action

        # 2. Révélation et résolution
        resultat, g_att, g_gar = compat.resolve_round(actions[ATTAQUANT], actions[GARDIEN])
        record = RoundResult(len(self.history) + 1, actions[ATTAQUANT], actions[GARDIEN], resultat, g_att, g_gar)

        # 3. Mise à jour des agents APRÈS révélation
        for role, agent in self.agents.items():
            reward = g_att if role == ATTAQUANT else g_gar
            agent.update(observations[role], actions[role], actions[autre_role(role)], reward)

        self.history.append(record)
        return record

    def run_all(self):
        """Simulation complète (IA contre IA)."""
        while not self.finished:
            self.play_round()

    # Statistiques pour l'affichage
    def summary(self):
        n = len(self.history)
        buts = sum(r.resultat == "but" for r in self.history)
        return {
            "manches": n,
            "buts": buts,
            "arrets": n - buts,
            "taux_buts": buts / n if n else 0.0,
            "taux_arrets": (n - buts) / n if n else 0.0,
            "gain_moyen_attaquant": sum(r.gain_attaquant for r in self.history) / n if n else 0.0,
        }

    def human_gain(self):
        role = self.config.role_humain
        return sum(r.gain_attaquant if role == ATTAQUANT else r.gain_gardien for r in self.history)

    def frequencies(self, role):
        n = len(self.history)
        key = "action_attaquant" if role == ATTAQUANT else "action_gardien"
        counts = Counter(getattr(r, key) for r in self.history)
        return {a: (counts[a] / n if n else 0.0) for a in ACTIONS}

    def history_rows(self):
        return [
            {"Manche": r.numero, "Attaquant": r.action_attaquant, "Gardien": r.action_gardien,
             "Résultat": "But" if r.resultat == "but" else "Arrêt"}
            for r in self.history
        ]
