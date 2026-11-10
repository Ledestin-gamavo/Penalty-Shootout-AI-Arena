"""Logique d'une partie humain contre IA — sans Streamlit, donc testable.

Garantie centrale (EF04, décisions simultanées) : dans ``play_round``, chaque
agent choisit à partir de l'historique PUBLIC des manches terminées
uniquement ; les actions sont ensuite révélées ensemble, puis seulement
``update()`` est appelé.

Les simulations IA contre IA de l'interface passent par
``game.simulation.run_match`` (voir ``ui/play_page.py``) : ce module ne sert
plus qu'aux parties où l'humain joue.
"""

import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from . import compat
from .compat import UiAgent, UiChoice, UiObservation

ACTIONS = compat.ACTIONS
ATTAQUANT = "attaquant"
GARDIEN = "gardien"
ROLES = (ATTAQUANT, GARDIEN)
MODE_HUMAIN_IA = "humain_vs_ia"
MODE_IA_IA = "ia_vs_ia"


def autre_role(role: str) -> str:
    """Retourne le rôle adverse.

    Args:
        role: ``"attaquant"`` ou ``"gardien"``.

    Returns:
        L'autre rôle.
    """
    return GARDIEN if role == ATTAQUANT else ATTAQUANT


@dataclass(frozen=True)
class RoundResult:
    """Résultat d'une manche de l'interface."""

    numero: int
    action_attaquant: str
    action_gardien: str
    resultat: str  # "but" ou "arret"
    gain_attaquant: float
    gain_gardien: float


@dataclass
class MatchConfig:
    """Configuration d'une partie de l'interface.

    ``agents`` associe à chaque rôle un nom d'agent, ou ``None`` si le rôle
    est tenu par l'humain.

    Raises:
        ValueError: Si le nombre de manches est inférieur à 1 ou si le rôle
            humain manque en mode humain contre IA.
    """

    mode: str
    nombre_manches: int
    agents: dict[str, str | None]
    graine: int = 0
    role_humain: str | None = None

    def __post_init__(self) -> None:
        """Vérifie les invariants de la configuration."""
        if self.nombre_manches < 1:
            raise ValueError("Il faut au moins une manche.")
        if self.mode == MODE_HUMAIN_IA and self.role_humain not in ROLES:
            raise ValueError("role_humain requis en mode Humain contre IA.")


class Match:
    """Partie en cours : historique, agents et déroulement manche par manche."""

    def __init__(
        self, config: MatchConfig, agents: Mapping[str, UiAgent] | None = None
    ) -> None:
        """Prépare la partie.

        Args:
            config: Configuration de la partie.
            agents: Agents à injecter (tests) ; sinon ils sont créés via
                ``compat.make_agent``.
        """
        self.config = config
        self.history: list[RoundResult] = []
        self._rng = random.Random(config.graine)
        self.agents: dict[str, UiAgent] = {}
        if agents is not None:
            self.agents = dict(agents)
        else:
            for i, role in enumerate(ROLES):
                name = config.agents.get(role)
                if name is not None:
                    self.agents[role] = compat.make_agent(name, seed=config.graine + i)
        for agent in self.agents.values():
            agent.reset()

    @property
    def finished(self) -> bool:
        """Indique si toutes les manches ont été jouées."""
        return len(self.history) >= self.config.nombre_manches

    def _observation(self, role: str) -> UiObservation:
        """Construit l'observation publique d'un rôle avant sa décision.

        Args:
            role: Rôle qui va décider.

        Returns:
            Le rôle, le numéro de manche et l'historique des manches terminées.
        """
        return {
            "role": role,
            "round": len(self.history) + 1,
            "total_rounds": self.config.nombre_manches,
            "history": [(r.action_attaquant, r.action_gardien) for r in self.history],
        }

    def _to_action(self, choice: UiChoice) -> str:
        """Accepte une action, ou une politique (dict ou séquence de 3 probabilités).

        Args:
            choice: Décision d'un agent.

        Returns:
            Le nom de l'action jouée.

        Raises:
            ValueError: Si l'action n'est pas valide.
        """
        if isinstance(choice, str):
            action = choice
        else:
            weights = (
                [choice[a] for a in ACTIONS]
                if isinstance(choice, Mapping)
                else list(choice)
            )
            action = self._rng.choices(ACTIONS, weights=weights)[0]
        if action not in ACTIONS:
            raise ValueError(f"Action invalide : {action!r}")
        return action

    def play_round(self, human_action: str | None = None) -> RoundResult:
        """Joue une manche.

        Args:
            human_action: Action de l'humain (obligatoire en mode humain
                contre IA).

        Returns:
            Le résultat de la manche.

        Raises:
            RuntimeError: Si la partie est terminée.
            ValueError: Si l'action humaine manque ou est invalide.
        """
        if self.finished:
            raise RuntimeError("La partie est terminée.")
        role_h = self.config.role_humain if self.config.mode == MODE_HUMAIN_IA else None

        # 1. Décisions simultanées : historique public uniquement
        observations: dict[str, UiObservation] = {}
        actions: dict[str, str] = {}
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
        resultat, g_att, g_gar = compat.resolve_round(
            actions[ATTAQUANT], actions[GARDIEN]
        )
        record = RoundResult(
            len(self.history) + 1,
            actions[ATTAQUANT],
            actions[GARDIEN],
            resultat,
            g_att,
            g_gar,
        )
        # 3. Mise à jour des agents APRÈS révélation
        for role, agent in self.agents.items():
            reward = g_att if role == ATTAQUANT else g_gar
            agent.update(
                observations[role], actions[role], actions[autre_role(role)], reward
            )


        self.history.append(record)
        return record

    def run_all(self) -> None:
        """Joue toutes les manches restantes (agents seulement)."""
        while not self.finished:
            self.play_round()

    def summary(self) -> dict[str, float]:
        """Résume les manches jouées (buts, arrêts, taux, gain moyen).

        Returns:
            Les statistiques d'affichage.
        """
        n = len(self.history)
        buts = sum(r.resultat == "but" for r in self.history)
        return {
            "manches": n,
            "buts": buts,
            "arrets": n - buts,
            "taux_buts": buts / n if n else 0.0,
            "taux_arrets": (n - buts) / n if n else 0.0,
            "gain_moyen_attaquant": (
                sum(r.gain_attaquant for r in self.history) / n if n else 0.0
            ),
        }

    def human_gain(self) -> int:
        """Retourne le gain cumulé du joueur humain.

        Returns:
            La somme des gains du rôle tenu par l'humain.
        """
        role = self.config.role_humain
        return sum(
            r.gain_attaquant if role == ATTAQUANT else r.gain_gardien
            for r in self.history
        )

    def frequencies(self, role: str) -> dict[str, float]:
        """Calcule la part de chaque zone jouée par un rôle.

        Args:
            role: Rôle étudié.

        Returns:
            Les fréquences par zone (0 si aucune manche).
        """
        n = len(self.history)
        key = "action_attaquant" if role == ATTAQUANT else "action_gardien"
        counts = Counter(getattr(r, key) for r in self.history)
        return {a: (counts[a] / n if n else 0.0) for a in ACTIONS}

    def history_rows(self) -> list[dict[str, object]]:
        """Met l'historique sous forme de lignes d'affichage.

        Returns:
            Une ligne par manche jouée.
        """
        return [
            {
                "Manche": r.numero,
                "Attaquant": r.action_attaquant,
                "Gardien": r.action_gardien,
                "Résultat": "But" if r.resultat == "but" else "Arrêt",
            }
            for r in self.history
        ]
