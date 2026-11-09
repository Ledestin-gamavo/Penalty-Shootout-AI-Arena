"""Structures de données partagées par tout le projet.

Ces structures sont utilisées par le moteur, les agents, l'interface
Streamlit, les benchmarks et les visualisations. Elles sont immuables et ne
contiennent aucune logique métier, hormis des invariants et des propriétés
dérivées.

Dépendances autorisées : ``game.actions`` et ``game.constants``.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final, TypeAlias
from game.actions import Action
from game.constants import (
    DEFAULT_SEED,
    MAX_NUM_ROUNDS,
    MIN_NUM_ROUNDS,
    REWARD_GOAL,
)

#: Distribution de probabilités sur les actions, indexée par ``Action``.
Strategy: TypeAlias = tuple[float, ...]

#: Matrice des gains de l'attaquant : ``matrix[attaquant][gardien]``.
PayoffMatrix: TypeAlias = tuple[tuple[float, ...], ...]


class Role(StrEnum):
    """Rôle d'un joueur dans le duel.

    Attributes:
        ATTACKER: Le tireur, qui maximise son gain.
        GOALKEEPER: Le gardien, qui minimise le gain du tireur.
    """

    ATTACKER = "attacker"
    GOALKEEPER = "goalkeeper"

    @property
    def opponent(self) -> Role:
        """Retourne le rôle adverse.

        Returns:
            ``GOALKEEPER`` pour un attaquant, ``ATTACKER`` pour un gardien.
        """
        return Role.GOALKEEPER if self is Role.ATTACKER else Role.ATTACKER

    @property
    def label(self) -> str:
        """Retourne le libellé français du rôle.

        Returns:
            Le libellé destiné à l'interface et aux graphiques.
        """
        return _ROLE_LABELS[self]


class AgentType(StrEnum):
    """Type d'agent logiciel disponible dans le projet.

    Attributes:
        DETERMINISTIC: Suit une règle fixe, sans adaptation.
        FREQUENCY: Répond aux fréquences récentes de l'adversaire.
        MINIMAX: Joue la stratégie minimax (éventuellement mixte).
        FICTITIOUS_PLAY: Répond à la stratégie empirique cumulée adverse.
    """

    DETERMINISTIC = "deterministic"
    FREQUENCY = "frequency"
    MINIMAX = "minimax"
    FICTITIOUS_PLAY = "fictitious_play"

    @property
    def label(self) -> str:
        """Retourne le libellé français du type d'agent.

        Returns:
            Le libellé destiné à l'interface et aux graphiques.
        """
        return _AGENT_LABELS[self]


_ROLE_LABELS: Final[dict[Role, str]] = {
    Role.ATTACKER: "Attaquant",
    Role.GOALKEEPER: "Gardien",
}

_AGENT_LABELS: Final[dict[AgentType, str]] = {
    AgentType.DETERMINISTIC: "Déterministe",
    AgentType.FREQUENCY: "Fréquences",
    AgentType.MINIMAX: "Minimax",
    AgentType.FICTITIOUS_PLAY: "Fictitious Play",
}


@dataclass(frozen=True, slots=True)
class Observation:
    """Information autorisée donnée à un agent avant qu'il ne décide.

    Cette structure ne contient volontairement aucune information sur
    l'action courante de l'adversaire. L'historique des manches terminées
    parvient à l'agent uniquement via ``BaseAgent.update``.

    Attributes:
        role: Rôle tenu par l'agent pour cette manche.
        round_number: Numéro de la manche en cours, à partir de 1.
    """

    role: Role
    round_number: int


@dataclass(frozen=True, slots=True)
class RoundResult:
    """Résultat d'une manche après révélation des deux actions.

    Le gain du gardien et l'indicateur de but sont dérivés du gain de
    l'attaquant, ce qui garantit la somme nulle par construction.

    Attributes:
        round_number: Numéro de la manche, à partir de 1.
        attacker_action: Zone choisie par l'attaquant.
        goalkeeper_action: Zone choisie par le gardien.
        attacker_reward: Gain réalisé par l'attaquant pour cette manche.
    """

    round_number: int
    attacker_action: Action
    goalkeeper_action: Action
    attacker_reward: float

    @property
    def goalkeeper_reward(self) -> float:
        """Retourne le gain du gardien (opposé de celui de l'attaquant).

        Returns:
            Le gain du gardien.
        """
        return -self.attacker_reward

    @property
    def is_goal(self) -> bool:
        """Indique si un but a été marqué.

        Returns:
            ``True`` si le gain de l'attaquant correspond à un but.
        """
        return self.attacker_reward == REWARD_GOAL

    def action_of(self, role: Role) -> Action:
        """Retourne l'action jouée par un rôle.

        Args:
            role: Rôle dont on souhaite l'action.

        Returns:
            L'action jouée par ce rôle lors de la manche.
        """
        if role is Role.ATTACKER:
            return self.attacker_action
        return self.goalkeeper_action

    def reward_for(self, role: Role) -> float:
        """Retourne le gain obtenu par un rôle.

        Args:
            role: Rôle dont on souhaite le gain.

        Returns:
            Le gain de ce rôle lors de la manche.
        """
        if role is Role.ATTACKER:
            return self.attacker_reward
        return self.goalkeeper_reward


@dataclass(frozen=True, slots=True)
class MatchConfig:
    """Paramètres d'un match ou d'une expérience.

    Un agent égal à ``None`` désigne le joueur humain. Le mode de jeu se
    déduit donc de la configuration : humain contre IA si l'un des deux
    agents est ``None``, IA contre IA sinon.

    Attributes:
        num_rounds: Nombre de manches à jouer.
        attacker_agent: Agent attaquant, ou ``None`` si l'humain attaque.
        goalkeeper_agent: Agent gardien, ou ``None`` si l'humain garde.
        seed: Graine de reproductibilité, entier positif ou nul.
        payoff_matrix: Matrice des gains personnalisée, ou ``None`` pour
            utiliser la matrice de référence.

    Raises:
        ValueError: Si le nombre de manches est hors bornes, si la graine
            est négative, ou si aucun des deux joueurs n'est un agent
            (le mode humain contre humain est hors périmètre).
    """

    num_rounds: int
    attacker_agent: AgentType | None
    goalkeeper_agent: AgentType | None
    seed: int = DEFAULT_SEED
    payoff_matrix: PayoffMatrix | None = None

    def __post_init__(self) -> None:
        """Vérifie les invariants de la configuration.

        Raises:
            ValueError: Si un invariant n'est pas respecté.
        """
        if not MIN_NUM_ROUNDS <= self.num_rounds <= MAX_NUM_ROUNDS:
            raise ValueError(
                f"num_rounds doit être compris entre {MIN_NUM_ROUNDS} et "
                f"{MAX_NUM_ROUNDS}, reçu : {self.num_rounds}."
            )
        if self.seed < 0:
            raise ValueError(f"seed doit être positif ou nul, reçu : {self.seed}.")
        if self.attacker_agent is None and self.goalkeeper_agent is None:
            raise ValueError(
                "Au moins un des deux joueurs doit être un agent "
                "(mode humain contre humain hors périmètre)."
            )

    @property
    def human_role(self) -> Role | None:
        """Retourne le rôle tenu par l'humain, s'il y en a un.

        Returns:
            Le rôle de l'humain, ou ``None`` pour un match IA contre IA.
        """
        if self.attacker_agent is None:
            return Role.ATTACKER
        if self.goalkeeper_agent is None:
            return Role.GOALKEEPER
        return None


@dataclass(frozen=True, slots=True)
class AgentStats:
    """Statistiques agrégées d'un agent sur un match.

    Les « buts » et « arrêts » concernent les manches jouées par l'agent,
    quel que soit son rôle : pour un gardien, ``goals`` correspond aux buts
    concédés. Les définitions de ``equilibrium_gap`` et ``exploitability``
    sont données dans ``theory/metrics.py`` et ``docs/game_theory.md``.

    Attributes:
        agent_type: Type de l'agent évalué.
        role: Rôle tenu par l'agent.
        rounds: Nombre de manches jouées (au moins 1).
        goals: Nombre de manches terminées par un but.
        saves: Nombre de manches terminées par un arrêt.
        mean_reward: Gain moyen de l'agent par manche, selon son rôle.
        mean_decision_time: Temps moyen de décision, en secondes.
        empirical_strategy: Fréquences observées des actions jouées.
        equilibrium_gap: Écart entre la stratégie empirique et la
            stratégie d'équilibre de référence.
        exploitability: Gain additionnel qu'une meilleure réponse
            obtiendrait contre la stratégie empirique.

    Raises:
        ValueError: Si ``rounds`` est inférieur à 1 ou si
            ``goals + saves`` diffère de ``rounds``.
    """

    agent_type: AgentType
    role: Role
    rounds: int
    goals: int
    saves: int
    mean_reward: float
    mean_decision_time: float
    empirical_strategy: Strategy
    equilibrium_gap: float
    exploitability: float

    def __post_init__(self) -> None:
        """Vérifie la cohérence des compteurs.

        Raises:
            ValueError: Si un invariant n'est pas respecté.
        """
        if self.rounds < 1:
            raise ValueError(f"rounds doit être au moins 1, reçu : {self.rounds}.")
        if self.goals + self.saves != self.rounds:
            raise ValueError(
                f"goals ({self.goals}) + saves ({self.saves}) "
                f"doit valoir rounds ({self.rounds})."
            )

    @property
    def goal_rate(self) -> float:
        """Retourne le taux de buts (buts / manches jouées).

        Returns:
            Le taux de buts, entre 0 et 1.
        """
        return self.goals / self.rounds

    @property
    def save_rate(self) -> float:
        """Retourne le taux d'arrêts (arrêts / manches jouées).

        Returns:
            Le taux d'arrêts, entre 0 et 1.
        """
        return self.saves / self.rounds


@dataclass(frozen=True, slots=True)
class Experiment:
    """Résultat complet et reproductible d'un match IA contre IA.

    Les parties humain contre IA ne produisent pas d'``Experiment`` : leur
    historique est une simple séquence de ``RoundResult``.

    Attributes:
        config: Configuration utilisée (contient la graine).
        created_at: Date de création, en UTC avec fuseau horaire.
        rounds: Résultats de toutes les manches, dans l'ordre.
        attacker_stats: Statistiques de l'agent attaquant.
        goalkeeper_stats: Statistiques de l'agent gardien.

    Raises:
        ValueError: Si la configuration contient un joueur humain ou si le
            nombre de manches diffère de ``config.num_rounds``.
    """

    config: MatchConfig
    created_at: datetime
    rounds: tuple[RoundResult, ...]
    attacker_stats: AgentStats
    goalkeeper_stats: AgentStats

    def __post_init__(self) -> None:
        """Vérifie la cohérence de l'expérience.

        Raises:
            ValueError: Si un invariant n'est pas respecté.
        """
        if self.config.human_role is not None:
            raise ValueError("Une expérience concerne uniquement des agents.")
        if len(self.rounds) != self.config.num_rounds:
            raise ValueError(
                f"{len(self.rounds)} manches reçues, "
                f"{self.config.num_rounds} attendues."
            )


@dataclass(frozen=True, slots=True)
class Equilibrium:
    """Solution minimax d'un jeu matriciel à somme nulle.

    Attributes:
        attacker_strategy: Stratégie optimale de l'attaquant.
        goalkeeper_strategy: Stratégie optimale du gardien.
        value: Valeur du jeu, du point de vue de l'attaquant.
    """

    attacker_strategy: Strategy
    goalkeeper_strategy: Strategy
    value: float