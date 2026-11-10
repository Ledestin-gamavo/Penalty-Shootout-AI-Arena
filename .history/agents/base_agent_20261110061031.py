"""Contrat commun des agents (cahier des charges, section 8.2).

Conventions (à partager avec les 3 autres membres) :
  * Actions : entiers 0 = gauche, 1 = centre, 2 = droite (pour les deux rôles).
  * Matrice de gains `payoff` : toujours du point de vue de l'ATTAQUANT.
        payoff[i, j] = gain de l'attaquant si le tir est i et le plongeon j.
    L'attaquant maximise ; le gardien minimise (son gain est -payoff).
  * Le rôle de l'agent ("attacker" ou "goalkeeper") est fixé à la création.
"""
from dataclasses import dataclass
import numpy as np

ACTIONS = (0, 1, 2)
ACTION_NAMES = ("gauche", "centre", "droite")
ROLES = ("attacker", "goalkeeper")


@dataclass(frozen=True, eq=False)
class Observation:
    """Seules informations données à un agent avant de choisir.

    Aucun champ ne contient l'action courante de l'adversaire (exigence EF04).
    La matrice est COPIÉE puis verrouillée en lecture seule : ni un agent ni
    l'interface ne peut modifier par erreur la matrice du moteur. Égalité et
    hachage se font sur les valeurs (utile pour les caches et les tests).
    """
    payoff: np.ndarray      # matrice 3x3, gains de l'attaquant
    round_index: int = 0    # numéro de la manche 

    def __post_init__(self):
        p = np.array(self.payoff, dtype=float, copy=True)     
        if p.shape != (3, 3):
            raise ValueError(f"payoff doit être une matrice 3x3, reçu {p.shape}")
        p.setflags(write=False)                               # lecture seule
        object.__setattr__(self, "payoff", p)                 # nécessaire car la classe est « frozen »

    def __eq__(self, other):
        return (isinstance(other, Observation) and self.round_index == other.round_index
                and np.array_equal(self.payoff, other.payoff))

    def __hash__(self):
        return hash((self.round_index, self.payoff.tobytes()))


def own_utility(payoff, role):
    """Matrice d'utilité vue par l'agent : lignes = ses actions, colonnes = celles
    de l'adversaire, et l'agent MAXIMISE toujours cette matrice.

    Attaquant : U = payoff.   Gardien : U = -payoff.T  (on transpose et on change le signe).
    Ainsi le même code (meilleure réponse, programmation linéaire) sert aux deux rôles.
    """
    payoff = np.asarray(payoff, dtype=float)
    return payoff if role == "attacker" else -payoff.T


def best_response_policy(utility, opponent_dist, tol=1e-9):
    """Politique de meilleure réponse à une distribution adverse estimée.

    On calcule le gain espéré de chaque action propre (utility @ q) puis on joue
    avec une probabilité égale toutes les actions optimales. Quand plusieurs
    actions sont à égalité (cas fréquent : le jeu de base est symétrique), le
    tirage au sort départage, au lieu de toujours prendre la première.
    """
    values = np.asarray(utility, dtype=float) @ np.asarray(opponent_dist, dtype=float)
    best = values >= values.max() - tol
    policy = best.astype(float)
    return policy / policy.sum()


class BaseAgent:
    """Classe mère. Les sous-classes définissent policy() et, si besoin, _observe()."""
    name = "base"

    def __init__(self, role, seed=None):
        if role not in ROLES:
            raise ValueError(f"role doit être dans {ROLES}, reçu : {role!r}")
        self.role = role
        self.seed = seed
        self.reset()

    #C'est une convention que j'ai prise mais qu'on pourra mpdofier lors de l'intégration avec le moteur. L'agent ne doit pas avoir accès à l'action de l'adversaire avant de choisir la sienne.
    def policy(self, observation):
        """Retourne un vecteur de 3 probabilités (>= 0, somme 1). Ne doit utiliser
        que l'état interne (mis à jour par update) et l'observation."""
        raise NotImplementedError

    def _observe(self, own_action, opponent_action):
        """Mise à jour de la mémoire spécifique à l'agent (optionnel)."""

    # contrat du cahier des charges
    def choose_action(self, observation):
        """Tire une action selon policy(). Retourne un entier 0, 1 ou 2."""
        p = np.asarray(self.policy(observation), dtype=float)
        if p.shape != (3,) or (p < -1e-12).any() or abs(p.sum() - 1.0) > 1e-9:
            raise ValueError(f"politique invalide : {p}")
        p = np.clip(p, 0.0, None)
        return int(self.rng.choice(3, p=p / p.sum()))

    def update(self, observation, own_action, opponent_action, reward):
        """Appelé APRÈS la révélation des deux actions de la manche."""
        for a in (own_action, opponent_action):
            if a not in ACTIONS:
                raise ValueError(f"action invalide : {a!r}")
        self.t += 1
        self._observe(own_action, opponent_action)

    def reset(self):
        """Efface la mémoire et réinitialise le générateur aléatoire (reproductibilité)."""
        self.rng = np.random.default_rng(self.seed)
        self.t = 0
