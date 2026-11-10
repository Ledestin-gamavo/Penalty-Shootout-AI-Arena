"""Tests de la couche interface (ui/match.py) branchée sur les vrais agents et game/."""
import itertools

import pytest

from agents.base_agent import Observation
from ui import compat
from ui.match import ATTAQUANT, GARDIEN, MODE_HUMAIN_IA, MODE_IA_IA, Match, MatchConfig

NIVEAUX = list(compat.LEVELS.values())


class Espion:
    """Faux agent qui enregistre ce qu'il reçoit (indices d'actions 0/1/2)."""

    def __init__(self, action):
        self.action, self.observations, self.updates, self.resets = action, [], [], 0

    def choose_action(self, observation):
        self.observations.append(observation)
        return self.action

    def update(self, observation, own_action, opponent_action, reward):
        self.updates.append((len(self.observations), own_action, opponent_action, reward))

    def reset(self):
        self.resets += 1


def match_humain(role, espion, n=3):
    ia = GARDIEN if role == ATTAQUANT else ATTAQUANT
    agents = {ATTAQUANT: None, GARDIEN: None}
    agents[ia] = "minimax"
    cfg = MatchConfig(MODE_HUMAIN_IA, n, agents, graine=1, role_humain=role)
    return Match(cfg, agents={ia: espion})


def test_observation_sans_action_adverse():
    assert set(Observation.__dataclass_fields__) == {"payoff", "round_index"}
    espion = Espion(1)
    m = match_humain(ATTAQUANT, espion)
    m.play_round("droite")
    m.play_round("gauche")
    assert [o.round_index for o in espion.observations] == [1, 2]


def test_update_apres_revelation_et_reset_au_debut():
    espion = Espion(0)  # le gardien plonge à gauche
    m = match_humain(ATTAQUANT, espion)
    assert espion.resets == 1
    m.play_round("gauche")  # tir arrêté : le gardien gagne +1
    assert espion.updates == [(1, 0, 0, 1.0)]
    m.play_round("droite")  # but : le gardien perd 1
    assert espion.updates[1] == (2, 0, 2, -1.0)


def test_resolution_conforme_a_la_matrice_de_reference():
    matrice = compat.reference_matrix()
    for i, j in itertools.product(range(3), repeat=2):
        res, ga, gg = compat.resolve_round(1, compat.ACTIONS[i], compat.ACTIONS[j], matrice)
        assert ga == matrice[i][j] and ga + gg == 0
        assert (res == "arret") == (i == j)


def test_action_invalide_rejetee():
    m = match_humain(GARDIEN, Espion(0))
    with pytest.raises(ValueError):
        m.play_round("haut")
    with pytest.raises(ValueError):
        m.play_round(None)


def test_partie_humaine_complete_contre_chaque_niveau_et_role():
    for niveau, role in itertools.product(NIVEAUX, (ATTAQUANT, GARDIEN)):
        ia = GARDIEN if role == ATTAQUANT else ATTAQUANT
        agents = {ATTAQUANT: None, GARDIEN: None}
        agents[ia] = niveau
        m = Match(MatchConfig(MODE_HUMAIN_IA, 6, agents, graine=5, role_humain=role))
        for k in range(6):
            m.play_round(compat.ACTIONS[k % 3])
        assert m.finished and len(m.history) == 6
        with pytest.raises(RuntimeError):
            m.play_round("gauche")


def test_ia_vs_ia_toutes_les_paires_nombre_exact_de_manches():
    for att, gar in itertools.product(NIVEAUX, NIVEAUX):
        m = Match(MatchConfig(MODE_IA_IA, 37, {ATTAQUANT: att, GARDIEN: gar}, graine=3))
        m.run_all()
        assert len(m.history) == 37


def test_reproductibilite_avec_graine():
    def run(seed):
        m = Match(MatchConfig(MODE_IA_IA, 60, {ATTAQUANT: "minimax", GARDIEN: "minimax"}, graine=seed))
        m.run_all()
        return m.history_rows()

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_le_niveau_deterministe_est_bien_deterministe():
    m = Match(MatchConfig(MODE_HUMAIN_IA, 5, {ATTAQUANT: "deterministe", GARDIEN: None},
                          graine=1, role_humain=GARDIEN))
    for _ in range(5):
        m.play_round("centre")
    assert {r.action_attaquant for r in m.history} == {"gauche"}
