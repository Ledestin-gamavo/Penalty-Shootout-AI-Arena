"""Tests de la logique de partie (ui/match.py) avec des agents espions."""
import itertools

import pytest

from ui import compat
from ui.match import ATTAQUANT, GARDIEN, MODE_HUMAIN_IA, MODE_IA_IA, Match, MatchConfig

NIVEAUX = list(compat.LEVELS.values())


class Espion:
    """Faux agent qui enregistre ce qu'il reçoit (actions sous forme de texte)."""

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


def test_agent_ne_voit_pas_l_action_courante():
    espion = Espion("centre")
    m = match_humain(ATTAQUANT, espion)
    m.play_round("droite")
    assert espion.observations[0]["history"] == []
    m.play_round("gauche")
    assert espion.observations[1]["history"] == [("droite", "centre")]


def test_update_apres_revelation_et_reset_au_debut():
    espion = Espion("gauche")
    m = match_humain(ATTAQUANT, espion)
    assert espion.resets == 1
    m.play_round("gauche")  # tir arrêté : le gardien gagne +1
    assert espion.updates == [(1, "gauche", "gauche", 1)]
    m.play_round("droite")  # but : le gardien perd 1
    assert espion.updates[1] == (2, "gauche", "droite", -1)


def test_gains_somme_nulle_et_arret_si_meme_zone():
    for a, g in itertools.product(compat.ACTIONS, repeat=2):
        res, ga, gg = compat.resolve_round(a, g)
        assert ga + gg == 0
        assert (res == "arret") == (a == g)


def test_action_invalide_rejetee():
    m = match_humain(GARDIEN, Espion("gauche"))
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


def test_ia_vs_ia_nombre_exact_de_manches():
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
    assert len({r.action_attaquant for r in m.history}) == 1
