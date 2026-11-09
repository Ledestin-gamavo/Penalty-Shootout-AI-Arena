import pytest

from ui import compat
from ui.match import (ATTAQUANT, GARDIEN, MODE_HUMAIN_IA, MODE_IA_IA, Match, MatchConfig)


class Espion:
    """Agent qui enregistre ce qu'il voit."""

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


def test_gains_somme_nulle_et_matrice():
    for a in compat.ACTIONS:
        for g in compat.ACTIONS:
            res, ga, gg = compat.resolve_round(a, g)
            assert ga + gg == 0
            assert (res == "arret") == (a == g)


def test_action_invalide_rejetee():
    m = match_humain(GARDIEN, Espion("gauche"))
    with pytest.raises(ValueError):
        m.play_round("haut")
    with pytest.raises(ValueError):
        m.play_round(None)


def test_les_deux_roles_se_terminent():
    for role in (ATTAQUANT, GARDIEN):
        m = match_humain(role, Espion("centre"), n=4)
        for _ in range(4):
            m.play_round("gauche")
        assert m.finished
        with pytest.raises(RuntimeError):
            m.play_round("gauche")


def test_ia_vs_ia_nombre_exact_de_manches():
    for name in compat.LEVELS.values():
        cfg = MatchConfig(MODE_IA_IA, 37, {ATTAQUANT: name, GARDIEN: "minimax"}, graine=3)
        m = Match(cfg)
        m.run_all()
        assert len(m.history) == 37


def test_reproductibilite_avec_graine():
    def run():
        m = Match(MatchConfig(MODE_IA_IA, 50, {ATTAQUANT: "frequences", GARDIEN: "minimax"}, graine=7))
        m.run_all()
        return m.history_rows()

    assert run() == run()


def test_les_quatre_agents_respectent_le_contrat():
    for name in compat.LEVELS.values():
        agent = compat.make_agent(name, seed=0)
        obs = {"role": ATTAQUANT, "round": 1, "total_rounds": 3, "history": []}
        assert agent.choose_action(obs) in compat.ACTIONS
        agent.update(obs, "gauche", "centre", 1)
        agent.reset()
