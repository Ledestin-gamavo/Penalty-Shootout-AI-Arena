"""Tests du module agents. Lancer depuis la racine du projet : python -m pytest tests -v"""
import numpy as np
import pytest
from agents import (AGENTS, ACTIONS, Observation, DeterministicAgent, FrequencyAgent,
                    FictitiousPlayAgent, MinimaxAgent, own_utility, solve_maximin)

BASE = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float)  # gains de l'attaquant
OBS = Observation(payoff=BASE)
ROLES = ("attacker", "goalkeeper")


def make(name, role, **kw):
    return AGENTS[name](role, seed=0, **kw)


def play_round(att, keeper, payoff=BASE, t=0):
    """Une manche simultanée : les deux choisissent AVANT de voir l'autre."""
    obs = Observation(payoff=payoff, round_index=t)
    a, g = att.choose_action(obs), keeper.choose_action(obs)
    r = payoff[a, g]
    att.update(obs, a, g, r)          # révélation : on met à jour après les deux choix
    keeper.update(obs, g, a, -r)
    return a, g, r


# ---------- validité générale ----------
@pytest.mark.parametrize("name", list(AGENTS))
@pytest.mark.parametrize("role", ROLES)
def test_policy_valide_et_action_valide(name, role):
    ag = make(name, role)
    for _ in range(20):
        p = ag.policy(OBS)
        assert p.shape == (3,) and (p >= 0).all() and abs(p.sum() - 1) < 1e-9
        a = ag.choose_action(OBS)
        assert a in ACTIONS
        ag.update(OBS, a, int(np.random.default_rng(1).integers(3)), 0.0)


def test_role_invalide_et_action_invalide():
    with pytest.raises(ValueError):
        DeterministicAgent("arbitre")
    ag = make("fictitious_play", "attacker")
    with pytest.raises(ValueError):
        ag.update(OBS, 0, 5, 0.0)


# ---------- simultanéité (pas de fuite d'information) ----------
def test_observation_sans_action_adverse():
    champs = set(Observation.__dataclass_fields__)
    assert champs == {"payoff", "round_index"}      # aucun champ « action adverse »


@pytest.mark.parametrize("name", list(AGENTS))
def test_politique_inchangee_sans_update(name):
    ag = make(name, "goalkeeper")
    p1 = ag.policy(OBS).copy()
    ag.choose_action(OBS); ag.choose_action(OBS)    # choisir ne doit rien apprendre
    assert np.allclose(p1, ag.policy(OBS))


@pytest.mark.parametrize("name", ["frequences", "fictitious_play"])
def test_adaptatif_change_seulement_apres_update(name):
    ag = make(name, "goalkeeper")
    avant = ag.policy(OBS).copy()
    for _ in range(5):
        ag.update(OBS, 1, 0, 0.0)                   # l'attaquant a tiré 5 fois à gauche
    assert not np.allclose(avant, ag.policy(OBS))


# ---------- agent déterministe ----------
def test_deterministe_toujours_gauche_puis_sequence():
    ag = DeterministicAgent("attacker", pattern=(0,), seed=0)
    assert all(ag.choose_action(OBS) == 0 for _ in range(10))
    seq = DeterministicAgent("attacker", pattern=(0, 1, 2), seed=0)
    out = []
    for _ in range(6):
        out.append(seq.choose_action(OBS)); seq.update(OBS, out[-1], 0, 0.0)
    assert out == [0, 1, 2, 0, 1, 2]


def test_deterministe_est_exploite():
    """Un gardien Fictitious Play finit par arrêter presque tous les tirs d'un attaquant fixe."""
    att, keeper = DeterministicAgent("attacker", seed=0), FictitiousPlayAgent("goalkeeper", seed=0)
    gains = [play_round(att, keeper)[2] for _ in range(200)]
    assert np.mean(gains[50:]) < -0.9               # presque tout arrêté


# ---------- fréquences vs Fictitious Play ----------
def test_meilleure_reponse_attaquant_et_gardien():
    freq_a = FrequencyAgent("attacker", seed=0)
    for _ in range(10):
        freq_a.update(OBS, 1, 0, 0.0)               # gardien plonge toujours à gauche
    p = freq_a.policy(OBS)
    assert p[0] == 0 and abs(p[1] - 0.5) < 1e-9 and abs(p[2] - 0.5) < 1e-9   # centre ou droite
    freq_g = FrequencyAgent("goalkeeper", seed=0)
    for _ in range(10):
        freq_g.update(OBS, 0, 0, 0.0)               # attaquant tire toujours à gauche
    assert np.allclose(freq_g.policy(OBS), [1, 0, 0])


def test_difference_fenetre_vs_cumul():
    """100 tirs à gauche puis 20 à droite : la fenêtre (10) s'adapte, le cumul non."""
    fen = FrequencyAgent("goalkeeper", window=10, seed=0)
    fp = FictitiousPlayAgent("goalkeeper", seed=0)
    for a in [0] * 100 + [2] * 20:
        fen.update(OBS, 1, a, 0.0); fp.update(OBS, 1, a, 0.0)
    assert np.allclose(fen.policy(OBS), [0, 0, 1])  # plonge à droite (récent)
    assert np.allclose(fp.policy(OBS), [1, 0, 0])   # plonge à gauche (historique)


def test_fenetre_limitee():
    fen = FrequencyAgent("goalkeeper", window=3, seed=0)
    for a in [0, 0, 0, 1, 1, 1]:
        fen.update(OBS, 0, a, 0.0)
    assert len(fen.recent) == 3 and list(fen.recent) == [1, 1, 1]


# ---------- Minimax ----------
@pytest.mark.parametrize("role", ROLES)
def test_minimax_uniforme_sur_matrice_de_base(role):
    ag = MinimaxAgent(role, seed=0)
    assert np.allclose(ag.policy(OBS), [1 / 3] * 3, atol=1e-6)
    v = ag.game_value(OBS)
    assert abs(v - (1 / 3 if role == "attacker" else -1 / 3)) < 1e-6


def test_minimax_certificat_d_optimalite_matrice_asymetrique():
    """Matrice A = 2p - 1 asymétrique : la garantie de l'attaquant égale celle du gardien (dualité forte)."""
    rng = np.random.default_rng(42)
    A = 2 * rng.uniform(0.3, 0.95, size=(3, 3)) - 1
    obs = Observation(payoff=A)
    x = MinimaxAgent("attacker").policy(obs)        # attaquant
    y = MinimaxAgent("goalkeeper").policy(obs)      # gardien
    garantie_att = (x @ A).min()                    # ce que l'attaquant assure au pire
    plafond_gar = (A @ y).max()                     # ce que le gardien concède au plus
    assert abs(garantie_att - plafond_gar) < 1e-7


def test_solve_maximin_jeu_2x2_connu():
    """Jeu du cours : A = [[3,1],[2,4]] -> x* = (1/2,1/2), v = 5/2."""
    x, v = solve_maximin(np.array([[3, 1], [2, 4]]))
    assert np.allclose(x, [0.5, 0.5], atol=1e-7) and abs(v - 2.5) < 1e-7


def test_minimax_ne_depend_pas_de_l_historique():
    ag = MinimaxAgent("attacker", seed=0)
    p0 = ag.policy(OBS).copy()
    for _ in range(30):
        ag.update(OBS, 0, 0, 0.0)
    assert np.allclose(p0, ag.policy(OBS))


# ---------- Fictitious Play en auto-jeu ----------
def test_fictitious_play_convergence_des_frequences():
    att, keeper = FictitiousPlayAgent("attacker", seed=1), FictitiousPlayAgent("goalkeeper", seed=2)
    na, ng = np.zeros(3), np.zeros(3)
    for t in range(4000):
        a, g, _ = play_round(att, keeper, t=t)
        na[a] += 1; ng[g] += 1
    assert np.allclose(na / na.sum(), 1 / 3, atol=0.05)
    assert np.allclose(ng / ng.sum(), 1 / 3, atol=0.05)


# ---------- reset et reproductibilité ----------
@pytest.mark.parametrize("name", list(AGENTS))
def test_reset_efface_la_memoire(name):
    ag = make(name, "goalkeeper")
    neuf = make(name, "goalkeeper").policy(OBS)
    for _ in range(15):
        ag.update(OBS, 1, 0, 0.0)
    ag.reset()
    assert ag.t == 0 and np.allclose(ag.policy(OBS), neuf)


def test_reproductibilite_meme_graine():
    def sequence(seed):
        ag = MinimaxAgent("attacker", seed=seed)
        return [ag.choose_action(OBS) for _ in range(60)]
    assert sequence(7) == sequence(7)
    assert sequence(7) != sequence(8)
    ag = MinimaxAgent("attacker", seed=7); s1 = [ag.choose_action(OBS) for _ in range(20)]
    ag.reset(); s2 = [ag.choose_action(OBS) for _ in range(20)]
    assert s1 == s2                                 # reset remet la graine


def test_own_utility_gardien_est_oppose_transpose():
    assert np.allclose(own_utility(BASE, "goalkeeper"), -BASE.T)
    A = np.arange(9.0).reshape(3, 3)
    assert np.allclose(own_utility(A, "goalkeeper"), -A.T)

# ---------- Observation : copie, lecture seule, égalité, hachage ----------
def test_observation_copie_en_lecture_seule():
    A = BASE.copy()
    obs = Observation(payoff=A)
    A[0, 0] = 99.0
    assert obs.payoff[0, 0] == -1.0                 # copie : le moteur ne modifie plus l'observation
    with pytest.raises(ValueError):
        obs.payoff[0, 0] = 5.0                      # lecture seule


def test_observation_egalite_et_hachage():
    o1, o2 = Observation(BASE, 3), Observation(BASE.copy(), 3)
    assert o1 == o2 and hash(o1) == hash(o2)
    assert o1 != Observation(BASE, 4)
    assert len({o1, o2}) == 1


def test_observation_forme_invalide():
    with pytest.raises(ValueError):
        Observation(payoff=np.zeros((2, 2)))