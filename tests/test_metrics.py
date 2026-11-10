"""Tests de ``theory.metrics`` : indicateurs de performance des agents.

Organisation du fichier :

* ``empirical_strategy`` ;
* ``goal_rate`` et ``save_rate`` ;
* ``average_reward`` ;
* ``distance_to_equilibrium`` ;
* ``exploitability`` ;
* ``compute_agent_stats`` (cohérence avec les fonctions élémentaires) ;
* robustesse numérique (``PROBABILITY_TOLERANCE``,
  ``NUMERICAL_NOISE_TOLERANCE``, ``EQUILIBRIUM_TOLERANCE``) ;
* limite connue de l'hypothèse de gains ±1 (``docs/game_theory.md``, §7).

Les manches sont fabriquées avec ``game.rules.resolve_round``, seul endroit
du projet où un ``RoundResult`` est construit à partir de deux actions. Les
valeurs attendues sont calculées à la main (détail en commentaire) et non
recopiées du code testé.
"""

import math
from collections.abc import Callable, Sequence
from itertools import permutations, product
from typing import Final

import numpy as np
import pytest

from game.actions import Action
from game.constants import (
    EQUILIBRIUM_TOLERANCE,
    N_ACTIONS,
    NUMERICAL_NOISE_TOLERANCE,
    PROBABILITY_TOLERANCE,
)
from game.models import (
    AgentStats,
    AgentType,
    PayoffMatrix,
    Role,
    RoundResult,
    Strategy,
)
from game.payoff import build_reference_matrix
from game.rules import resolve_round
from theory.equilibrium import best_response, solve_equilibrium
from theory.metrics import (
    average_reward,
    compute_agent_stats,
    distance_to_equilibrium,
    empirical_strategy,
    exploitability,
    goal_rate,
    save_rate,
)
from theory.validation import validate_strategy

NAN: Final[float] = float("nan")
INF: Final[float] = float("inf")
LEFT, CENTER, RIGHT = Action.LEFT, Action.CENTER, Action.RIGHT

REFERENCE: Final[PayoffMatrix] = build_reference_matrix()

# Stratégie uniforme écrite indépendamment des constantes du projet.
UNIFORM: Final[Strategy] = (1 / 3,) * N_ACTIONS

# Matrice symétrique à équilibre mixte non uniforme, calculé à la main :
# l'attaquant égalise ses gains contre chaque colonne avec p_i proportionnel à
# 1/d_i, où d = (2, 1, 1) -> p = (0.2, 0.4, 0.4) et v = 0.6 (idem pour le
# gardien, la matrice étant symétrique).
BIASED_MATRIX: Final[PayoffMatrix] = (
    (-1.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
)
BIASED_STRATEGY: Final[Strategy] = (0.2, 0.4, 0.4)
BIASED_VALUE: Final[float] = 0.6

# Même jeu, colonnes permutées (1, 2, 0) : la stratégie de l'attaquant ne change
# pas, celle du gardien est permutée. Les deux rôles ont donc des équilibres
# différents, ce que la matrice symétrique ci-dessus ne permet pas de détecter.
ASYMMETRIC_MATRIX: Final[PayoffMatrix] = tuple(
    tuple(row[column] for column in (1, 2, 0)) for row in BIASED_MATRIX
)
ASYMMETRIC_ATTACKER_STRATEGY: Final[Strategy] = (0.2, 0.4, 0.4)
ASYMMETRIC_GOALKEEPER_STRATEGY: Final[Strategy] = (0.4, 0.4, 0.2)

# Huit manches de référence (attaquant, gardien) -> résultat :
#   1 G-G arrêt   2 G-C but   3 G-D but   4 C-C arrêt
#   5 C-G but     6 D-G but   7 D-D arrêt 8 G-C but
# Attaquant : G x4, C x2, D x2 -> (0.5, 0.25, 0.25)
# Gardien   : G x3, C x3, D x2 -> (0.375, 0.375, 0.25)
# 5 buts, 3 arrêts, gain moyen de l'attaquant (5 - 3) / 8 = 0.25.
SCENARIO_PAIRS: Final[tuple[tuple[Action, Action], ...]] = (
    (LEFT, LEFT),
    (LEFT, CENTER),
    (LEFT, RIGHT),
    (CENTER, CENTER),
    (CENTER, LEFT),
    (RIGHT, LEFT),
    (RIGHT, RIGHT),
    (LEFT, CENTER),
)

GOAL_PAIR: Final[tuple[Action, Action]] = (LEFT, CENTER)
SAVE_PAIR: Final[tuple[Action, Action]] = (LEFT, LEFT)

STRATEGY_SEEDS: Final[list[int]] = list(range(10))


def _rounds(pairs: tuple[tuple[Action, Action], ...]) -> list[RoundResult]:
    """Résout une suite de couples d'actions avec la matrice de référence.

    Args:
        pairs: Couples ``(action attaquant, action gardien)``.

    Returns:
        Les manches, numérotées à partir de 1.
    """
    return [
        resolve_round(number, attacker, goalkeeper, REFERENCE)
        for number, (attacker, goalkeeper) in enumerate(pairs, start=1)
    ]


def _repeat(pair: tuple[Action, Action], times: int) -> list[RoundResult]:
    """Retourne ``times`` manches identiques."""
    return _rounds((pair,) * times)


def _rounds_with_rewards(rewards: tuple[float, ...]) -> list[RoundResult]:
    """Fabrique des manches dont seul le gain de l'attaquant importe."""
    return [
        RoundResult(number, LEFT, LEFT, reward)
        for number, reward in enumerate(rewards, start=1)
    ]


def _pure(action: Action) -> Strategy:
    """Retourne la stratégie pure jouant toujours ``action``."""
    return tuple(1.0 if candidate is action else 0.0 for candidate in Action)


def _random_strategy(seed: int) -> Strategy:
    """Tire une stratégie valide reproductible (loi de Dirichlet uniforme)."""
    weights = np.random.default_rng(seed).dirichlet(np.ones(N_ACTIONS))
    return tuple(float(weight) for weight in weights)


def _scaled_and_shifted(factor: float, shift: float) -> PayoffMatrix:
    """Retourne ``factor * REFERENCE + shift`` (même équilibre, autre échelle)."""
    return tuple(tuple(factor * value + shift for value in row) for row in REFERENCE)


class TestEmpiricalStrategy:
    """``empirical_strategy`` : fréquences des actions jouées."""

    def test_perfectly_uniform_sequence(self) -> None:
        actions = [LEFT, CENTER, RIGHT] * 4
        assert empirical_strategy(actions) == pytest.approx(UNIFORM)

    @pytest.mark.parametrize("action", list(Action), ids=lambda a: a.name)
    def test_pure_sequence(self, action: Action) -> None:
        assert empirical_strategy([action] * 5) == _pure(action)

    def test_unbalanced_sequence(self) -> None:
        actions = [LEFT] * 6 + [CENTER] * 3 + [RIGHT]
        assert empirical_strategy(actions) == pytest.approx((0.6, 0.3, 0.1))

    def test_single_action(self) -> None:
        assert empirical_strategy([RIGHT]) == (0.0, 0.0, 1.0)

    def test_result_is_indexed_by_action(self) -> None:
        strategy = empirical_strategy([CENTER, CENTER, LEFT, RIGHT])
        assert strategy[Action.LEFT] == 0.25
        assert strategy[Action.CENTER] == 0.5
        assert strategy[Action.RIGHT] == 0.25

    def test_order_of_actions_does_not_matter(self) -> None:
        actions = [LEFT, LEFT, CENTER, RIGHT, CENTER]
        assert empirical_strategy(actions) == empirical_strategy(actions[::-1])

    def test_accepts_tuples_and_lists(self) -> None:
        assert empirical_strategy((LEFT, CENTER)) == empirical_strategy([LEFT, CENTER])

    @pytest.mark.parametrize("seed", STRATEGY_SEEDS)
    def test_result_is_always_a_valid_distribution(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        size = int(rng.integers(1, 200))
        actions = [Action(int(index)) for index in rng.integers(0, N_ACTIONS, size)]
        strategy = empirical_strategy(actions)
        assert len(strategy) == N_ACTIONS
        validate_strategy(strategy)

    def test_rejects_empty_sequence(self) -> None:
        with pytest.raises(ValueError, match="Au moins une action"):
            empirical_strategy([])


class TestGoalRateAndSaveRate:
    """``goal_rate`` et ``save_rate`` : taux de buts et d'arrêts."""

    def test_all_goals(self) -> None:
        rounds = _repeat(GOAL_PAIR, 5)
        assert goal_rate(rounds) == 1.0
        assert save_rate(rounds) == 0.0

    def test_all_saves(self) -> None:
        rounds = _repeat(SAVE_PAIR, 5)
        assert goal_rate(rounds) == 0.0
        assert save_rate(rounds) == 1.0

    def test_mixed_rounds(self) -> None:
        rounds = _repeat(GOAL_PAIR, 3) + _repeat(SAVE_PAIR, 2)
        assert goal_rate(rounds) == pytest.approx(0.6)
        assert save_rate(rounds) == pytest.approx(0.4)

    def test_reference_scenario(self) -> None:
        rounds = _rounds(SCENARIO_PAIRS)
        assert goal_rate(rounds) == pytest.approx(5 / 8)
        assert save_rate(rounds) == pytest.approx(3 / 8)

    @pytest.mark.parametrize(
        "pair", [GOAL_PAIR, SAVE_PAIR], ids=["single-goal", "single-save"]
    )
    def test_single_round(self, pair: tuple[Action, Action]) -> None:
        rounds = _rounds((pair,))
        assert goal_rate(rounds) == (1.0 if pair == GOAL_PAIR else 0.0)

    def test_rates_are_complementary(self) -> None:
        rounds = _rounds(SCENARIO_PAIRS)
        assert goal_rate(rounds) + save_rate(rounds) == pytest.approx(1.0)

    def test_order_of_rounds_does_not_matter(self) -> None:
        rounds = _rounds(SCENARIO_PAIRS)
        assert goal_rate(rounds[::-1]) == goal_rate(rounds)
        assert save_rate(rounds[::-1]) == save_rate(rounds)

    def test_accepts_a_tuple_of_rounds(self) -> None:
        rounds = tuple(_rounds(SCENARIO_PAIRS))
        assert goal_rate(rounds) == pytest.approx(5 / 8)

    @pytest.mark.parametrize("rate", [goal_rate, save_rate], ids=["goal", "save"])
    def test_rejects_empty_sequence(
        self, rate: Callable[[Sequence[RoundResult]], float]
    ) -> None:
        with pytest.raises(ValueError, match="Au moins une manche"):
            rate([])


class TestAverageReward:
    """``average_reward`` : gain moyen par manche selon le rôle."""

    def test_attacker_point_of_view(self) -> None:
        assert average_reward(_rounds(SCENARIO_PAIRS), Role.ATTACKER) == 0.25

    def test_goalkeeper_point_of_view(self) -> None:
        assert average_reward(_rounds(SCENARIO_PAIRS), Role.GOALKEEPER) == -0.25

    def test_all_goals(self) -> None:
        rounds = _repeat(GOAL_PAIR, 4)
        assert average_reward(rounds, Role.ATTACKER) == 1.0
        assert average_reward(rounds, Role.GOALKEEPER) == -1.0

    def test_all_saves(self) -> None:
        rounds = _repeat(SAVE_PAIR, 4)
        assert average_reward(rounds, Role.ATTACKER) == -1.0
        assert average_reward(rounds, Role.GOALKEEPER) == 1.0

    def test_rewards_of_both_roles_sum_to_zero(self) -> None:
        rounds = _rounds(SCENARIO_PAIRS)
        total = average_reward(rounds, Role.ATTACKER) + average_reward(
            rounds, Role.GOALKEEPER
        )
        assert total == 0.0

    def test_equals_goal_rate_minus_save_rate_under_unit_gains(self) -> None:
        rounds = _rounds(SCENARIO_PAIRS)
        assert average_reward(rounds, Role.ATTACKER) == pytest.approx(
            goal_rate(rounds) - save_rate(rounds)
        )

    def test_arbitrary_gains_are_averaged(self) -> None:
        rounds = _rounds_with_rewards((2.5, -0.5, 1.0))
        assert average_reward(rounds, Role.ATTACKER) == pytest.approx(1.0)
        assert average_reward(rounds, Role.GOALKEEPER) == pytest.approx(-1.0)

    def test_single_round(self) -> None:
        assert average_reward(_repeat(GOAL_PAIR, 1), Role.ATTACKER) == 1.0

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_rejects_empty_sequence(self, role: Role) -> None:
        with pytest.raises(ValueError, match="Au moins une manche"):
            average_reward([], role)


class TestDistanceToEquilibrium:
    """``distance_to_equilibrium`` : variation totale ``0.5 * somme |p - q|``."""

    @pytest.mark.parametrize(
        "strategy",
        [UNIFORM, _pure(LEFT), (0.5, 0.25, 0.25), BIASED_STRATEGY],
        ids=["uniform", "pure", "mixed", "biased"],
    )
    def test_identical_strategies_have_zero_distance(self, strategy: Strategy) -> None:
        assert distance_to_equilibrium(strategy, strategy) == 0.0

    @pytest.mark.parametrize(
        ("first", "second"),
        list(permutations(list(Action), 2)),
        ids=lambda action: action.name,
    )
    def test_distinct_pure_strategies_have_maximal_distance(
        self, first: Action, second: Action
    ) -> None:
        assert distance_to_equilibrium(_pure(first), _pure(second)) == 1.0

    @pytest.mark.parametrize(
        ("strategy", "reference", "expected"),
        [
            pytest.param((0.5, 0.5, 0.0), UNIFORM, 1 / 3, id="half-half"),
            pytest.param(_pure(LEFT), UNIFORM, 2 / 3, id="pure-vs-uniform"),
            pytest.param((0.5, 0.25, 0.25), UNIFORM, 1 / 6, id="scenario-attacker"),
            pytest.param((0.6, 0.3, 0.1), (0.2, 0.3, 0.5), 0.4, id="mixed-pair"),
        ],
    )
    def test_known_intermediate_distances(
        self, strategy: Strategy, reference: Strategy, expected: float
    ) -> None:
        assert distance_to_equilibrium(strategy, reference) == pytest.approx(expected)

    @pytest.mark.parametrize("seed", STRATEGY_SEEDS)
    def test_is_symmetric_and_bounded(self, seed: int) -> None:
        first, second = _random_strategy(seed), _random_strategy(seed + 100)
        forward = distance_to_equilibrium(first, second)
        backward = distance_to_equilibrium(second, first)
        assert forward == pytest.approx(backward)
        assert 0.0 <= forward <= 1.0

    @pytest.mark.parametrize("seed", STRATEGY_SEEDS)
    def test_satisfies_the_triangle_inequality(self, seed: int) -> None:
        a, b, c = (_random_strategy(seed + offset) for offset in (0, 100, 200))
        direct = distance_to_equilibrium(a, c)
        detour = distance_to_equilibrium(a, b) + distance_to_equilibrium(b, c)
        assert direct <= detour + 1e-12

    @pytest.mark.parametrize(
        ("first", "second"),
        [((0.5, 0.5), UNIFORM), (UNIFORM, (0.5, 0.5)), ((), UNIFORM)],
        ids=["short-first", "short-second", "empty"],
    )
    def test_rejects_strategies_of_different_sizes(
        self, first: Strategy, second: Strategy
    ) -> None:
        with pytest.raises(ValueError):
            distance_to_equilibrium(first, second)


class TestExploitability:
    """``exploitability`` : gain d'une meilleure réponse au-delà de ``v``."""

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_equilibrium_strategy_is_not_exploitable(self, role: Role) -> None:
        value = exploitability(REFERENCE, role, UNIFORM)
        assert value == pytest.approx(0.0, abs=EQUILIBRIUM_TOLERANCE)

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_computed_equilibrium_is_not_exploitable(self, role: Role) -> None:
        for matrix in (REFERENCE, BIASED_MATRIX):
            strategy = solve_equilibrium(matrix).strategy_of(role)
            value = exploitability(matrix, role, strategy)
            assert value == pytest.approx(0.0, abs=EQUILIBRIUM_TOLERANCE)

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_known_mixed_equilibrium_is_not_exploitable(self, role: Role) -> None:
        value = exploitability(BIASED_MATRIX, role, BIASED_STRATEGY)
        assert value == pytest.approx(0.0, abs=EQUILIBRIUM_TOLERANCE)

    # Attaquant pur : le gardien l'arrête à coup sûr, 1/3 - (-1) = 4/3.
    # Gardien pur : l'attaquant marque à coup sûr, 1 - 1/3 = 2/3.
    @pytest.mark.parametrize(
        ("role", "action", "expected"),
        [
            pytest.param(role, action, expected, id=f"{role.value}-{action.name}")
            for role, expected in ((Role.ATTACKER, 4 / 3), (Role.GOALKEEPER, 2 / 3))
            for action in Action
        ],
    )
    def test_pure_strategy_is_strictly_exploitable(
        self, role: Role, action: Action, expected: float
    ) -> None:
        value = exploitability(REFERENCE, role, _pure(action))
        assert value > 0.0
        assert value == pytest.approx(expected, abs=EQUILIBRIUM_TOLERANCE)

    @pytest.mark.parametrize(
        ("role", "expected"),
        [(Role.ATTACKER, 1 / 3 + 0.96), (Role.GOALKEEPER, 0.98 - 1 / 3)],
        ids=lambda item: item.value if isinstance(item, Role) else "",
    )
    def test_very_unbalanced_strategy(self, role: Role, expected: float) -> None:
        # p = (0.98, 0.01, 0.01) : p^T A = (-0.96, 0.98, 0.98) pour l'attaquant,
        # A q = (-0.96, 0.98, 0.98) pour le gardien.
        value = exploitability(REFERENCE, role, (0.98, 0.01, 0.01))
        assert value == pytest.approx(expected, abs=EQUILIBRIUM_TOLERANCE)

    @pytest.mark.parametrize(
        ("role", "expected"),
        [(Role.ATTACKER, 1 / 3), (Role.GOALKEEPER, 2 / 3)],
        ids=lambda item: item.value if isinstance(item, Role) else "",
    )
    def test_known_intermediate_strategy(self, role: Role, expected: float) -> None:
        # Pour (0.5, 0.5, 0), p^T A = A q = (0, 0, 1).
        value = exploitability(REFERENCE, role, (0.5, 0.5, 0.0))
        assert value == pytest.approx(expected, abs=EQUILIBRIUM_TOLERANCE)

    @pytest.mark.parametrize(
        ("role", "expected_pure"),
        [(Role.ATTACKER, (1.6, 0.6, 0.6)), (Role.GOALKEEPER, (0.4, 0.4, 0.4))],
        ids=lambda item: item.value if isinstance(item, Role) else "",
    )
    def test_non_uniform_game_pure_strategies(
        self, role: Role, expected_pure: tuple[float, ...]
    ) -> None:
        values = [
            exploitability(BIASED_MATRIX, role, _pure(action)) for action in Action
        ]
        assert values == pytest.approx(expected_pure, abs=EQUILIBRIUM_TOLERANCE)

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    @pytest.mark.parametrize("seed", STRATEGY_SEEDS)
    def test_is_never_negative(self, role: Role, seed: int) -> None:
        assert exploitability(REFERENCE, role, _random_strategy(seed)) >= 0.0

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    @pytest.mark.parametrize("action", list(Action), ids=lambda a: a.name)
    def test_decreases_when_moving_toward_the_equilibrium(
        self, role: Role, action: Action
    ) -> None:
        """Fonction convexe, nulle à l'équilibre : décroissante sur le segment."""
        pure = _pure(action)
        values = [
            exploitability(
                REFERENCE,
                role,
                tuple((1 - t) * p + t * u for p, u in zip(pure, UNIFORM)),
            )
            for t in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))
        assert values[-1] == pytest.approx(0.0, abs=EQUILIBRIUM_TOLERANCE)

    @pytest.mark.parametrize("seed", STRATEGY_SEEDS[:5])
    def test_attacker_value_matches_the_goalkeeper_best_response(
        self, seed: int
    ) -> None:
        """Cohérence avec ``best_response`` : ``v - (p^T A)[réponse du gardien]``."""
        strategy = _random_strategy(seed)
        reply = best_response(BIASED_MATRIX, Role.GOALKEEPER, strategy)
        expected = BIASED_VALUE - float((np.asarray(strategy) @ BIASED_MATRIX)[reply])
        value = exploitability(BIASED_MATRIX, Role.ATTACKER, strategy)
        assert value == pytest.approx(max(0.0, expected), abs=EQUILIBRIUM_TOLERANCE)

    @pytest.mark.parametrize("seed", STRATEGY_SEEDS[:5])
    def test_goalkeeper_value_matches_the_attacker_best_response(
        self, seed: int
    ) -> None:
        """Cohérence avec ``best_response`` : ``(A q)[réponse attaquant] - v``."""
        strategy = _random_strategy(seed)
        reply = best_response(BIASED_MATRIX, Role.ATTACKER, strategy)
        expected = float((np.asarray(BIASED_MATRIX) @ strategy)[reply]) - BIASED_VALUE
        value = exploitability(BIASED_MATRIX, Role.GOALKEEPER, strategy)
        assert value == pytest.approx(max(0.0, expected), abs=EQUILIBRIUM_TOLERANCE)

    def test_is_invariant_under_a_constant_shift_of_the_gains(self) -> None:
        shifted = _scaled_and_shifted(1.0, 5.0)
        for role in Role:
            assert exploitability(shifted, role, (0.5, 0.5, 0.0)) == pytest.approx(
                exploitability(REFERENCE, role, (0.5, 0.5, 0.0)),
                abs=EQUILIBRIUM_TOLERANCE,
            )

    def test_scales_with_the_gains(self) -> None:
        scaled = _scaled_and_shifted(3.0, 0.0)
        assert exploitability(scaled, Role.ATTACKER, _pure(LEFT)) == pytest.approx(
            4.0, abs=EQUILIBRIUM_TOLERANCE
        )
        assert exploitability(scaled, Role.GOALKEEPER, _pure(LEFT)) == pytest.approx(
            2.0, abs=EQUILIBRIUM_TOLERANCE
        )

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    @pytest.mark.parametrize(
        "strategy",
        [(0.5, 0.5), (-0.1, 0.6, 0.5), (0.5, 0.5, 0.5), (NAN, 0.5, 0.5)],
        ids=["too-short", "negative", "sum-above-one", "nan"],
    )
    def test_rejects_invalid_strategy(self, role: Role, strategy: Strategy) -> None:
        with pytest.raises(ValueError):
            exploitability(REFERENCE, role, strategy)

    @pytest.mark.parametrize(
        "matrix",
        [
            ((1.0, 1.0), (1.0, 1.0)),
            ((NAN, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0)),
            ((INF, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0)),
        ],
        ids=["2x2", "nan", "inf"],
    )
    def test_rejects_invalid_matrix(self, matrix: PayoffMatrix) -> None:
        with pytest.raises(ValueError):
            exploitability(matrix, Role.ATTACKER, UNIFORM)


class TestComputeAgentStats:
    """``compute_agent_stats`` : agrégation et cohérence avec les métriques."""

    ROUNDS: Final[list[RoundResult]] = _rounds(SCENARIO_PAIRS)

    @staticmethod
    def _stats(role: Role, decision_time: float = 0.002) -> AgentStats:
        """Calcule les statistiques du scénario de référence pour un rôle."""
        return compute_agent_stats(
            AgentType.FREQUENCY,
            role,
            TestComputeAgentStats.ROUNDS,
            decision_time,
            REFERENCE,
        )

    def test_attacker_scenario(self) -> None:
        stats = self._stats(Role.ATTACKER)
        assert stats.agent_type is AgentType.FREQUENCY
        assert stats.role is Role.ATTACKER
        assert stats.rounds == 8
        assert stats.goals == 5
        assert stats.saves == 3
        assert stats.mean_reward == pytest.approx(0.25)
        assert stats.mean_decision_time == 0.002
        assert stats.empirical_strategy == pytest.approx((0.5, 0.25, 0.25))
        assert stats.equilibrium_gap == pytest.approx(1 / 6)
        assert stats.exploitability == pytest.approx(1 / 3, abs=EQUILIBRIUM_TOLERANCE)

    def test_goalkeeper_scenario(self) -> None:
        """Pour un gardien, ``goals`` compte les buts concédés."""
        stats = self._stats(Role.GOALKEEPER)
        assert stats.role is Role.GOALKEEPER
        assert stats.rounds == 8
        assert stats.goals == 5
        assert stats.saves == 3
        assert stats.mean_reward == pytest.approx(-0.25)
        assert stats.empirical_strategy == pytest.approx((0.375, 0.375, 0.25))
        assert stats.equilibrium_gap == pytest.approx(1 / 12)
        assert stats.exploitability == pytest.approx(1 / 6, abs=EQUILIBRIUM_TOLERANCE)

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_counters_are_consistent(self, role: Role) -> None:
        stats = self._stats(role)
        assert stats.goals + stats.saves == stats.rounds == len(self.ROUNDS)

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_rates_are_consistent_with_the_rate_functions(self, role: Role) -> None:
        stats = self._stats(role)
        assert stats.goal_rate == pytest.approx(goal_rate(self.ROUNDS))
        assert stats.save_rate == pytest.approx(save_rate(self.ROUNDS))
        assert stats.goal_rate + stats.save_rate == pytest.approx(1.0)

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_mean_reward_is_consistent_with_average_reward(self, role: Role) -> None:
        assert self._stats(role).mean_reward == average_reward(self.ROUNDS, role)

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_empirical_strategy_is_built_from_the_role_actions(
        self, role: Role
    ) -> None:
        stats = self._stats(role)
        played = [result.action_of(role) for result in self.ROUNDS]
        assert stats.empirical_strategy == empirical_strategy(played)
        validate_strategy(stats.empirical_strategy)

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_equilibrium_gap_is_consistent(self, role: Role) -> None:
        stats = self._stats(role)
        reference = solve_equilibrium(REFERENCE).strategy_of(role)
        assert stats.equilibrium_gap == pytest.approx(
            distance_to_equilibrium(stats.empirical_strategy, reference)
        )

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_exploitability_is_consistent(self, role: Role) -> None:
        stats = self._stats(role)
        assert stats.exploitability == pytest.approx(
            exploitability(REFERENCE, role, stats.empirical_strategy)
        )

    @pytest.mark.parametrize("agent_type", list(AgentType), ids=lambda t: t.value)
    @pytest.mark.parametrize("decision_time", [0.0, 1.5])
    def test_agent_type_and_decision_time_are_passed_through(
        self, agent_type: AgentType, decision_time: float
    ) -> None:
        stats = compute_agent_stats(
            agent_type, Role.ATTACKER, self.ROUNDS, decision_time, REFERENCE
        )
        assert stats.agent_type is agent_type
        assert stats.mean_decision_time == decision_time

    def test_deterministic_attacker_is_fully_exploitable(self) -> None:
        """Toujours LEFT (cahier §4.1) : stratégie pure, exploitabilité 4/3."""
        goalkeeper_cycle = (CENTER, RIGHT, LEFT, CENTER, RIGHT)
        rounds = _rounds(tuple((LEFT, reply) for reply in goalkeeper_cycle))
        stats = compute_agent_stats(
            AgentType.DETERMINISTIC, Role.ATTACKER, rounds, 0.0, REFERENCE
        )
        assert stats.empirical_strategy == _pure(LEFT)
        assert stats.equilibrium_gap == pytest.approx(2 / 3)
        assert stats.exploitability == pytest.approx(4 / 3, abs=EQUILIBRIUM_TOLERANCE)
        assert (stats.goals, stats.saves) == (4, 1)
        assert stats.mean_reward == pytest.approx(0.6)

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_uniform_play_has_no_gap_and_no_exploitability(self, role: Role) -> None:
        cycle = ((LEFT, CENTER), (CENTER, RIGHT), (RIGHT, LEFT)) * 3
        stats = compute_agent_stats(
            AgentType.MINIMAX, role, _rounds(cycle), 0.0, REFERENCE
        )
        assert stats.equilibrium_gap == pytest.approx(0.0, abs=EQUILIBRIUM_TOLERANCE)
        assert stats.exploitability == pytest.approx(0.0, abs=EQUILIBRIUM_TOLERANCE)

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_uses_the_equilibrium_of_the_given_matrix_and_role(
        self, role: Role
    ) -> None:
        """Jouer l'équilibre de son rôle donne écart et exploitabilité nuls.

        Sur une matrice asymétrique, l'équilibre de l'attaquant diffère de
        celui du gardien : une référence prise pour le mauvais rôle serait
        détectée.
        """
        attacker_actions = [LEFT] * 2 + [CENTER] * 4 + [RIGHT] * 4
        goalkeeper_actions = [LEFT] * 4 + [CENTER] * 4 + [RIGHT] * 2
        rounds = [
            resolve_round(number, attacker, goalkeeper, ASYMMETRIC_MATRIX)
            for number, (attacker, goalkeeper) in enumerate(
                zip(attacker_actions, goalkeeper_actions), start=1
            )
        ]
        stats = compute_agent_stats(
            AgentType.MINIMAX, role, rounds, 0.0, ASYMMETRIC_MATRIX
        )
        expected = (
            ASYMMETRIC_ATTACKER_STRATEGY
            if role is Role.ATTACKER
            else ASYMMETRIC_GOALKEEPER_STRATEGY
        )
        assert stats.empirical_strategy == pytest.approx(expected)
        assert stats.equilibrium_gap == pytest.approx(0.0, abs=EQUILIBRIUM_TOLERANCE)
        assert stats.exploitability == pytest.approx(0.0, abs=EQUILIBRIUM_TOLERANCE)
        assert stats.goals + stats.saves == stats.rounds

    def test_asymmetric_matrix_has_the_documented_equilibria(self) -> None:
        """Garde-fou sur la fixture : équilibres et valeur calculés à la main."""
        equilibrium = solve_equilibrium(ASYMMETRIC_MATRIX)
        assert equilibrium.attacker_strategy == pytest.approx(
            ASYMMETRIC_ATTACKER_STRATEGY, abs=EQUILIBRIUM_TOLERANCE
        )
        assert equilibrium.goalkeeper_strategy == pytest.approx(
            ASYMMETRIC_GOALKEEPER_STRATEGY, abs=EQUILIBRIUM_TOLERANCE
        )
        assert equilibrium.value == pytest.approx(
            BIASED_VALUE, abs=EQUILIBRIUM_TOLERANCE
        )

    def test_accepts_a_tuple_of_rounds(self) -> None:
        stats = compute_agent_stats(
            AgentType.FICTITIOUS_PLAY,
            Role.ATTACKER,
            tuple(self.ROUNDS),
            0.0,
            REFERENCE,
        )
        assert stats.rounds == len(self.ROUNDS)

    def test_handles_thousands_of_rounds(self) -> None:
        """ENF05 : plusieurs milliers de manches, sans dérive numérique."""
        cycle = ((LEFT, CENTER), (CENTER, RIGHT), (RIGHT, LEFT))
        stats = compute_agent_stats(
            AgentType.MINIMAX, Role.ATTACKER, _rounds(cycle * 1000), 0.0, REFERENCE
        )
        assert stats.rounds == 3000
        assert (stats.goals, stats.saves) == (3000, 0)
        assert stats.equilibrium_gap == pytest.approx(0.0, abs=PROBABILITY_TOLERANCE)
        assert stats.exploitability == pytest.approx(0.0, abs=EQUILIBRIUM_TOLERANCE)

    def test_rejects_empty_rounds(self) -> None:
        with pytest.raises(ValueError):
            compute_agent_stats(AgentType.MINIMAX, Role.ATTACKER, [], 0.0, REFERENCE)

    def test_rejects_invalid_matrix(self) -> None:
        with pytest.raises(ValueError):
            compute_agent_stats(
                AgentType.MINIMAX,
                Role.ATTACKER,
                self.ROUNDS,
                0.0,
                ((1.0, 1.0), (1.0, 1.0)),
            )


class TestNumericalRobustness:
    """Les petites erreurs numériques ne doivent provoquer aucun échec."""

    # -- PROBABILITY_TOLERANCE ------------------------------------------------

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_strategy_with_sum_error_within_tolerance_is_accepted(
        self, role: Role
    ) -> None:
        strategy = (1 / 3, 1 / 3, 1 / 3 + PROBABILITY_TOLERANCE / 2)
        assert exploitability(REFERENCE, role, strategy) >= 0.0

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_strategy_with_sum_error_beyond_tolerance_is_rejected(
        self, role: Role
    ) -> None:
        strategy = (1 / 3, 1 / 3, 1 / 3 + 10 * PROBABILITY_TOLERANCE)
        with pytest.raises(ValueError, match="sommer à 1"):
            exploitability(REFERENCE, role, strategy)

    def test_negative_noise_within_tolerance_is_accepted(self) -> None:
        noise = PROBABILITY_TOLERANCE / 2
        strategy = (-noise, 0.5, 0.5 + noise)
        assert exploitability(REFERENCE, Role.ATTACKER, strategy) >= 0.0

    def test_long_empirical_strategy_stays_a_valid_distribution(self) -> None:
        actions = [Action(index % N_ACTIONS) for index in range(10_007)]
        strategy = empirical_strategy(actions)
        validate_strategy(strategy)
        assert math.fsum(strategy) == pytest.approx(1.0, abs=PROBABILITY_TOLERANCE)

    # -- NUMERICAL_NOISE_TOLERANCE -------------------------------------------

    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_exploitability_is_clipped_at_zero_at_the_equilibrium(
        self, role: Role
    ) -> None:
        """Un écart négatif dû aux arrondis est ramené à 0, jamais en dessous."""
        assert exploitability(REFERENCE, role, UNIFORM) >= 0.0

    def test_noise_around_the_equilibrium_gives_negligible_exploitability(
        self,
    ) -> None:
        noise = NUMERICAL_NOISE_TOLERANCE / 10
        strategy = (1 / 3 + noise, 1 / 3 - noise, 1 / 3)
        value = exploitability(REFERENCE, Role.ATTACKER, strategy)
        assert 0.0 <= value <= NUMERICAL_NOISE_TOLERANCE

    def test_noise_around_the_equilibrium_gives_negligible_distance(self) -> None:
        noise = NUMERICAL_NOISE_TOLERANCE / 10
        strategy = (1 / 3 + noise, 1 / 3 - noise, 1 / 3)
        assert distance_to_equilibrium(strategy, UNIFORM) <= NUMERICAL_NOISE_TOLERANCE

    # -- EQUILIBRIUM_TOLERANCE -----------------------------------------------

    @pytest.mark.parametrize(
        ("factor", "shift"),
        list(product((0.1, 1.0, 10.0, 100.0), (-50.0, 0.0, 50.0))),
    )
    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_equilibrium_stays_unexploitable_whatever_the_scale(
        self, role: Role, factor: float, shift: float
    ) -> None:
        matrix = _scaled_and_shifted(factor, shift)
        value = exploitability(matrix, role, UNIFORM)
        assert value == pytest.approx(0.0, abs=EQUILIBRIUM_TOLERANCE)


class TestKnownLimitations:
    """Limites documentées : à réviser si le modèle de gains évolue."""

    def test_non_unit_gains_are_not_counted_as_goals(self) -> None:
        """``docs/game_theory.md`` §7 : un but est déduit d'un gain égal à +1.

        Avec des gains personnalisés différents de ±1 (par exemple 0.5), le
        compteur ``goals`` reste nul alors que le gain moyen est positif.
        Ce test fige la limite connue : le jour où l'issue d'une manche sera
        stockée dans ``RoundResult``, il devra être mis à jour.
        """
        rounds = _rounds_with_rewards((0.5, 0.5, 0.5))
        stats = compute_agent_stats(
            AgentType.MINIMAX, Role.ATTACKER, rounds, 0.0, REFERENCE
        )
        assert stats.mean_reward == pytest.approx(0.5)
        assert (stats.goals, stats.saves) == (0, 3)