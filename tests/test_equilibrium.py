"""Tests de ``theory.equilibrium`` : minimax, valeur du jeu, meilleure réponse.
 
Organisation du fichier :
 
* ``solve_equilibrium`` sur la matrice de référence (cahier des charges, §3.3) ;
* symétrie du jeu ;
* valeur du jeu sur des jeux connus (point selle, pierre-papier-ciseaux) ;
* cohérence attaquant / gardien (théorème minimax) sur des jeux connus et des
  matrices aléatoires à graine fixe ;
* nettoyage des vecteurs bruts issus de ``linprog`` ;
* ``best_response`` et règle de départage ;
* robustesse numérique (``PROBABILITY_TOLERANCE``,
  ``NUMERICAL_NOISE_TOLERANCE``, ``EQUILIBRIUM_TOLERANCE``).
 
Les attentes sont vérifiées par des propriétés mathématiques (garantie
minimax, invariance par décalage ou changement d'échelle) autant que par des
valeurs écrites en dur, afin de ne pas dépendre de la solution particulière
choisie par le solveur quand l'équilibre n'est pas unique.
 
Deux tests marqués ``xfail(strict=True)`` documentent des faiblesses connues
(W1 et W2 du rapport d'analyse). Lorsqu'elles seront corrigées, le test
passera et pytest signalera l'échec « strict » : il suffira alors de retirer
le marqueur.
"""

import math
from collections.abc import Callable
from itertools import permutations
from types import SimpleNamespace
from typing import Final
import numpy as np
import pytest 
from numpy.typing import NDArray
from game.actions import Action
from game.constants import (EQUILIBRIUM_TOLERANCE, N_ACTIONS, NUMERICAL_NOISE_TOLERANCE, PROBABILITY_TOLERANCE, REFERENCE_EQUILIBRIUM_STRATEGY, REFERENCE_GAME_VALUE)
from game.models import Equilibrium, PayoffMatrix, Role, Strategy
from game.payoff import build_reference_matrix
from theory import equilibrium as equilibrium_module
from theory.equilibrium import _to_strategy, best_response, solve_equilibrium
from theory.validation import validate_strategy

NAN: Final[float] = float("nan")
INF: Final[float] = float("inf")

REFERENCE: Final[PayoffMatrix] = build_reference_matrix()

# Stratégie uniforme écrite indépendamment des constantes prédéfinies.
UNIFORM: Final[Strategy] = (1 / 3,) * N_ACTIONS

# Pierre-papier-ciseaux : équilibre uniforme unique, valeur du jeu nulle.
ROCK_PAPER_SCISSORS: Final[PayoffMatrix] = (
    (0.0, -1.0, 1.0),
    (1.0, 0.0, -1.0),
    (-1.0, 1.0, 0.0),
)

# Point selle en (LEFT, CENTER), de valeur 2. La colonne CENTER domine strictement les deux autres pour le gardien ; contre elle, la ligne LEFT est l'unique meilleure réponse de l'attaquant : l'équilibre est pur et unique.
SADDLE_POINT: Final[PayoffMatrix] = (
    (3.0, 2.0, 4.0),
    (1.0, 0.0, 5.0),
    (2.0, 1.0, 3.0),
)

ZERO_MATRIX: Final[PayoffMatrix] = ((0.0,) * N_ACTIONS,) * N_ACTIONS

PERMUTATIONS: Final[list[tuple[int, ...]]] = list(permutations(range(N_ACTIONS)))

def _random_matrix(seed: int) -> PayoffMatrix:
    """Construit une matrice 3x3 reproductible à gains dans [-5, 5].
 
    Args:
        seed: Graine du générateur NumPy.
 
    Returns:
        Une matrice des gains de l'attaquant.
    """
    values = np.random.default_rng(seed).uniform(-5.0, 5.0, (N_ACTIONS, N_ACTIONS))
    return tuple(tuple(float(value) for value in row) for row in values)
 
 
GAMES: Final[dict[str, PayoffMatrix]] = {
    "reference": REFERENCE,
    "rock_paper_scissors": ROCK_PAPER_SCISSORS,
    "saddle_point": SADDLE_POINT,
    "zero": ZERO_MATRIX,
    **{f"random_{seed}": _random_matrix(seed) for seed in range(10)},
}
GAME_MATRICES: Final[list[PayoffMatrix]] = list(GAMES.values())
GAME_IDS: Final[list[str]] = list(GAMES)
 
 
def _pure(action: Action) -> Strategy:
    """Retourne la stratégie pure jouant toujours ``action``."""
    return tuple(1.0 if candidate is action else 0.0 for candidate in Action)
 
 
def _transform(
    matrix: PayoffMatrix, function: Callable[[float], float]
) -> PayoffMatrix:
    """Applique ``function`` à chaque gain d'une matrice."""
    return tuple(tuple(function(value) for value in row) for row in matrix)
 
 
def _attacker_guarantee(matrix: PayoffMatrix, strategy: Strategy) -> float:
    """Gain minimal garanti à l'attaquant : ``min_j (p^T A)_j``."""
    return float(np.min(np.asarray(strategy) @ np.asarray(matrix)))
 
 
def _goalkeeper_ceiling(matrix: PayoffMatrix, strategy: Strategy) -> float:
    """Gain maximal concédé par le gardien : ``max_i (A q)_i``."""
    return float(np.max(np.asarray(matrix) @ np.asarray(strategy)))
 
 
class TestSolveEquilibriumReference:
    """Équilibre de la matrice de référence (EF09, critère d'acceptation §10.2)."""
 
    def test_returns_an_equilibrium(self) -> None:
        assert isinstance(solve_equilibrium(REFERENCE), Equilibrium)
 
    def test_attacker_strategy_is_uniform(self) -> None:
        strategy = solve_equilibrium(REFERENCE).attacker_strategy
        assert strategy == pytest.approx(UNIFORM, abs=EQUILIBRIUM_TOLERANCE)
 
    def test_goalkeeper_strategy_is_uniform(self) -> None:
        strategy = solve_equilibrium(REFERENCE).goalkeeper_strategy
        assert strategy == pytest.approx(UNIFORM, abs=EQUILIBRIUM_TOLERANCE)
 
    def test_matches_reference_constants(self) -> None:
        """Contrat d'API §11 : valeur et stratégie de référence retrouvées."""
        equilibrium = solve_equilibrium(REFERENCE)
        expected = REFERENCE_EQUILIBRIUM_STRATEGY
        assert equilibrium.attacker_strategy == pytest.approx(
            expected, abs=EQUILIBRIUM_TOLERANCE
        )
        assert equilibrium.goalkeeper_strategy == pytest.approx(
            expected, abs=EQUILIBRIUM_TOLERANCE
        )
        assert equilibrium.value == pytest.approx(
            REFERENCE_GAME_VALUE, abs=EQUILIBRIUM_TOLERANCE
        )
 
    def test_game_value_is_one_third(self) -> None:
        value = solve_equilibrium(REFERENCE).value
        assert value == pytest.approx(1 / 3, abs=EQUILIBRIUM_TOLERANCE)
 
    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_probabilities_sum_to_one(self, role: Role) -> None:
        strategy = solve_equilibrium(REFERENCE).strategy_of(role)
        assert math.fsum(strategy) == pytest.approx(1.0, abs=PROBABILITY_TOLERANCE)
 
    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_strategies_pass_validation(self, role: Role) -> None:
        strategy = solve_equilibrium(REFERENCE).strategy_of(role)
        assert len(strategy) == N_ACTIONS
        validate_strategy(strategy)
 
    def test_strategy_of_returns_the_matching_field(self) -> None:
        equilibrium = solve_equilibrium(REFERENCE)
        assert equilibrium.strategy_of(Role.ATTACKER) == equilibrium.attacker_strategy
        assert equilibrium.strategy_of(Role.GOALKEEPER) == (
            equilibrium.goalkeeper_strategy
        )
 
    def test_is_deterministic(self) -> None:
        assert solve_equilibrium(REFERENCE) == solve_equilibrium(REFERENCE)
 
 
class TestSolveEquilibriumSymmetry:
    """Symétrie du jeu de référence et de sa résolution."""
 
    def test_reference_matrix_is_symmetric(self) -> None:
        array = np.asarray(REFERENCE)
        assert np.array_equal(array, array.T)
 
    @pytest.mark.parametrize("order", PERMUTATIONS, ids=str)
    def test_reference_matrix_is_invariant_under_relabeling(
        self, order: tuple[int, ...]
    ) -> None:
        """Renommer les zones (même permutation des deux côtés) ne change rien."""
        array = np.asarray(REFERENCE)
        assert np.array_equal(array[np.ix_(order, order)], array)
 
    def test_both_roles_play_the_same_strategy(self) -> None:
        equilibrium = solve_equilibrium(REFERENCE)
        assert equilibrium.attacker_strategy == pytest.approx(
            equilibrium.goalkeeper_strategy, abs=EQUILIBRIUM_TOLERANCE
        )
 
    def test_transposed_opposite_game_swaps_the_roles(self) -> None:
        """Résoudre ``-A^T`` revient à échanger attaquant et gardien."""
        original = solve_equilibrium(SADDLE_POINT)
        transposed = tuple(
            tuple(-value for value in column) for column in zip(*SADDLE_POINT)
        )
        swapped = solve_equilibrium(transposed)
        assert swapped.attacker_strategy == pytest.approx(
            original.goalkeeper_strategy, abs=EQUILIBRIUM_TOLERANCE
        )
        assert swapped.goalkeeper_strategy == pytest.approx(
            original.attacker_strategy, abs=EQUILIBRIUM_TOLERANCE
        )
        assert swapped.value == pytest.approx(
            -original.value, abs=EQUILIBRIUM_TOLERANCE
        )
 
    @pytest.mark.parametrize("row_order", PERMUTATIONS, ids=str)
    @pytest.mark.parametrize("column_order", PERMUTATIONS, ids=str)
    def test_relabeling_actions_permutes_the_optimal_strategies(
        self, row_order: tuple[int, ...], column_order: tuple[int, ...]
    ) -> None:
        """Les stratégies suivent le renommage des actions de chaque joueur."""
        permuted = np.asarray(SADDLE_POINT)[np.ix_(row_order, column_order)]
        matrix = tuple(tuple(float(value) for value in row) for row in permuted)
        equilibrium = solve_equilibrium(matrix)
        # Dans SADDLE_POINT, l'attaquant joue LEFT et le gardien CENTER.
        expected_attacker = tuple(
            1.0 if source == Action.LEFT else 0.0 for source in row_order
        )
        expected_goalkeeper = tuple(
            1.0 if source == Action.CENTER else 0.0 for source in column_order
        )
        assert equilibrium.attacker_strategy == pytest.approx(
            expected_attacker, abs=EQUILIBRIUM_TOLERANCE
        )
        assert equilibrium.goalkeeper_strategy == pytest.approx(
            expected_goalkeeper, abs=EQUILIBRIUM_TOLERANCE
        )
 
 
class TestKnownGames:
    """Valeur et stratégies sur des jeux dont la solution est connue."""
 
    @pytest.mark.parametrize(
        ("matrix", "value", "attacker", "goalkeeper"),
        [
            pytest.param(REFERENCE, 1 / 3, UNIFORM, UNIFORM, id="reference"),
            pytest.param(
                ROCK_PAPER_SCISSORS, 0.0, UNIFORM, UNIFORM, id="rock_paper_scissors"
            ),
            pytest.param(
                SADDLE_POINT,
                2.0,
                _pure(Action.LEFT),
                _pure(Action.CENTER),
                id="saddle_point",
            ),
        ],
    )
    def test_solution_matches_theory(
        self,
        matrix: PayoffMatrix,
        value: float,
        attacker: Strategy,
        goalkeeper: Strategy,
    ) -> None:
        equilibrium = solve_equilibrium(matrix)
        assert equilibrium.value == pytest.approx(value, abs=EQUILIBRIUM_TOLERANCE)
        assert equilibrium.attacker_strategy == pytest.approx(
            attacker, abs=EQUILIBRIUM_TOLERANCE
        )
        assert equilibrium.goalkeeper_strategy == pytest.approx(
            goalkeeper, abs=EQUILIBRIUM_TOLERANCE
        )
 
    @pytest.mark.parametrize("constant", [-3.0, 0.0, 7.0])
    def test_constant_matrix_has_that_constant_as_value(self, constant: float) -> None:
        matrix = _transform(ZERO_MATRIX, lambda value: value + constant)
        assert solve_equilibrium(matrix).value == pytest.approx(
            constant, abs=EQUILIBRIUM_TOLERANCE
        )
 
    def test_degenerate_game_still_returns_valid_strategies(self) -> None:
        """L'équilibre n'est pas unique : seule la validité est exigée."""
        equilibrium = solve_equilibrium(ZERO_MATRIX)
        validate_strategy(equilibrium.attacker_strategy)
        validate_strategy(equilibrium.goalkeeper_strategy)
 
 
class TestEquilibriumInvariance:
    """Invariance de l'équilibre par décalage et changement d'échelle."""
 
    @pytest.mark.parametrize("shift", [-10.0, 0.5, 1000.0])
    def test_adding_a_constant_shifts_the_value_only(self, shift: float) -> None:
        base = solve_equilibrium(SADDLE_POINT)
        shifted = solve_equilibrium(_transform(SADDLE_POINT, lambda v: v + shift))
        assert shifted.value == pytest.approx(
            base.value + shift, abs=EQUILIBRIUM_TOLERANCE
        )
        assert shifted.attacker_strategy == pytest.approx(
            base.attacker_strategy, abs=EQUILIBRIUM_TOLERANCE
        )
        assert shifted.goalkeeper_strategy == pytest.approx(
            base.goalkeeper_strategy, abs=EQUILIBRIUM_TOLERANCE
        )
 
    @pytest.mark.parametrize("factor", [0.1, 3.0, 1000.0])
    def test_positive_scaling_scales_the_value_only(self, factor: float) -> None:
        base = solve_equilibrium(REFERENCE)
        scaled = solve_equilibrium(_transform(REFERENCE, lambda v: v * factor))
        assert scaled.value == pytest.approx(
            base.value * factor, abs=EQUILIBRIUM_TOLERANCE
        )
        assert scaled.attacker_strategy == pytest.approx(
            base.attacker_strategy, abs=EQUILIBRIUM_TOLERANCE
        )
 
    def test_all_negative_matrix_is_supported(self) -> None:
        """La valeur ``v`` du programme linéaire est libre, donc négative ok."""
        matrix = _transform(REFERENCE, lambda value: value - 10.0)
        assert solve_equilibrium(matrix).value == pytest.approx(
            1 / 3 - 10.0, abs=EQUILIBRIUM_TOLERANCE
        )
 
 
@pytest.mark.parametrize("matrix", GAME_MATRICES, ids=GAME_IDS)
class TestAttackerGoalkeeperConsistency:
    """Cohérence des deux rôles : théorème minimax et point selle."""
 
    def test_strategies_are_valid_and_non_negative(self, matrix: PayoffMatrix) -> None:
        equilibrium = solve_equilibrium(matrix)
        for role in Role:
            strategy = equilibrium.strategy_of(role)
            validate_strategy(strategy)
            assert min(strategy) >= 0.0
            assert math.fsum(strategy) == pytest.approx(
                1.0, abs=PROBABILITY_TOLERANCE
            )
 
    def test_attacker_strategy_guarantees_the_value(self, matrix: PayoffMatrix) -> None:
        equilibrium = solve_equilibrium(matrix)
        guarantee = _attacker_guarantee(matrix, equilibrium.attacker_strategy)
        assert guarantee == pytest.approx(equilibrium.value, abs=EQUILIBRIUM_TOLERANCE)
 
    def test_goalkeeper_strategy_caps_the_value(self, matrix: PayoffMatrix) -> None:
        equilibrium = solve_equilibrium(matrix)
        ceiling = _goalkeeper_ceiling(matrix, equilibrium.goalkeeper_strategy)
        assert ceiling == pytest.approx(equilibrium.value, abs=EQUILIBRIUM_TOLERANCE)
 
    def test_goalkeeper_value_is_opposite_of_attacker_value(
        self, matrix: PayoffMatrix
    ) -> None:
        """Par dualité, le gardien garantit ``-v`` (gain du gardien = ``-A``)."""
        equilibrium = solve_equilibrium(matrix)
        q = np.asarray(equilibrium.goalkeeper_strategy)
        goalkeeper_guarantee = float(np.min(-(np.asarray(matrix) @ q)))
        assert goalkeeper_guarantee == pytest.approx(
            -equilibrium.value, abs=EQUILIBRIUM_TOLERANCE
        )
 
    def test_expected_payoff_at_equilibrium_equals_the_value(
        self, matrix: PayoffMatrix
    ) -> None:
        equilibrium = solve_equilibrium(matrix)
        p = np.asarray(equilibrium.attacker_strategy)
        q = np.asarray(equilibrium.goalkeeper_strategy)
        assert float(p @ np.asarray(matrix) @ q) == pytest.approx(
            equilibrium.value, abs=EQUILIBRIUM_TOLERANCE
        )
 
    def test_best_responses_to_equilibrium_earn_exactly_the_value(
        self, matrix: PayoffMatrix
    ) -> None:
        """Aucun joueur ne gagne à dévier de l'équilibre (propriété de Nash)."""
        equilibrium = solve_equilibrium(matrix)
        array = np.asarray(matrix)
        attacker_deviation = best_response(
            matrix, Role.ATTACKER, equilibrium.goalkeeper_strategy
        )
        goalkeeper_deviation = best_response(
            matrix, Role.GOALKEEPER, equilibrium.attacker_strategy
        )
        q = np.asarray(equilibrium.goalkeeper_strategy)
        p = np.asarray(equilibrium.attacker_strategy)
        assert float((array @ q)[attacker_deviation]) == pytest.approx(
            equilibrium.value, abs=EQUILIBRIUM_TOLERANCE
        )
        assert float((p @ array)[goalkeeper_deviation]) == pytest.approx(
            equilibrium.value, abs=EQUILIBRIUM_TOLERANCE
        )
 
 
INVALID_MATRICES: Final[dict[str, PayoffMatrix]] = {
    "empty": (),
    "2x2": ((1.0, 1.0), (1.0, 1.0)),
    "4x4": ((1.0,) * 4,) * 4,
    "ragged": ((1.0, 1.0, 1.0), (1.0, 1.0), (1.0, 1.0, 1.0)),
}
 
 
class TestSolveEquilibriumValidation:
    """Rejet des matrices invalides (``ValueError``) et échec du solveur."""
 
    @pytest.mark.parametrize(
        "matrix", list(INVALID_MATRICES.values()), ids=list(INVALID_MATRICES)
    )
    def test_rejects_wrong_shape(self, matrix: PayoffMatrix) -> None:
        with pytest.raises(ValueError, match="taille"):
            solve_equilibrium(matrix)
 
    @pytest.mark.parametrize("bad", [NAN, INF, -INF], ids=["nan", "inf", "-inf"])
    def test_rejects_non_finite_gains(self, bad: float) -> None:
        matrix: PayoffMatrix = (
            (-1.0, 1.0, 1.0),
            (1.0, bad, 1.0),
            (1.0, 1.0, -1.0),
        )
        with pytest.raises(ValueError, match="finies"):
            solve_equilibrium(matrix)
 
    def test_raises_runtime_error_when_the_solver_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Seul mock du fichier, volontaire et nécessaire.
 
        Un programme linéaire fini 3x3 est toujours faisable et borné : le
        solveur ne peut pas échouer naturellement. Le chemin d'erreur
        ``RuntimeError`` du contrat d'API (§7.4) n'est donc testable qu'en
        simulant l'échec de ``linprog``.
        """
 
        def failing_linprog(*_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(success=False, message="boom", x=None)
 
        monkeypatch.setattr(equilibrium_module, "linprog", failing_linprog)
        with pytest.raises(
            RuntimeError, match="Échec de linprog pour le rôle attacker : boom"
        ):
            solve_equilibrium(REFERENCE)
 
 
class TestToStrategyCleaning:
    """Nettoyage des vecteurs bruts renvoyés par ``linprog`` (§7.4)."""
 
    @staticmethod
    def _raw(*values: float) -> NDArray[np.float64]:
        """Construit un vecteur brut de type ``float64``."""
        return np.array(values, dtype=np.float64)
 
    def test_noise_slightly_below_zero_is_set_to_zero(self) -> None:
        strategy = _to_strategy(self._raw(-NUMERICAL_NOISE_TOLERANCE / 2, 0.5, 0.5))
        assert strategy == (0.0, 0.5, 0.5)
 
    def test_noise_exactly_at_the_tolerance_is_cleaned(self) -> None:
        strategy = _to_strategy(self._raw(-NUMERICAL_NOISE_TOLERANCE, 0.5, 0.5))
        assert strategy[0] == 0.0
 
    def test_exact_zero_and_positive_values_are_kept(self) -> None:
        assert _to_strategy(self._raw(0.0, 0.25, 0.75)) == (0.0, 0.25, 0.75)
 
    def test_vector_is_renormalized_after_cleaning(self) -> None:
        strategy = _to_strategy(self._raw(2.0, 1.0, 1.0))
        assert strategy == pytest.approx((0.5, 0.25, 0.25))
        assert math.fsum(strategy) == pytest.approx(1.0, abs=1e-15)
 
    def test_small_sum_error_from_the_solver_is_absorbed(self) -> None:
        offset = PROBABILITY_TOLERANCE / 2
        strategy = _to_strategy(self._raw(1 / 3 + offset, 1 / 3, 1 / 3))
        assert math.fsum(strategy) == pytest.approx(1.0, abs=1e-15)
 
    @pytest.mark.parametrize("residual", [1e-6, 10 * PROBABILITY_TOLERANCE])
    def test_marked_negative_value_is_rejected(self, residual: float) -> None:
        with pytest.raises(ValueError, match="positives"):
            _to_strategy(self._raw(-residual, 0.5, 0.5 + residual))
 
    @pytest.mark.parametrize(
        "raw",
        [(NAN, 0.5, 0.5), (INF, 0.5, 0.5), (0.0, 0.0, 0.0)],
        ids=["nan", "inf", "all-zero"],
    )
    def test_degenerate_vector_is_rejected(self, raw: tuple[float, ...]) -> None:
        with np.errstate(all="ignore"), pytest.raises(ValueError):
            _to_strategy(self._raw(*raw))
 
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "W1 : le nettoyage (NUMERICAL_NOISE_TOLERANCE = 1e-12) est plus "
            "strict que la validation (PROBABILITY_TOLERANCE = 1e-9) ; un "
            "résidu négatif intermédiaire est renvoyé tel quel et fait échouer "
            "numpy.random.Generator.choice dans l'agent Minimax."
        ),
    )
    def test_known_weakness_intermediate_negative_residual_is_kept(self) -> None:
        residual = 100 * NUMERICAL_NOISE_TOLERANCE
        assert NUMERICAL_NOISE_TOLERANCE < residual < PROBABILITY_TOLERANCE
        strategy = _to_strategy(self._raw(-residual, 0.5, 0.5 + residual))
        assert min(strategy) >= 0.0
 
 
class TestBestResponse:
    """``best_response`` : meilleure réponse pure et règle de départage."""
 
    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_returns_an_action(self, role: Role) -> None:
        assert isinstance(best_response(REFERENCE, role, UNIFORM), Action)
 
    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_against_uniform_all_actions_tie_and_left_wins(self, role: Role) -> None:
        """Contrat d'API §11 : égalité parfaite, indice le plus faible."""
        assert best_response(REFERENCE, role, UNIFORM) is Action.LEFT
 
    @pytest.mark.parametrize(
        ("role", "opponent_action", "expected"),
        [
            pytest.param(
                role,
                opponent,
                expected,
                id=f"{role.value}_vs_{opponent.name.lower()}",
            )
            for role, opponent, expected in [
                # Attaquant : éviter la zone couverte ; égalité -> indice faible.
                (Role.ATTACKER, Action.LEFT, Action.CENTER),
                (Role.ATTACKER, Action.CENTER, Action.LEFT),
                (Role.ATTACKER, Action.RIGHT, Action.LEFT),
                # Gardien : plonger dans la zone la plus tirée.
                (Role.GOALKEEPER, Action.LEFT, Action.LEFT),
                (Role.GOALKEEPER, Action.CENTER, Action.CENTER),
                (Role.GOALKEEPER, Action.RIGHT, Action.RIGHT),
            ]
        ],
    )
    def test_against_pure_opponent(
        self, role: Role, opponent_action: Action, expected: Action
    ) -> None:
        response = best_response(REFERENCE, role, _pure(opponent_action))
        assert response is expected
 
    @pytest.mark.parametrize(
        ("attacker_strategy", "expected"),
        [
            ((0.6, 0.3, 0.1), Action.LEFT),
            ((0.3, 0.6, 0.1), Action.CENTER),
            ((0.1, 0.3, 0.6), Action.RIGHT),
        ],
    )
    def test_goalkeeper_dives_where_the_attacker_shoots_most(
        self, attacker_strategy: Strategy, expected: Action
    ) -> None:
        response = best_response(REFERENCE, Role.GOALKEEPER, attacker_strategy)
        assert response is expected
 
    @pytest.mark.parametrize(
        ("goalkeeper_strategy", "expected"),
        [
            ((0.1, 0.3, 0.6), Action.LEFT),
            ((0.6, 0.1, 0.3), Action.CENTER),
            ((0.6, 0.3, 0.1), Action.RIGHT),
        ],
    )
    def test_attacker_shoots_where_the_goalkeeper_dives_least(
        self, goalkeeper_strategy: Strategy, expected: Action
    ) -> None:
        response = best_response(REFERENCE, Role.ATTACKER, goalkeeper_strategy)
        assert response is expected
 
    def test_empirical_style_strategy_is_accepted(self) -> None:
        """Une stratégie empirique (fréquences) est une entrée valide."""
        strategy = (4 / 10, 3 / 10, 3 / 10)
        assert best_response(REFERENCE, Role.GOALKEEPER, strategy) is Action.LEFT
 
 
class TestBestResponseTieBreak:
    """Règle : « en cas d'égalité, choisir l'action d'indice le plus faible »."""
 
    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    @pytest.mark.parametrize("opponent", list(Action), ids=lambda a: a.name)
    def test_full_tie_returns_left(self, role: Role, opponent: Action) -> None:
        assert best_response(ZERO_MATRIX, role, _pure(opponent)) is Action.LEFT
 
    def test_attacker_tie_between_center_and_right_returns_center(self) -> None:
        matrix: PayoffMatrix = (
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        )
        assert best_response(matrix, Role.ATTACKER, UNIFORM) is Action.CENTER
 
    def test_goalkeeper_tie_between_center_and_right_returns_center(self) -> None:
        matrix: PayoffMatrix = (
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        )
        assert best_response(matrix, Role.GOALKEEPER, UNIFORM) is Action.CENTER
 
    def test_attacker_unique_maximum_on_last_index_is_found(self) -> None:
        matrix: PayoffMatrix = (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        )
        assert best_response(matrix, Role.ATTACKER, UNIFORM) is Action.RIGHT
 
    def test_goalkeeper_unique_maximum_on_last_index_is_found(self) -> None:
        matrix: PayoffMatrix = (
            (1.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        )
        assert best_response(matrix, Role.GOALKEEPER, UNIFORM) is Action.RIGHT
 
    def test_gap_below_noise_tolerance_counts_as_a_tie(self) -> None:
        delta = NUMERICAL_NOISE_TOLERANCE / 2
        matrix: PayoffMatrix = (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0 + delta, 0.0, 0.0),
        )
        response = best_response(matrix, Role.ATTACKER, _pure(Action.LEFT))
        assert response is Action.LEFT
 
    def test_gap_above_noise_tolerance_is_a_real_difference(self) -> None:
        delta = 100 * NUMERICAL_NOISE_TOLERANCE
        matrix: PayoffMatrix = (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0 + delta, 0.0, 0.0),
        )
        response = best_response(matrix, Role.ATTACKER, _pure(Action.LEFT))
        assert response is Action.RIGHT
 
 
INVALID_STRATEGIES: Final[dict[str, Strategy]] = {
    "too_short": (0.5, 0.5),
    "too_long": (0.25,) * 4,
    "negative": (-0.1, 0.6, 0.5),
    "sum_above_one": (0.5, 0.5, 0.5),
    "sum_below_one": (0.2, 0.2, 0.2),
    "nan": (NAN, 0.5, 0.5),
    "inf": (INF, 0.0, 0.0),
}
 
 
class TestBestResponseValidation:
    """Entrées invalides de ``best_response``."""
 
    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    @pytest.mark.parametrize(
        "strategy", list(INVALID_STRATEGIES.values()), ids=list(INVALID_STRATEGIES)
    )
    def test_rejects_invalid_opponent_strategy(
        self, role: Role, strategy: Strategy
    ) -> None:
        with pytest.raises(ValueError):
            best_response(REFERENCE, role, strategy)
 
    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    @pytest.mark.parametrize(
        "matrix", [INVALID_MATRICES["2x2"], INVALID_MATRICES["ragged"]], ids=str
    )
    def test_shape_mismatch_raises_value_error(
        self, role: Role, matrix: PayoffMatrix
    ) -> None:
        """Garde-fou de NumPy : ``best_response`` ne valide pas la matrice."""
        with pytest.raises(ValueError):
            best_response(matrix, role, UNIFORM)
 
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "W2 : best_response ne valide pas la matrice (« déjà validée »). "
            "Une matrice NaN lève IndexError au lieu de ValueError et une "
            "matrice 3x4 est acceptée silencieusement pour le gardien."
        ),
    )
    @pytest.mark.parametrize(
        ("role", "matrix"),
        [
            pytest.param(
                Role.ATTACKER,
                ((NAN, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0)),
                id="nan-attacker",
            ),
            pytest.param(
                Role.GOALKEEPER,
                ((NAN, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0)),
                id="nan-goalkeeper",
            ),
            pytest.param(
                Role.GOALKEEPER,
                ((1.0, 2.0, 3.0, 4.0),) * 3,
                id="3x4-goalkeeper",
            ),
        ],
    )
    def test_known_weakness_invalid_matrix_should_raise_value_error(
        self, role: Role, matrix: PayoffMatrix
    ) -> None:
        with pytest.raises(ValueError):
            best_response(matrix, role, UNIFORM)
 
 
class TestNumericalTolerances:
    """Robustesse numérique : une petite erreur ne doit pas provoquer d'échec."""
 
    def test_tolerances_are_ordered_from_noise_to_equilibrium(self) -> None:
        """Bruit < validation des probabilités < comparaison à la théorie."""
        assert (
            NUMERICAL_NOISE_TOLERANCE < PROBABILITY_TOLERANCE < EQUILIBRIUM_TOLERANCE
        )
 
    # -- PROBABILITY_TOLERANCE ------------------------------------------------
 
    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_sum_error_within_probability_tolerance_is_accepted(
        self, role: Role
    ) -> None:
        strategy = (0.5, 0.5, PROBABILITY_TOLERANCE / 2)
        assert isinstance(best_response(REFERENCE, role, strategy), Action)
 
    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_sum_error_beyond_probability_tolerance_is_rejected(
        self, role: Role
    ) -> None:
        strategy = (0.5, 0.5, 10 * PROBABILITY_TOLERANCE)
        with pytest.raises(ValueError, match="sommer à 1"):
            best_response(REFERENCE, role, strategy)
 
    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_negative_noise_within_probability_tolerance_is_accepted(
        self, role: Role
    ) -> None:
        noise = PROBABILITY_TOLERANCE / 2
        strategy = (-noise, 0.5, 0.5 + noise)
        assert isinstance(best_response(REFERENCE, role, strategy), Action)
 
    @pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
    def test_negative_value_beyond_probability_tolerance_is_rejected(
        self, role: Role
    ) -> None:
        noise = 10 * PROBABILITY_TOLERANCE
        strategy = (-noise, 0.5, 0.5 + noise)
        with pytest.raises(ValueError, match="positives"):
            best_response(REFERENCE, role, strategy)
 
    # -- NUMERICAL_NOISE_TOLERANCE -------------------------------------------
 
    def test_floating_point_noise_does_not_break_the_uniform_tie(self) -> None:
        """Les écarts d'arrondi (~1e-16) sur la stratégie uniforme sont ignorés."""
        strategy = (1 / 3 + 1e-16, 1 / 3 - 1e-16, 1 / 3)
        assert best_response(REFERENCE, Role.ATTACKER, strategy) is Action.LEFT
 
    # -- EQUILIBRIUM_TOLERANCE -----------------------------------------------
 
    def test_tiny_perturbation_of_the_matrix_keeps_the_equilibrium(self) -> None:
        """Une perturbation de l'ordre de 1e-10 ne change pas la solution."""
        perturbed = tuple(
            tuple(
                value + 1e-10 * ((3 * i + j) % 4 - 1.5)
                for j, value in enumerate(row)
            )
            for i, row in enumerate(REFERENCE)
        )
        equilibrium = solve_equilibrium(perturbed)
        assert equilibrium.attacker_strategy == pytest.approx(
            UNIFORM, abs=EQUILIBRIUM_TOLERANCE
        )
        assert equilibrium.goalkeeper_strategy == pytest.approx(
            UNIFORM, abs=EQUILIBRIUM_TOLERANCE
        )
        assert equilibrium.value == pytest.approx(1 / 3, abs=EQUILIBRIUM_TOLERANCE)
 
    @pytest.mark.parametrize("matrix", GAME_MATRICES, ids=GAME_IDS)
    def test_solver_output_never_contains_a_negative_probability(
        self, matrix: PayoffMatrix
    ) -> None:
        equilibrium = solve_equilibrium(matrix)
        assert min(equilibrium.attacker_strategy) >= 0.0
        assert min(equilibrium.goalkeeper_strategy) >= 0.0
 