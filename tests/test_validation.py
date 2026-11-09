"""Tests de ``theory.validation`` : stratégies et matrices des gains.

Chaque test est indépendant et déterministe : aucune donnée aléatoire,
aucun état partagé entre les tests.
"""

import pytest
from game.constants import PROBABILITY_TOLERANCE
from game.models import PayoffMatrix, Strategy
from theory.validation import validate_payoff_matrix, validate_strategy

NAN = float("nan")
INF = float("inf")

VALID_MATRIX: PayoffMatrix = (
    (-1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, -1.0),
)


class TestValidateStrategyAccepted:
    """Distributions de probabilités valides."""

    @pytest.mark.parametrize(
        "strategy",
        [
            (1 / 3, 1 / 3, 1 / 3),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.0),
            (0.2, 0.3, 0.5),
        ],
        ids=["uniform", "pure-left", "pure-center", "pure-right", "half", "mixed"],
    )
    def test_accepts_valid_strategy(self, strategy: Strategy) -> None:
        validate_strategy(strategy)

    def test_accepts_sum_within_tolerance(self) -> None:
        strategy = (0.5, 0.5, PROBABILITY_TOLERANCE / 2)
        validate_strategy(strategy)

    def test_accepts_tiny_negative_noise_within_tolerance(self) -> None:
        epsilon = PROBABILITY_TOLERANCE / 2
        strategy = (-epsilon, 0.5, 0.5 + epsilon)
        validate_strategy(strategy)


class TestValidateStrategyRejected:
    """Distributions de probabilités invalides."""

    @pytest.mark.parametrize(
        "strategy",
        [(), (1.0,), (0.5, 0.5), (0.25, 0.25, 0.25, 0.25)],
        ids=["empty", "one", "two", "four"],
    )
    def test_rejects_wrong_length(self, strategy: Strategy) -> None:
        with pytest.raises(ValueError, match="doit contenir"):
            validate_strategy(strategy)

    @pytest.mark.parametrize(
        "strategy",
        [(0.5, 0.5, 0.5), (0.2, 0.2, 0.2), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
        ids=["sum-1.5", "sum-0.6", "all-zero", "sum-3"],
    )
    def test_rejects_sum_different_from_one(self, strategy: Strategy) -> None:
        with pytest.raises(ValueError, match="sommer à 1"):
            validate_strategy(strategy)

    def test_rejects_sum_just_outside_tolerance(self) -> None:
        strategy = (0.5, 0.5, PROBABILITY_TOLERANCE * 10)
        with pytest.raises(ValueError, match="sommer à 1"):
            validate_strategy(strategy)

    @pytest.mark.parametrize(
        "strategy",
        [(-0.1, 0.6, 0.5), (1.5, -0.25, -0.25), (-1.0, 1.0, 1.0)],
        ids=["small-negative", "two-negatives", "minus-one"],
    )
    def test_rejects_negative_value(self, strategy: Strategy) -> None:
        with pytest.raises(ValueError, match="positives"):
            validate_strategy(strategy)

    @pytest.mark.parametrize(
        "strategy",
        [(NAN, 0.5, 0.5), (0.5, NAN, 0.5), (0.5, 0.5, NAN)],
        ids=["nan-first", "nan-middle", "nan-last"],
    )
    def test_rejects_nan(self, strategy: Strategy) -> None:
        with pytest.raises(ValueError, match="finies"):
            validate_strategy(strategy)

    @pytest.mark.parametrize(
        "strategy",
        [(INF, 0.0, 0.0), (0.0, -INF, 0.0), (0.5, 0.5, INF)],
        ids=["inf", "-inf", "inf-last"],
    )
    def test_rejects_infinity(self, strategy: Strategy) -> None:
        with pytest.raises(ValueError, match="finies"):
            validate_strategy(strategy)


class TestValidatePayoffMatrixAccepted:
    """Matrices exploitables par le moteur."""

    def test_accepts_reference_matrix(self) -> None:
        validate_payoff_matrix(VALID_MATRIX)

    def test_accepts_any_finite_values(self) -> None:
        matrix: PayoffMatrix = (
            (0.0, 2.5, -3.0),
            (10.0, 0.0, 0.5),
            (-7.0, 1e6, 0.0),
        )
        validate_payoff_matrix(matrix)

    def test_accepts_all_zero_matrix(self) -> None:
        validate_payoff_matrix(((0.0,) * 3,) * 3)


class TestValidatePayoffMatrixRejected:
    """Matrices invalides : forme incorrecte ou valeurs non finies."""

    @pytest.mark.parametrize(
        "matrix",
        [
            (),
            ((1.0, 1.0, 1.0),),
            ((1.0, 1.0), (1.0, 1.0)),
            ((1.0,) * 4,) * 4,
        ],
        ids=["empty", "one-row", "2x2", "4x4"],
    )
    def test_rejects_non_3x3_matrix(self, matrix: PayoffMatrix) -> None:
        with pytest.raises(ValueError, match="taille 3x3"):
            validate_payoff_matrix(matrix)

    @pytest.mark.parametrize(
        "matrix",
        [
            ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
            ((1.0, 1.0, 1.0), (1.0, 1.0), (1.0, 1.0, 1.0)),
            ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)),
            ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), ()),
        ],
        ids=["two-rows", "short-middle-row", "long-last-row", "empty-last-row"],
    )
    def test_rejects_non_square_matrix(self, matrix: PayoffMatrix) -> None:
        with pytest.raises(ValueError, match="taille 3x3"):
            validate_payoff_matrix(matrix)

    @pytest.mark.parametrize("row", [0, 1, 2])
    @pytest.mark.parametrize("col", [0, 1, 2])
    @pytest.mark.parametrize("bad", [NAN, INF, -INF], ids=["nan", "inf", "-inf"])
    def test_rejects_non_finite_value_anywhere(
        self, row: int, col: int, bad: float
    ) -> None:
        cells = [list(line) for line in VALID_MATRIX]
        cells[row][col] = bad
        matrix: PayoffMatrix = tuple(tuple(line) for line in cells)
        with pytest.raises(ValueError, match="finies"):
            validate_payoff_matrix(matrix)