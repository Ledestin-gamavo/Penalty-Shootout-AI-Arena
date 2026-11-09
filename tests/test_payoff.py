"""Tests de ``game.payoff`` : matrice de référence et lecture des gains.

Les valeurs attendues sont écrites en littéral (±1) et non recalculées à
partir des constantes : le test vérifie ainsi le modèle du cahier des
charges, pas seulement la cohérence interne du code.
"""

import pytest
from game.actions import Action
from game.constants import N_ACTIONS, REFERENCE_GAME_VALUE, REWARD_GOAL, REWARD_SAVE
from game.models import PayoffMatrix
from game.payoff import (attacker_reward, build_reference_matrix, resolve_payoff_matrix)

# colonnes = gardien, dans l'ordre gauche, centre, droite.
EXPECTED_REFERENCE: PayoffMatrix = (
    (-1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, -1.0),
)

# Matrice à valeurs toutes distinctes : détecte toute inversion de lignes et de colonnes.
DISTINCT_MATRIX: PayoffMatrix = (
    (1.0, 2.0, 3.0),
    (4.0, 5.0, 6.0),
    (7.0, 8.0, 9.0),
)


class TestBuildReferenceMatrix:
    """Construction de la matrice de référence."""

    def test_has_expected_dimensions(self) -> None:
        matrix = build_reference_matrix()
        assert len(matrix) == N_ACTIONS
        assert all(len(row) == N_ACTIONS for row in matrix)

    def test_has_expected_values(self) -> None:
        assert build_reference_matrix() == EXPECTED_REFERENCE

    def test_is_immutable_tuple_of_tuples(self) -> None:
        matrix = build_reference_matrix()
        assert isinstance(matrix, tuple)
        assert all(isinstance(row, tuple) for row in matrix)

    def test_save_is_on_the_diagonal_only(self) -> None:
        matrix = build_reference_matrix()
        for attacker in Action:
            for goalkeeper in Action:
                expected = -1.0 if attacker == goalkeeper else 1.0
                assert matrix[attacker][goalkeeper] == expected

    def test_each_row_and_column_has_exactly_one_save(self) -> None:
        matrix = build_reference_matrix()
        for index in range(N_ACTIONS):
            row = matrix[index]
            column = tuple(matrix[i][index] for i in range(N_ACTIONS))
            assert row.count(-1.0) == 1
            assert column.count(-1.0) == 1

    def test_uniform_play_gives_game_value(self) -> None:
        """Sous jeu uniforme, le gain moyen vaut 1/3 (valeur du jeu)."""
        matrix = build_reference_matrix()
        mean = sum(sum(row) for row in matrix) / (N_ACTIONS**2)
        assert mean == pytest.approx(1 / 3)
        assert mean == pytest.approx(REFERENCE_GAME_VALUE)

    def test_rewards_match_constants(self) -> None:
        """Les constantes du moteur restent alignées sur le modèle ±1."""
        assert REWARD_GOAL == 1.0
        assert REWARD_SAVE == -1.0

    def test_returns_equal_matrix_on_each_call(self) -> None:
        assert build_reference_matrix() == build_reference_matrix()


class TestResolvePayoffMatrix:
    """Règle « None signifie matrice de référence »."""

    def test_none_returns_reference_matrix(self) -> None:
        assert resolve_payoff_matrix(None) == EXPECTED_REFERENCE

    def test_custom_matrix_is_returned_unchanged(self) -> None:
        resolved = resolve_payoff_matrix(DISTINCT_MATRIX)
        assert resolved is DISTINCT_MATRIX

    def test_reference_matrix_passed_explicitly_is_kept(self) -> None:
        assert resolve_payoff_matrix(EXPECTED_REFERENCE) == EXPECTED_REFERENCE


class TestAttackerReward:
    """Lecture d'un gain : ``matrix[attaquant][gardien]``."""

    @pytest.mark.parametrize("attacker", list(Action), ids=lambda a: a.name)
    @pytest.mark.parametrize("goalkeeper", list(Action), ids=lambda a: a.name)
    def test_reference_matrix_all_combinations(
        self, attacker: Action, goalkeeper: Action
    ) -> None:
        expected = -1.0 if attacker == goalkeeper else 1.0
        assert attacker_reward(attacker, goalkeeper, EXPECTED_REFERENCE) == expected

    @pytest.mark.parametrize(
        ("attacker", "goalkeeper", "expected"),
        [
            (Action.LEFT, Action.LEFT, 1.0),
            (Action.LEFT, Action.CENTER, 2.0),
            (Action.LEFT, Action.RIGHT, 3.0),
            (Action.CENTER, Action.LEFT, 4.0),
            (Action.CENTER, Action.CENTER, 5.0),
            (Action.CENTER, Action.RIGHT, 6.0),
            (Action.RIGHT, Action.LEFT, 7.0),
            (Action.RIGHT, Action.CENTER, 8.0),
            (Action.RIGHT, Action.RIGHT, 9.0),
        ],
    )
    def test_orientation_rows_attacker_columns_goalkeeper(
        self, attacker: Action, goalkeeper: Action, expected: float
    ) -> None:
        assert attacker_reward(attacker, goalkeeper, DISTINCT_MATRIX) == expected

    def test_is_not_symmetric_in_its_arguments(self) -> None:
        forward = attacker_reward(Action.LEFT, Action.RIGHT, DISTINCT_MATRIX)
        swapped = attacker_reward(Action.RIGHT, Action.LEFT, DISTINCT_MATRIX)
        assert forward != swapped