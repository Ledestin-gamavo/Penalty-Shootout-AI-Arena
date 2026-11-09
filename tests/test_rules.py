"""Tests de ``game.rules`` : validation des actions, résolution, score, config.

Les valeurs attendues sont écrites en littéral (±1) pour vérifier le modèle
du cahier des charges et non une simple recopie du code testé.
"""

from itertools import product
from typing import cast
import pytest
from game.actions import Action
from game.models import AgentType, MatchConfig, PayoffMatrix, RoundResult
from game.rules import (compute_score, resolve_round, validate_action, validate_config,)

NAN = float("nan")
INF = float("inf")

REFERENCE: PayoffMatrix = (
    (-1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, -1.0),
)

CUSTOM_MATRIX: PayoffMatrix = (
    (1.0, 2.0, 3.0),
    (4.0, 5.0, 6.0),
    (7.0, 8.0, 9.0),
)


def _unsafe(value: object) -> Action:
    """Fait passer une valeur invalide pour une ``Action`` (typage strict)."""
    return cast(Action, value)


def _result(reward: float, round_number: int = 1) -> RoundResult:
    """Fabrique un ``RoundResult`` dont seul le gain compte pour le score."""
    return RoundResult(
        round_number=round_number,
        attacker_action=Action.LEFT,
        goalkeeper_action=Action.LEFT,
        attacker_reward=reward,
    )


def _config(payoff_matrix: PayoffMatrix | None) -> MatchConfig:
    """Fabrique une configuration IA contre IA valide."""
    return MatchConfig(
        num_rounds=10,
        attacker_agent=AgentType.MINIMAX,
        goalkeeper_agent=AgentType.DETERMINISTIC,
        payoff_matrix=payoff_matrix,
    )


class TestValidateAction:
    """Conversion d'une valeur en ``Action``."""

    @pytest.mark.parametrize("action", list(Action), ids=lambda a: a.name)
    def test_accepts_action_members(self, action: Action) -> None:
        assert validate_action(action) is action

    @pytest.mark.parametrize(
        ("index", "expected"),
        [(0, Action.LEFT), (1, Action.CENTER), (2, Action.RIGHT)],
    )
    def test_converts_valid_integer_indices(
        self, index: int, expected: Action
    ) -> None:
        result = validate_action(index)
        assert result is expected
        assert isinstance(result, Action)

    @pytest.mark.parametrize(
        "value",
        [-1, 3, 100, True, False, 1.0, 0.0, "0", "LEFT", None, [], (0,)],
        ids=repr,
    )
    def test_rejects_invalid_values(self, value: object) -> None:
        with pytest.raises(ValueError, match="Action invalide"):
            validate_action(value)

    def test_error_message_lists_allowed_actions(self) -> None:
        with pytest.raises(ValueError, match="LEFT.*CENTER.*RIGHT"):
            validate_action(7)


class TestResolveRound:
    """Résolution d'une manche à partir de deux actions révélées."""

    @pytest.mark.parametrize(
        ("attacker", "goalkeeper"),
        list(product(Action, Action)),
        ids=lambda a: a.name,
    )
    def test_nine_combinations_reference_matrix(
        self, attacker: Action, goalkeeper: Action
    ) -> None:
        result = resolve_round(3, attacker, goalkeeper, REFERENCE)
        is_save = attacker == goalkeeper
        assert result.round_number == 3
        assert result.attacker_action is attacker
        assert result.goalkeeper_action is goalkeeper
        assert result.attacker_reward == (-1.0 if is_save else 1.0)
        assert result.is_goal is (not is_save)

    @pytest.mark.parametrize(
        ("attacker", "goalkeeper"),
        list(product(Action, Action)),
        ids=lambda a: a.name,
    )
    def test_zero_sum_for_every_combination(
        self, attacker: Action, goalkeeper: Action
    ) -> None:
        result = resolve_round(1, attacker, goalkeeper, REFERENCE)
        assert result.attacker_reward + result.goalkeeper_reward == 0

    def test_uses_the_provided_matrix(self) -> None:
        result = resolve_round(1, Action.CENTER, Action.RIGHT, CUSTOM_MATRIX)
        assert result.attacker_reward == 6.0

    def test_normalizes_integer_actions_to_action(self) -> None:
        result = resolve_round(1, _unsafe(2), _unsafe(0), REFERENCE)
        assert result.attacker_action is Action.RIGHT
        assert result.goalkeeper_action is Action.LEFT

    @pytest.mark.parametrize("round_number", [0, -1, -100])
    def test_rejects_round_number_below_one(self, round_number: int) -> None:
        with pytest.raises(ValueError, match="round_number"):
            resolve_round(round_number, Action.LEFT, Action.LEFT, REFERENCE)

    @pytest.mark.parametrize("bad", [3, -1, True, None, "LEFT"], ids=repr)
    def test_rejects_invalid_attacker_action(self, bad: object) -> None:
        with pytest.raises(ValueError, match="Action invalide"):
            resolve_round(1, _unsafe(bad), Action.LEFT, REFERENCE)

    @pytest.mark.parametrize("bad", [3, -1, True, None, "LEFT"], ids=repr)
    def test_rejects_invalid_goalkeeper_action(self, bad: object) -> None:
        with pytest.raises(ValueError, match="Action invalide"):
            resolve_round(1, Action.LEFT, _unsafe(bad), REFERENCE)


class TestComputeScore:
    """Score ``(buts, arrêts)`` d'une suite de manches."""

    def test_empty_sequence(self) -> None:
        assert compute_score([]) == (0, 0)

    def test_only_goals(self) -> None:
        results = [_result(1.0, n) for n in range(1, 5)]
        assert compute_score(results) == (4, 0)

    def test_only_saves(self) -> None:
        results = [_result(-1.0, n) for n in range(1, 4)]
        assert compute_score(results) == (0, 3)

    def test_mixed_results(self) -> None:
        results = [_result(r, n) for n, r in enumerate([1.0, -1.0, 1.0, 1.0, -1.0], 1)]
        assert compute_score(results) == (3, 2)

    def test_goals_plus_saves_equals_number_of_rounds(self) -> None:
        results = [_result(r) for r in (1.0, -1.0, -1.0, 1.0, 1.0, 1.0)]
        goals, saves = compute_score(results)
        assert goals + saves == len(results)

    def test_accepts_any_sequence_type(self) -> None:
        results = (_result(1.0), _result(-1.0))
        assert compute_score(results) == compute_score(list(results)) == (1, 1)

    def test_score_of_resolved_rounds(self) -> None:
        """Intégration légère : score d'une série de manches résolues."""
        results = [
            resolve_round(1, Action.LEFT, Action.LEFT, REFERENCE),  # arrêt
            resolve_round(2, Action.LEFT, Action.RIGHT, REFERENCE),  # but
            resolve_round(3, Action.CENTER, Action.CENTER, REFERENCE),  # arrêt
            resolve_round(4, Action.RIGHT, Action.LEFT, REFERENCE),  # but
        ]
        assert compute_score(results) == (2, 2)


class TestValidateConfig:
    """Vérification de la matrice personnalisée d'une configuration."""

    def test_accepts_default_reference_matrix(self) -> None:
        validate_config(_config(None))

    def test_accepts_explicit_valid_matrix(self) -> None:
        validate_config(_config(REFERENCE))

    def test_accepts_human_versus_ai_config(self) -> None:
        config = MatchConfig(
            num_rounds=5,
            attacker_agent=None,
            goalkeeper_agent=AgentType.FICTITIOUS_PLAY,
        )
        validate_config(config)

    @pytest.mark.parametrize(
        "matrix",
        [
            (),
            ((1.0, 1.0), (1.0, 1.0)),
            ((1.0,) * 4,) * 4,
            ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
            ((1.0, 1.0, 1.0), (1.0, 1.0), (1.0, 1.0, 1.0)),
        ],
        ids=["empty", "2x2", "4x4", "2-rows", "ragged"],
    )
    def test_rejects_wrong_shape(self, matrix: PayoffMatrix) -> None:
        with pytest.raises(ValueError, match="taille"):
            validate_config(_config(matrix))

    @pytest.mark.parametrize("bad", [NAN, INF, -INF], ids=["nan", "inf", "-inf"])
    def test_rejects_non_finite_values(self, bad: float) -> None:
        matrix: PayoffMatrix = (
            (-1.0, 1.0, 1.0),
            (1.0, bad, 1.0),
            (1.0, 1.0, -1.0),
        )
        with pytest.raises(ValueError, match="finies"):
            validate_config(_config(matrix))