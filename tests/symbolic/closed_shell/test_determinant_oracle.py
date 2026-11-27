"""Independent exact checks of the closed-shell substitution-operator algebra."""

from __future__ import annotations

import itertools

from tests.symbolic.closed_shell.determinant_oracle import (
    add_states,
    apply_E,
    apply_e,
    apply_operator_word,
    basis_states,
    closed_shell_reference,
    scale_state,
)


def _E(p: int, q: int):
    return "E", (p, q)


def _e(p: int, q: int, r: int, s: int):
    return "e", (p, q, r, s)


def _commutator(state, left, right):
    return add_states(
        apply_operator_word(state, (left, right)),
        scale_state(apply_operator_word(state, (right, left)), -1),
    )


def test_two_body_definition() -> None:
    for state in basis_states(2):
        for p, q, r, s in itertools.product(range(2), repeat=4):
            direct = apply_e(state, p, q, r, s)
            definition = apply_operator_word(state, (_E(p, r), _E(q, s)))
            if q == r:
                definition = add_states(
                    definition,
                    scale_state(apply_E(state, p, s), -1),
                )
            assert direct == definition


def test_one_body_commutator() -> None:
    for state in basis_states(2):
        for p, q, r, s in itertools.product(range(2), repeat=4):
            actual = _commutator(state, _E(p, q), _E(r, s))
            expected = {}
            if q == r:
                expected = add_states(expected, apply_E(state, p, s))
            if p == s:
                expected = add_states(
                    expected,
                    scale_state(apply_E(state, r, q), -1),
                )
            assert actual == expected


def test_two_body_one_body_commutator() -> None:
    for state in basis_states(2):
        for p, q, r, s, t, u in itertools.product(range(2), repeat=6):
            actual = _commutator(state, _e(p, q, r, s), _E(t, u))
            expected = {}
            if r == t:
                expected = add_states(expected, apply_e(state, p, q, u, s))
            if s == t:
                expected = add_states(expected, apply_e(state, p, q, r, u))
            if p == u:
                expected = add_states(
                    expected,
                    scale_state(apply_e(state, t, q, r, s), -1),
                )
            if q == u:
                expected = add_states(
                    expected,
                    scale_state(apply_e(state, p, t, r, s), -1),
                )
            assert actual == expected


def test_closed_shell_reference_termination_rules() -> None:
    reference = closed_shell_reference(n_spatial=3, n_occupied=2)

    for occupied in range(2):
        for virtual in range(2, 3):
            assert apply_E(reference, occupied, virtual) == {}
            assert apply_E(reference, virtual, virtual) == {}

    for i, j in itertools.product(range(2), repeat=2):
        expected = scale_state(reference, 2) if i == j else {}
        assert apply_E(reference, i, j) == expected
