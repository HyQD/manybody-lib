"""Small explicit-spin validation backend for closed-shell operator words."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from .expressions import Expression
from .indices import Index, OrbitalSpace
from .operators import OneBodyOperator, OperatorWord, TwoBodyOperator
from .scalars import ExactScalar, ZERO
from .tensors import Delta, OrbitalDimension


State: TypeAlias = dict[int, int]
FermionOperator: TypeAlias = tuple[str, int]


def _phase(determinant: int, orbital: int) -> int:
    return -1 if (determinant & ((1 << orbital) - 1)).bit_count() % 2 else 1


def _apply_fermion_sequence(
    state: Mapping[int, int],
    operators: Sequence[FermionOperator],
) -> State:
    result: State = {}
    for determinant, coefficient in state.items():
        current = determinant
        value = coefficient
        for kind, orbital in reversed(operators):
            mask = 1 << orbital
            if kind == "annihilate":
                if current & mask == 0:
                    break
            elif kind == "create":
                if current & mask:
                    break
            else:
                raise ValueError(f"unknown fermion operator {kind!r}")
            value *= _phase(current, orbital)
            current ^= mask
        else:
            updated = result.get(current, 0) + value
            if updated:
                result[current] = updated
            else:
                result.pop(current, None)
    return result


def _spin_orbital(spatial: int, spin: int) -> int:
    return 2 * spatial + spin


def _add_state(target: State, source: Mapping[int, int]) -> None:
    for determinant, coefficient in source.items():
        updated = target.get(determinant, 0) + coefficient
        if updated:
            target[determinant] = updated
        else:
            target.pop(determinant, None)


def _apply_one_body(state: Mapping[int, int], p: int, q: int) -> State:
    result: State = {}
    for spin in (0, 1):
        _add_state(
            result,
            _apply_fermion_sequence(
                state,
                (
                    ("create", _spin_orbital(p, spin)),
                    ("annihilate", _spin_orbital(q, spin)),
                ),
            ),
        )
    return result


def _apply_two_body(
    state: Mapping[int, int],
    p: int,
    q: int,
    r: int,
    s: int,
) -> State:
    result: State = {}
    for sigma in (0, 1):
        for tau in (0, 1):
            _add_state(
                result,
                _apply_fermion_sequence(
                    state,
                    (
                        ("create", _spin_orbital(p, sigma)),
                        ("create", _spin_orbital(q, tau)),
                        ("annihilate", _spin_orbital(s, tau)),
                        ("annihilate", _spin_orbital(r, sigma)),
                    ),
                ),
            )
    return result


def _validate_orbital_assignment(
    index: Index,
    orbital: int,
    n_spatial: int,
    n_occupied: int,
) -> None:
    if not isinstance(orbital, int) or not 0 <= orbital < n_spatial:
        raise ValueError(f"invalid spatial-orbital assignment for {index.name!r}")
    if index.space is OrbitalSpace.OCCUPIED and orbital >= n_occupied:
        raise ValueError(
            f"occupied index {index.name!r} maps outside the occupied space"
        )
    if index.space is OrbitalSpace.VIRTUAL and orbital < n_occupied:
        raise ValueError(
            f"virtual index {index.name!r} maps outside the virtual space"
        )
    if index.space is OrbitalSpace.SPIN:
        raise ValueError("spin indices cannot label spin-free spatial operators")


def _assigned_operator_indices(
    operator,
    orbital_assignment: Mapping[Index, int],
    n_spatial: int,
    n_occupied: int,
) -> tuple[int, ...]:
    assigned = []
    for index in operator.indices:
        try:
            orbital = orbital_assignment[index]
        except KeyError as error:
            raise KeyError(f"missing assignment for index {index!r}") from error
        _validate_orbital_assignment(index, orbital, n_spatial, n_occupied)
        assigned.append(orbital)
    return tuple(assigned)


def explicit_spin_reference_expectation(
    word: OperatorWord,
    orbital_assignment: Mapping[Index, int],
    *,
    n_spatial: int,
    n_occupied: int,
) -> int:
    """Evaluate a short spin-free word by explicit alpha/beta expansion."""

    if not isinstance(word, OperatorWord):
        raise TypeError("word must be an OperatorWord")
    if not 0 <= n_occupied <= n_spatial:
        raise ValueError("require 0 <= n_occupied <= n_spatial")

    determinant = sum(
        (1 << _spin_orbital(orbital, spin))
        for orbital in range(n_occupied)
        for spin in (0, 1)
    )
    state: State = {determinant: 1}
    for operator in reversed(word.operators):
        indices = _assigned_operator_indices(
            operator,
            orbital_assignment,
            n_spatial,
            n_occupied,
        )
        if isinstance(operator, OneBodyOperator):
            state = _apply_one_body(state, *indices)
        elif isinstance(operator, TwoBodyOperator):
            state = _apply_two_body(state, *indices)
        else:
            raise TypeError(f"unsupported operator {type(operator).__name__}")
        if not state:
            return 0
    return state.get(determinant, 0)


def evaluate_scalar_expression(
    expression: Expression,
    orbital_assignment: Mapping[Index, int],
    *,
    n_spatial: int,
    n_occupied: int,
) -> ExactScalar:
    """Evaluate a delta/dimension scalar expression for one orbital assignment."""

    if not isinstance(expression, Expression):
        raise TypeError("expression must be an Expression")
    if not 0 <= n_occupied <= n_spatial:
        raise ValueError("require 0 <= n_occupied <= n_spatial")
    dimensions = {
        OrbitalSpace.OCCUPIED: n_occupied,
        OrbitalSpace.VIRTUAL: n_spatial - n_occupied,
        OrbitalSpace.GENERAL: n_spatial,
        OrbitalSpace.SPIN: 2,
    }

    total = ZERO
    for term in expression:
        if term.operators:
            raise ValueError("expression is not scalar")
        value = term.coefficient
        for factor in term.factors:
            if isinstance(factor, Delta):
                try:
                    left = orbital_assignment[factor.left]
                    right = orbital_assignment[factor.right]
                except KeyError as error:
                    raise KeyError(
                        f"missing assignment for delta index {error.args[0]!r}"
                    ) from error
                value *= int(left == right)
            elif isinstance(factor, OrbitalDimension):
                value *= dimensions[factor.space]
            else:
                raise TypeError(
                    "validation evaluation supports only deltas and dimensions; "
                    f"got {type(factor).__name__}"
                )
        total += value
    return total
