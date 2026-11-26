"""Small independent determinant oracle for closed-shell symbolic tests.

This module deliberately uses only Python integers and dictionaries.  It does
not import SymPy, Drudge, or the future closed-shell symbolic engine.  A
determinant is a bit string over interleaved spin orbitals

    (0 alpha, 0 beta, 1 alpha, 1 beta, ...).

Operator sequences are written in their usual left-to-right order and are
applied to kets from right to left.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TypeAlias


Coefficient: TypeAlias = complex | float | int
State: TypeAlias = dict[int, complex]
FermionOperator: TypeAlias = tuple[str, int]
SpinFreeOperator: TypeAlias = tuple[str, tuple[int, ...]]


def _phase(determinant: int, orbital: int) -> int:
    """Return the fermionic phase for acting at ``orbital``."""

    lower_occupations = determinant & ((1 << orbital) - 1)
    return -1 if lower_occupations.bit_count() % 2 else 1


def annihilate(determinant: int, orbital: int) -> tuple[int, int] | None:
    """Apply an annihilator and return ``(phase, determinant)`` or ``None``."""

    mask = 1 << orbital
    if determinant & mask == 0:
        return None
    return _phase(determinant, orbital), determinant ^ mask


def create(determinant: int, orbital: int) -> tuple[int, int] | None:
    """Apply a creator and return ``(phase, determinant)`` or ``None``."""

    mask = 1 << orbital
    if determinant & mask:
        return None
    return _phase(determinant, orbital), determinant | mask


def _add(target: State, determinant: int, value: Coefficient) -> None:
    updated = target.get(determinant, 0.0j) + value
    if updated == 0:
        target.pop(determinant, None)
    else:
        target[determinant] = complex(updated)


def apply_fermion_sequence(
    state: Mapping[int, Coefficient],
    operators: Sequence[FermionOperator],
) -> State:
    """Apply a left-to-right fermion-operator sequence to ``state``."""

    result: State = {}
    for determinant, coefficient in state.items():
        current_determinant = determinant
        current_coefficient: Coefficient = coefficient
        for kind, orbital in reversed(operators):
            if kind == "create":
                action = create(current_determinant, orbital)
            elif kind == "annihilate":
                action = annihilate(current_determinant, orbital)
            else:
                raise ValueError(f"unknown fermion operator {kind!r}")
            if action is None:
                break
            phase, current_determinant = action
            current_coefficient *= phase
        else:
            _add(result, current_determinant, current_coefficient)
    return result


def spin_orbital(spatial_orbital: int, spin: int) -> int:
    """Map a spatial orbital and spin ``0/1`` to an interleaved spin orbital."""

    if spatial_orbital < 0:
        raise ValueError("spatial orbital indices must be nonnegative")
    if spin not in (0, 1):
        raise ValueError("spin must be 0 (alpha) or 1 (beta)")
    return 2 * spatial_orbital + spin


def apply_E(
    state: Mapping[int, Coefficient],
    p: int,
    q: int,
) -> State:
    """Apply ``E_pq = sum_sigma a^dagger_(p sigma) a_(q sigma)``."""

    result: State = {}
    for spin in (0, 1):
        contribution = apply_fermion_sequence(
            state,
            (
                ("create", spin_orbital(p, spin)),
                ("annihilate", spin_orbital(q, spin)),
            ),
        )
        for determinant, coefficient in contribution.items():
            _add(result, determinant, coefficient)
    return result


def apply_e(
    state: Mapping[int, Coefficient],
    p: int,
    q: int,
    r: int,
    s: int,
) -> State:
    """Apply ``e_pqrs`` in its explicit spin-orbital definition."""

    result: State = {}
    for sigma in (0, 1):
        for tau in (0, 1):
            contribution = apply_fermion_sequence(
                state,
                (
                    ("create", spin_orbital(p, sigma)),
                    ("create", spin_orbital(q, tau)),
                    ("annihilate", spin_orbital(s, tau)),
                    ("annihilate", spin_orbital(r, sigma)),
                ),
            )
            for determinant, coefficient in contribution.items():
                _add(result, determinant, coefficient)
    return result


def apply_spin_free_operator(
    state: Mapping[int, Coefficient],
    operator: SpinFreeOperator,
) -> State:
    """Apply an encoded ``("E", indices)`` or ``("e", indices)`` operator."""

    kind, indices = operator
    if kind == "E" and len(indices) == 2:
        return apply_E(state, *indices)
    if kind == "e" and len(indices) == 4:
        return apply_e(state, *indices)
    raise ValueError(f"invalid spin-free operator {operator!r}")


def apply_operator_word(
    state: Mapping[int, Coefficient],
    operators: Sequence[SpinFreeOperator],
) -> State:
    """Apply a left-to-right spin-free operator word to ``state``."""

    result = {determinant: complex(value) for determinant, value in state.items()}
    for operator in reversed(operators):
        result = apply_spin_free_operator(result, operator)
        if not result:
            break
    return result


def scale_state(state: Mapping[int, Coefficient], factor: Coefficient) -> State:
    """Return ``factor * state`` with exact zeros removed."""

    result: State = {}
    for determinant, coefficient in state.items():
        _add(result, determinant, factor * coefficient)
    return result


def add_states(*states: Mapping[int, Coefficient]) -> State:
    """Return the sum of any number of determinant states."""

    result: State = {}
    for state in states:
        for determinant, coefficient in state.items():
            _add(result, determinant, coefficient)
    return result


def basis_states(n_spatial: int) -> Iterable[State]:
    """Yield every Fock-space basis state for ``n_spatial`` orbitals."""

    if n_spatial < 0:
        raise ValueError("n_spatial must be nonnegative")
    for determinant in range(1 << (2 * n_spatial)):
        yield {determinant: 1.0 + 0.0j}


def closed_shell_reference(n_spatial: int, n_occupied: int) -> State:
    """Return the determinant with the first ``n_occupied`` orbitals doubly filled."""

    if not 0 <= n_occupied <= n_spatial:
        raise ValueError("require 0 <= n_occupied <= n_spatial")
    determinant = 0
    for orbital in range(n_occupied):
        determinant |= 1 << spin_orbital(orbital, 0)
        determinant |= 1 << spin_orbital(orbital, 1)
    return {determinant: 1.0 + 0.0j}


def bra_ket(
    bra: Mapping[int, Coefficient],
    ket: Mapping[int, Coefficient],
) -> complex:
    """Return the standard determinant-basis inner product."""

    return sum(
        complex(coefficient).conjugate() * ket.get(determinant, 0.0j)
        for determinant, coefficient in bra.items()
    )
