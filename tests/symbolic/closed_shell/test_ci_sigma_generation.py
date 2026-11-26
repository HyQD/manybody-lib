"""CIS/CID/CISD symbolic sigma-vector tests for Milestone 3."""

from __future__ import annotations

from fractions import Fraction
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from sympy import Indexed, IndexedBase, KroneckerDelta, Mul, Symbol, sympify

from manybody_lib.configuration_interaction.closed_shell.cis import (
    compute_sigma as closed_cis_sigma,
)
from manybody_lib.configuration_interaction.closed_shell.cid import (
    compute_sigma as closed_cid_sigma,
)
from manybody_lib.configuration_interaction.closed_shell.cisd import (
    compute_sigma as closed_cisd_sigma,
)
from manybody_lib.configuration_interaction.spin_orbital.cis import (
    compute_sigma as spin_cis_sigma,
)
from manybody_lib.configuration_interaction.spin_orbital.cid import (
    compute_sigma as spin_cid_sigma,
)
from manybody_lib.configuration_interaction.spin_orbital.cisd import (
    compute_sigma as spin_cisd_sigma,
)
from manybody_lib.symbolic.closed_shell import (
    Delta,
    Expression,
    FOCK,
    OrbitalDimension,
    REFERENCE_ENERGY,
    TWO_ELECTRON,
    Tensor,
    Term,
    build_hjo_right_state,
    build_normal_ordered_hamiltonian,
    derive_right_sigma,
    factor_column_permutation,
    hjo_metric,
    occupied,
    reference_expectation,
    right_sigma_to_latex,
    virtual,
)
from manybody_lib.symbolic.closed_shell.manifolds.hjo import (
    DOUBLES_AMPLITUDE,
    REFERENCE_COEFFICIENT,
    SINGLES_AMPLITUDE,
)


FIXTURES = Path(__file__).with_name("fixtures")
_OCCUPIED_NAMES = frozenset("ijklmn")
_VIRTUAL_NAMES = frozenset("abcdefgh")


def _fixture_index(name: str, sum_spaces: dict[str, str]):
    if name in sum_spaces:
        space = sum_spaces[name]
        if space == "O":
            return occupied(name, dummy=True)
        if space == "V":
            return virtual(name, dummy=True)
        raise ValueError(f"unknown fixture range {space!r}")
    if name[0] in _OCCUPIED_NAMES:
        return occupied(name)
    if name[0] in _VIRTUAL_NAMES:
        return virtual(name)
    raise ValueError(f"cannot infer the orbital space of {name!r}")


def _fixture_expression(record: dict) -> Expression:
    indexed_bases = {
        name: IndexedBase(name)
        for name in ("C", "f", "u")
    }
    local_symbols = {
        **indexed_bases,
        "C0": Symbol("C0"),
        "Eref": Symbol("Eref"),
        "KroneckerDelta": KroneckerDelta,
    }
    terms = []
    for fixture_term in record["terms"]:
        sum_spaces = {
            item["index"]: item["range"]
            for item in fixture_term["sums"]
        }
        parsed = sympify(fixture_term["amplitude"], locals=local_symbols)
        coefficient, remainder = parsed.as_coeff_Mul()
        factors = []
        for factor in Mul.make_args(remainder):
            if isinstance(factor, Indexed):
                label = str(factor.base.label)
                indices = tuple(
                    _fixture_index(str(index), sum_spaces)
                    for index in factor.indices
                )
                if label == "C" and len(indices) == 2:
                    factors.append(SINGLES_AMPLITUDE(*indices))
                elif label == "C" and len(indices) == 4:
                    factors.append(DOUBLES_AMPLITUDE(*indices))
                elif label == "f":
                    factors.append(FOCK(*indices))
                elif label == "u":
                    factors.append(TWO_ELECTRON(*indices))
                else:
                    raise ValueError(f"unknown fixture tensor {label!r}")
            elif isinstance(factor, KroneckerDelta):
                left, right = (
                    _fixture_index(str(index), sum_spaces)
                    for index in factor.args
                )
                factors.append(Delta(left, right))
            elif factor == local_symbols["C0"]:
                factors.append(REFERENCE_COEFFICIENT())
            elif factor == local_symbols["Eref"]:
                factors.append(REFERENCE_ENERGY())
            else:
                raise TypeError(f"unsupported fixture factor {factor!r}")
        terms.append(
            Term(
                Fraction(int(coefficient.p), int(coefficient.q)),
                tuple(factors),
            )
        )
    return Expression.from_terms(terms)


def _fixture_sigma(method: str) -> dict[int, Expression]:
    data = json.loads((FIXTURES / f"hjo_{method}.json").read_text())
    result = {}
    for lhs, record in data["expressions"]["right_sigma"].items():
        if lhs == r"\sigma_{0}":
            rank = 0
        elif "^{ab}" in lhs:
            rank = 2
        else:
            rank = 1
        result[rank] = _fixture_expression(record)
    return result


def test_hjo_state_sectors_and_metrics() -> None:
    assert build_hjo_right_state("cis").ranks == (0, 1)
    assert build_hjo_right_state("cid").ranks == (0, 2)
    assert build_hjo_right_state("cisd").ranks == (0, 1, 2)

    a, b, c, d = (virtual(name) for name in "abcd")
    i, j, k, l = (occupied(name) for name in "ijkl")
    assert hjo_metric(1, (a, i), (b, j)) == Expression.from_terms(
        (Term(factors=(Delta(a, b), Delta(i, j))),)
    )
    assert hjo_metric(2, (a, b, i, j), (c, d, k, l)) == (
        Expression.from_terms(
            (
                Term(
                    factors=(
                        Delta(a, c),
                        Delta(b, d),
                        Delta(i, k),
                        Delta(j, l),
                    )
                ),
                Term(
                    factors=(
                        Delta(a, d),
                        Delta(b, c),
                        Delta(i, l),
                        Delta(j, k),
                    )
                ),
            )
        )
    )


def test_normal_ordered_hamiltonian_has_reference_energy_expectation() -> None:
    assert reference_expectation(build_normal_ordered_hamiltonian()) == (
        Expression.from_terms((Term(factors=(REFERENCE_ENERGY(),)),))
    )


@pytest.mark.parametrize("method", ("cis", "cid", "cisd"))
def test_generated_right_sigma_matches_expanded_drudge_fixture(method: str) -> None:
    generated = derive_right_sigma(method)
    expected = _fixture_sigma(method)

    assert set(generated.ranks) == set(expected)
    for rank, expression in generated.components:
        assert expression == expected[rank]


@pytest.mark.parametrize("method", ("cid", "cisd"))
def test_column_permutation_presentation_is_verified_and_smaller(method: str) -> None:
    expression = derive_right_sigma(method)[2]
    presentation = factor_column_permutation(
        expression,
        first_upper=virtual("a"),
        second_upper=virtual("b"),
        first_lower=occupied("i"),
        second_lower=occupied("j"),
        operator_latex=r"\mathcal{P}^{ab}_{ij}",
    )

    assert presentation.expand() == expression
    assert presentation.displayed_term_count < expression.term_count
    assert r"\mathcal{P}^{ab}_{ij}" in presentation.to_latex()


def test_sigma_latex_is_a_complete_align_environment() -> None:
    rendered = right_sigma_to_latex(derive_right_sigma("cisd"))
    assert rendered.startswith(r"\begin{align}")
    assert rendered.endswith(r"\end{align}")
    assert r"\sigma_{0}" in rendered
    assert r"\sigma^{a}_{i}" in rendered
    assert r"\sigma^{ab}_{ij}" in rendered
    assert r"\mathcal{P}^{ab}_{ij}" in rendered


def _index_values(index, n_occupied: int, n_virtual: int) -> range:
    if index.space.value == "occupied":
        return range(n_occupied)
    if index.space.value == "virtual":
        return range(n_occupied, n_occupied + n_virtual)
    raise ValueError(f"unsupported numerical index space {index.space}")


def _tensor_value(
    tensor: Tensor,
    assignment: dict,
    arrays: dict,
    n_occupied: int,
):
    value = arrays[tensor.symbol]
    if not tensor.indices:
        return value
    indices = tuple(assignment[index] for index in tensor.indices)
    if tensor.symbol in (SINGLES_AMPLITUDE, DOUBLES_AMPLITUDE):
        indices = tuple(
            index - n_occupied if orbital.space.value == "virtual" else index
            for orbital, index in zip(tensor.indices, indices, strict=True)
        )
    return value[indices]


def _evaluate_component(
    expression: Expression,
    external_indices: tuple,
    arrays: dict,
    n_occupied: int,
    n_virtual: int,
):
    shape = tuple(
        len(_index_values(index, n_occupied, n_virtual))
        for index in external_indices
    )
    result = np.zeros(shape or (), dtype=np.complex128)
    external_ranges = tuple(
        _index_values(index, n_occupied, n_virtual)
        for index in external_indices
    )
    for external_local in itertools.product(*(range(size) for size in shape)):
        assignment = {}
        for index, local, values in zip(
            external_indices,
            external_local,
            external_ranges,
            strict=True,
        ):
            assignment[index] = values[local]

        total = 0.0j
        for term in expression:
            assert not term.operators
            dummy_ranges = tuple(
                _index_values(index, n_occupied, n_virtual)
                for index in term.dummies
            )
            for dummy_values in itertools.product(*dummy_ranges):
                term_assignment = assignment | dict(
                    zip(term.dummies, dummy_values, strict=True)
                )
                value = complex(term.coefficient.real) + 1j * complex(
                    term.coefficient.imag
                )
                for factor in term.factors:
                    if isinstance(factor, Tensor):
                        value *= _tensor_value(
                            factor,
                            term_assignment,
                            arrays,
                            n_occupied,
                        )
                    elif isinstance(factor, Delta):
                        value *= int(
                            term_assignment[factor.left]
                            == term_assignment[factor.right]
                        )
                    elif isinstance(factor, OrbitalDimension):
                        value *= len(
                            _index_values(factor, n_occupied, n_virtual)
                        )
                    else:
                        raise TypeError(type(factor).__name__)
                total += value
        result[external_local] = total
    return result[()] if not shape else result


def _spin_fock(fock: np.ndarray) -> np.ndarray:
    n_orbitals = fock.shape[0]
    result = np.zeros((2 * n_orbitals, 2 * n_orbitals), dtype=fock.dtype)
    for p, q, spin in itertools.product(
        range(n_orbitals),
        range(n_orbitals),
        range(2),
    ):
        result[2 * p + spin, 2 * q + spin] = fock[p, q]
    return result


def _spin_interaction(two_body: np.ndarray) -> np.ndarray:
    n_orbitals = two_body.shape[0]
    interaction = np.zeros((2 * n_orbitals,) * 4, dtype=two_body.dtype)
    for p, q, r, s in itertools.product(range(n_orbitals), repeat=4):
        for sigma, tau in itertools.product(range(2), repeat=2):
            interaction[
                2 * p + sigma,
                2 * q + tau,
                2 * r + sigma,
                2 * s + tau,
            ] = two_body[p, q, r, s]
    return interaction - interaction.swapaxes(2, 3)


def _spin_singles(singles: np.ndarray) -> np.ndarray:
    n_virtual, n_occupied = singles.shape
    result = np.zeros((2 * n_virtual, 2 * n_occupied), dtype=singles.dtype)
    for a, i, spin in itertools.product(
        range(n_virtual),
        range(n_occupied),
        range(2),
    ):
        result[2 * a + spin, 2 * i + spin] = singles[a, i]
    return result


def _spin_doubles(doubles: np.ndarray) -> np.ndarray:
    n_virtual, _, n_occupied, _ = doubles.shape
    result = np.zeros(
        (2 * n_virtual, 2 * n_virtual, 2 * n_occupied, 2 * n_occupied),
        dtype=doubles.dtype,
    )
    for a_spin, b_spin, i_spin, j_spin in itertools.product(
        range(2 * n_virtual),
        range(2 * n_virtual),
        range(2 * n_occupied),
        range(2 * n_occupied),
    ):
        a, sigma_a = divmod(a_spin, 2)
        b, sigma_b = divmod(b_spin, 2)
        i, sigma_i = divmod(i_spin, 2)
        j, sigma_j = divmod(j_spin, 2)
        if sigma_a == sigma_i and sigma_b == sigma_j:
            result[a_spin, b_spin, i_spin, j_spin] += doubles[a, b, i, j]
        if sigma_a == sigma_j and sigma_b == sigma_i:
            result[a_spin, b_spin, i_spin, j_spin] -= doubles[a, b, j, i]
    return result


@pytest.mark.parametrize("method", ("cis", "cid", "cisd"))
def test_generated_sigma_matches_complex_closed_and_spin_kernels(method: str) -> None:
    rng = np.random.default_rng(3100 + len(method))
    n_occupied = 2
    n_virtual = 2
    n_orbitals = n_occupied + n_virtual
    def random(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    e_ref = np.complex128(0.4 - 0.3j)
    c0 = np.complex128(-0.2 + 0.7j)
    fock = random((n_orbitals, n_orbitals))
    raw_u = random((n_orbitals,) * 4)
    two_body = 0.5 * (raw_u + raw_u.transpose(1, 0, 3, 2))
    singles = random((n_virtual, n_occupied))
    raw_doubles = random((n_virtual, n_virtual, n_occupied, n_occupied))
    doubles = 0.5 * (raw_doubles + raw_doubles.transpose(1, 0, 3, 2))

    arrays = {
        REFERENCE_ENERGY: e_ref,
        REFERENCE_COEFFICIENT: c0,
        FOCK: fock,
        TWO_ELECTRON: two_body,
        SINGLES_AMPLITUDE: singles,
        DOUBLES_AMPLITUDE: doubles,
    }
    sigma = derive_right_sigma(method)
    generated = [_evaluate_component(sigma[0], (), arrays, 2, 2)]
    if 1 in sigma.ranks:
        generated.append(
            _evaluate_component(
                sigma[1],
                (virtual("a"), occupied("i")),
                arrays,
                n_occupied,
                n_virtual,
            )
        )
    if 2 in sigma.ranks:
        generated.append(
            _evaluate_component(
                sigma[2],
                (virtual("a"), virtual("b"), occupied("i"), occupied("j")),
                arrays,
                n_occupied,
                n_virtual,
            )
        )

    occupied_slice = slice(0, n_occupied)
    virtual_slice = slice(n_occupied, n_orbitals)
    if method == "cis":
        closed = closed_cis_sigma(
            e_ref, fock, two_body, c0, singles, occupied_slice, virtual_slice
        )
    elif method == "cid":
        closed = closed_cid_sigma(
            e_ref, fock, two_body, c0, doubles, occupied_slice, virtual_slice
        )
    else:
        closed = closed_cisd_sigma(
            e_ref,
            fock,
            two_body,
            c0,
            singles,
            doubles,
            occupied_slice,
            virtual_slice,
        )
    for actual, expected in zip(generated, closed, strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-10)

    spin_fock = _spin_fock(fock)
    spin_two_body = _spin_interaction(two_body)
    spin_singles = _spin_singles(singles)
    spin_doubles = _spin_doubles(doubles)
    occupied_spin = slice(0, 2 * n_occupied)
    virtual_spin = slice(2 * n_occupied, 2 * n_orbitals)
    if method == "cis":
        spin = spin_cis_sigma(
            e_ref,
            spin_fock,
            spin_two_body,
            c0,
            spin_singles,
            occupied_spin,
            virtual_spin,
        )
        expected_spin = (generated[0], _spin_singles(generated[1]))
    elif method == "cid":
        spin = spin_cid_sigma(
            e_ref,
            spin_fock,
            spin_two_body,
            c0,
            spin_doubles,
            occupied_spin,
            virtual_spin,
        )
        expected_spin = (generated[0], _spin_doubles(generated[1]))
    else:
        spin = spin_cisd_sigma(
            e_ref,
            spin_fock,
            spin_two_body,
            c0,
            spin_singles,
            spin_doubles,
            occupied_spin,
            virtual_spin,
        )
        expected_spin = (
            generated[0],
            _spin_singles(generated[1]),
            _spin_doubles(generated[2]),
        )
    for actual, expected in zip(spin, expected_spin, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-10)
