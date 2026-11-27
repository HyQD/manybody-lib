"""Norm and RDM generation tests for closed-shell symbolic Milestone 4."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from manybody_lib.symbolic.closed_shell import (
    Delta,
    Expression,
    FOCK,
    OrbitalDimension,
    REFERENCE_ENERGY,
    TWO_ELECTRON,
    Tensor,
    conjugate_scalar_expression,
    derive_right_sigma,
    occupied,
    virtual,
)
from manybody_lib.symbolic.closed_shell.equations.rdm import (
    derive_ci_rdms,
    derive_one_rdm_block,
    derive_two_rdm_block,
    normalize_by_norm,
    rdms_to_latex,
)
from manybody_lib.symbolic.closed_shell.manifolds.hjo import (
    DOUBLES_AMPLITUDE,
    LEFT_DOUBLES_AMPLITUDE,
    LEFT_REFERENCE_COEFFICIENT,
    LEFT_SINGLES_AMPLITUDE,
    REFERENCE_COEFFICIENT,
    SINGLES_AMPLITUDE,
    build_hjo_left_state,
    hermitian_bra_substitution,
)
from tests.symbolic.closed_shell._fixture_parser import load_fixture_section


def test_left_state_is_independent_until_explicit_hermitian_substitution() -> None:
    left = build_hjo_left_state("cisd").expression
    left_symbols = {
        factor.symbol
        for term in left
        for factor in term.factors
        if isinstance(factor, Tensor)
    }
    assert left_symbols == {
        LEFT_REFERENCE_COEFFICIENT,
        LEFT_SINGLES_AMPLITUDE,
        LEFT_DOUBLES_AMPLITUDE,
    }

    hermitian = hermitian_bra_substitution(left)
    hermitian_tensors = [
        factor
        for term in hermitian
        for factor in term.factors
        if isinstance(factor, Tensor)
    ]
    assert all(
        tensor.symbol
        in {REFERENCE_COEFFICIENT, SINGLES_AMPLITUDE, DOUBLES_AMPLITUDE}
        for tensor in hermitian_tensors
    )
    assert all(tensor.conjugated for tensor in hermitian_tensors)


@pytest.mark.parametrize("method", ("cis", "cid", "cisd"))
def test_norm_and_all_rdm_blocks_match_expanded_drudge_fixtures(method: str) -> None:
    generated = derive_ci_rdms(method)
    fixture_norm = load_fixture_section(method, "norm")
    fixture_one = load_fixture_section(method, "one_rdm")
    fixture_two = load_fixture_section(method, "two_rdm")

    assert generated.norm == next(iter(fixture_norm.values()))
    assert {block.label for block in generated.one_body} == set(fixture_one)
    assert {block.label for block in generated.two_body} == set(fixture_two)
    for block in generated.one_body:
        assert block.expression == fixture_one[block.label]
    for block in generated.two_body:
        assert block.expression == fixture_two[block.label]


def test_normalization_is_an_explicit_api_boundary_operation() -> None:
    rdms = derive_ci_rdms("cis")
    block = rdms.block(1, r"\gamma^{j}_{i}")
    normalized = normalize_by_norm(block.expression, rdms.norm)

    assert normalized.numerator is block.expression
    assert normalized.denominator is rdms.norm
    assert normalized.to_latex().startswith(r"\frac{")


@pytest.mark.parametrize("method", ("cis", "cid", "cisd"))
def test_hermiticity_after_explicit_hermitian_bra_substitution(method: str) -> None:
    rdms = derive_ci_rdms(method)
    hermitian_norm = hermitian_bra_substitution(rdms.norm)
    assert hermitian_norm == conjugate_scalar_expression(hermitian_norm)

    index_pairs = (
        (occupied("i"), occupied("j")),
        (occupied("i"), virtual("a")),
        (virtual("a"), occupied("i")),
        (virtual("a"), virtual("b")),
    )
    for p, q in index_pairs:
        direct = hermitian_bra_substitution(
            derive_one_rdm_block(method, p, q)
        )
        adjoint = hermitian_bra_substitution(
            derive_one_rdm_block(method, q, p)
        )
        assert direct == conjugate_scalar_expression(adjoint)

    for block in rdms.two_body:
        p, q, r, s = block.indices
        direct = hermitian_bra_substitution(block.expression)
        adjoint = hermitian_bra_substitution(
            derive_two_rdm_block(method, r, s, p, q)
        )
        assert direct == conjugate_scalar_expression(adjoint)


def test_rdm_latex_contains_norm_and_all_cisd_blocks() -> None:
    rendered = rdms_to_latex(derive_ci_rdms("cisd"))
    assert rendered.startswith(r"\begin{align}")
    assert rendered.endswith(r"\end{align}")
    assert r"\langle\widetilde{\Psi}\vert\Psi\rangle" in rendered
    assert r"\gamma^{j}_{i}" in rendered
    assert r"\Gamma^{cd}_{ab}" in rendered
    assert r"\mathcal{P}^{kl}_{ij}" in rendered


_AMPLITUDE_SYMBOLS = {
    SINGLES_AMPLITUDE,
    DOUBLES_AMPLITUDE,
    LEFT_SINGLES_AMPLITUDE,
    LEFT_DOUBLES_AMPLITUDE,
}


def _index_values(index, n_occupied: int, n_virtual: int) -> range:
    if index.space.value == "occupied":
        return range(n_occupied)
    if index.space.value == "virtual":
        return range(n_occupied, n_occupied + n_virtual)
    raise ValueError(f"unsupported numerical index space {index.space}")


def _tensor_value(tensor: Tensor, assignment: dict, arrays: dict, n_occupied: int):
    value = arrays[tensor.symbol]
    if tensor.indices:
        indices = tuple(assignment[index] for index in tensor.indices)
        if tensor.symbol in _AMPLITUDE_SYMBOLS:
            indices = tuple(
                index - n_occupied if orbital.space.value == "virtual" else index
                for orbital, index in zip(tensor.indices, indices, strict=True)
            )
        value = value[indices]
    return value.conjugate() if tensor.conjugated else value


def _evaluate(
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
        assignment = {
            index: values[local]
            for index, local, values in zip(
                external_indices,
                external_local,
                external_ranges,
                strict=True,
            )
        }
        total = 0.0j
        for term in expression:
            assert not term.operators
            dummy_ranges = tuple(
                _index_values(index, n_occupied, n_virtual)
                for index in term.dummies
            )
            for values in itertools.product(*dummy_ranges):
                term_assignment = assignment | dict(
                    zip(term.dummies, values, strict=True)
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
                        if factor.space.value == "occupied":
                            value *= n_occupied
                        elif factor.space.value == "virtual":
                            value *= n_virtual
                        else:
                            value *= n_occupied + n_virtual
                    else:
                        raise TypeError(type(factor).__name__)
                total += value
        result[external_local] = total
    return result[()] if not shape else result


def _full_rdms(method: str, arrays: dict, n_occupied: int, n_virtual: int):
    n_orbitals = n_occupied + n_virtual
    gamma = np.zeros((n_orbitals,) * 2, dtype=np.complex128)
    gamma2 = np.zeros((n_orbitals,) * 4, dtype=np.complex128)
    spaces = ("o", "v")
    factories = {"o": occupied, "v": virtual}

    for p_space, q_space in itertools.product(spaces, repeat=2):
        p = factories[p_space]("i" if p_space == "o" else "a")
        q = factories[q_space]("j" if q_space == "o" else "b")
        p_values = _index_values(p, n_occupied, n_virtual)
        q_values = _index_values(q, n_occupied, n_virtual)
        gamma[np.ix_(p_values, q_values)] = _evaluate(
            derive_one_rdm_block(method, p, q),
            (p, q),
            arrays,
            n_occupied,
            n_virtual,
        )

    occupied_names = "ijkl"
    virtual_names = "abcd"
    for block_spaces in itertools.product(spaces, repeat=4):
        indices = tuple(
            factories[space](
                occupied_names[position]
                if space == "o"
                else virtual_names[position]
            )
            for position, space in enumerate(block_spaces)
        )
        ranges = tuple(
            _index_values(index, n_occupied, n_virtual)
            for index in indices
        )
        gamma2[np.ix_(*ranges)] = _evaluate(
            derive_two_rdm_block(method, *indices),
            indices,
            arrays,
            n_occupied,
            n_virtual,
        )
    return gamma, gamma2


@pytest.mark.parametrize("method", ("cis", "cid", "cisd"))
def test_particle_number_contraction_and_rdm_sigma_energy(method: str) -> None:
    rng = np.random.default_rng(4200 + len(method))
    n_occupied = 2
    n_virtual = 2
    n_orbitals = n_occupied + n_virtual

    def random(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    h = random((n_orbitals, n_orbitals))
    raw_u = random((n_orbitals,) * 4)
    u = 0.5 * (raw_u + raw_u.transpose(1, 0, 3, 2))
    o = slice(0, n_occupied)
    f = h.copy()
    f += 2 * np.einsum("piqi->pq", u[:, o, :, o], optimize=True)
    f -= np.einsum("piiq->pq", u[:, o, o, :], optimize=True)
    e_ref = 2 * np.trace(h[o, o])
    e_ref += 2 * np.einsum("ijij->", u[o, o, o, o], optimize=True)
    e_ref -= np.einsum("ijji->", u[o, o, o, o], optimize=True)

    c0 = np.complex128(0.3 - 0.4j)
    c0_left = np.complex128(-0.2 + 0.6j)
    c1 = random((n_virtual, n_occupied))
    c1_left = random((n_occupied, n_virtual))
    raw_c2 = random((n_virtual, n_virtual, n_occupied, n_occupied))
    c2 = 0.5 * (raw_c2 + raw_c2.transpose(1, 0, 3, 2))
    raw_l2 = random((n_occupied, n_occupied, n_virtual, n_virtual))
    c2_left = 0.5 * (raw_l2 + raw_l2.transpose(1, 0, 3, 2))
    arrays = {
        REFERENCE_ENERGY: e_ref,
        REFERENCE_COEFFICIENT: c0,
        LEFT_REFERENCE_COEFFICIENT: c0_left,
        FOCK: f,
        TWO_ELECTRON: u,
        SINGLES_AMPLITUDE: c1,
        LEFT_SINGLES_AMPLITUDE: c1_left,
        DOUBLES_AMPLITUDE: c2,
        LEFT_DOUBLES_AMPLITUDE: c2_left,
    }

    rdms = derive_ci_rdms(method)
    norm = _evaluate(rdms.norm, (), arrays, n_occupied, n_virtual)
    gamma, gamma2 = _full_rdms(method, arrays, n_occupied, n_virtual)
    electron_count = 2 * n_occupied
    np.testing.assert_allclose(np.trace(gamma), electron_count * norm, atol=2e-9)
    np.testing.assert_allclose(
        np.einsum("pqrq->pr", gamma2, optimize=True),
        (electron_count - 1) * gamma,
        atol=3e-9,
    )

    rdm_energy = np.einsum("pq,pq->", h, gamma, optimize=True)
    rdm_energy += 0.5 * np.einsum("pqrs,pqrs->", u, gamma2, optimize=True)
    sigma = derive_right_sigma(method)
    sigma0 = _evaluate(sigma[0], (), arrays, n_occupied, n_virtual)
    sigma_energy = c0_left * sigma0
    if 1 in sigma.ranks:
        sigma1 = _evaluate(
            sigma[1],
            (virtual("a"), occupied("i")),
            arrays,
            n_occupied,
            n_virtual,
        )
        sigma_energy += np.einsum("ia,ai->", c1_left, sigma1)
    if 2 in sigma.ranks:
        sigma2 = _evaluate(
            sigma[2],
            (virtual("a"), virtual("b"), occupied("i"), occupied("j")),
            arrays,
            n_occupied,
            n_virtual,
        )
        sigma_energy += 0.5 * np.einsum(
            "ijab,abij->",
            c2_left,
            sigma2,
        )
    np.testing.assert_allclose(rdm_energy, sigma_energy, atol=4e-9)
    np.testing.assert_allclose(rdm_energy / norm, sigma_energy / norm, atol=4e-9)
