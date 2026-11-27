"""Numerical invariant tests for closed-shell Matthews CISDT."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from manybody_lib.configuration_interaction.closed_shell.cisd import (
    compute_norm as cisd_norm,
)
from manybody_lib.configuration_interaction.closed_shell.cisd import (
    compute_one_rdm as cisd_one_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cisd import (
    compute_sigma as cisd_sigma,
)
from manybody_lib.configuration_interaction.closed_shell.cisd import (
    compute_two_rdm as cisd_two_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt import (
    compute_hermitian_left,
    compute_norm,
    compute_one_rdm,
    compute_sigma,
    compute_two_rdm,
    project_physical_triples,
    symmetrize_triples_columns,
    triples_null_component,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt._contractions import (
    SIGMA_SPECS,
    TERM_COUNTS,
)


_S3 = tuple(itertools.permutations(range(3)))


def test_generated_tables_preserve_symbolic_counts_and_permutation_factoring() -> None:
    assert TERM_COUNTS["sigma"] == (4, 16, 37, 294)
    stored_counts = tuple(
        len(SIGMA_SPECS[rank]["explicit"])
        + sum(len(group[1]) for group in SIGMA_SPECS[rank]["groups"])
        for rank in range(4)
    )
    assert stored_counts == (4, 16, 21, 59)


def _complex_random(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _column_symmetric_doubles(raw: np.ndarray) -> np.ndarray:
    return 0.5 * (raw + raw.transpose(1, 0, 3, 2))


def _amplitudes(
    rng: np.random.Generator,
    nocc: int,
    nvir: int,
) -> tuple[np.complex128, np.ndarray, np.ndarray, np.ndarray]:
    c0 = np.complex128(0.4 - 0.3j)
    c1 = _complex_random(rng, (nvir, nocc))
    c2 = _column_symmetric_doubles(
        _complex_random(rng, (nvir, nvir, nocc, nocc))
    )
    raw_c3 = _complex_random(
        rng,
        (nvir, nvir, nvir, nocc, nocc, nocc),
    )
    c3 = project_physical_triples(symmetrize_triples_columns(raw_c3))
    return c0, c1, c2, c3


def _hermitian_spatial_system(
    rng: np.random.Generator,
    nocc: int,
    nvir: int,
) -> tuple[complex, np.ndarray, np.ndarray, np.ndarray, slice, slice]:
    norb = nocc + nvir
    occupied = slice(0, nocc)
    virtual = slice(nocc, norb)
    raw_h = _complex_random(rng, (norb, norb))
    h = 0.5 * (raw_h + raw_h.T.conj())

    raw_u = _complex_random(rng, (norb, norb, norb, norb))
    u = 0.25 * (
        raw_u
        + raw_u.transpose(1, 0, 3, 2)
        + raw_u.transpose(2, 3, 0, 1).conj()
        + raw_u.transpose(3, 2, 1, 0).conj()
    )
    f = h.copy()
    f += 2.0 * np.einsum(
        "piqi->pq",
        u[:, occupied, :, occupied],
        optimize=True,
    )
    f -= np.einsum(
        "piiq->pq",
        u[:, occupied, occupied, :],
        optimize=True,
    )
    e_ref = 2.0 * np.trace(h[occupied, occupied])
    e_ref += 2.0 * np.einsum(
        "ijij->",
        u[occupied, occupied, occupied, occupied],
        optimize=True,
    )
    e_ref -= np.einsum(
        "ijji->",
        u[occupied, occupied, occupied, occupied],
        optimize=True,
    )
    return e_ref, f, u, h, occupied, virtual


def _triples_metric_action(c3: np.ndarray) -> np.ndarray:
    result = np.zeros_like(c3)
    for permutation in _S3:
        cycle_count = _cycle_count(permutation)
        coefficient = {3: 8.0, 2: -4.0, 1: 2.0}[cycle_count]
        axes = (0, 1, 2) + tuple(3 + item for item in permutation)
        result += coefficient * c3.transpose(axes)
    return result


def _cycle_count(permutation: tuple[int, ...]) -> int:
    seen: set[int] = set()
    count = 0
    for start in range(len(permutation)):
        if start in seen:
            continue
        count += 1
        position = start
        while position not in seen:
            seen.add(position)
            position = permutation[position]
    return count


def test_cisdt_rdms_obey_closed_shell_invariants() -> None:
    rng = np.random.default_rng(1601)
    nocc, nvir = 2, 3
    c0, c1, c2, c3 = _amplitudes(rng, nocc, nvir)

    norm = compute_norm(c0, c1, c2, c3)
    gamma = compute_one_rdm(c0, c1, c2, c3)
    gamma2 = compute_two_rdm(c0, c1, c2, c3)

    np.testing.assert_allclose(gamma, gamma.T.conj(), atol=2e-10)
    np.testing.assert_allclose(
        gamma2,
        gamma2.transpose(2, 3, 0, 1).conj(),
        atol=2e-10,
    )
    np.testing.assert_allclose(
        gamma2,
        gamma2.transpose(1, 0, 3, 2),
        atol=2e-10,
    )
    np.testing.assert_allclose(np.trace(gamma), 2 * nocc * norm, atol=2e-10)
    np.testing.assert_allclose(
        np.einsum("pqrq->pr", gamma2, optimize=True),
        (2 * nocc - 1) * gamma,
        atol=3e-10,
    )


def test_cisdt_sigma_and_rdms_have_consistent_energy() -> None:
    rng = np.random.default_rng(1613)
    nocc, nvir = 2, 3
    e_ref, f, u, h, occupied, virtual = _hermitian_spatial_system(
        rng,
        nocc,
        nvir,
    )
    c0, c1, c2, c3 = _amplitudes(rng, nocc, nvir)
    left0, left1, left2, left3 = compute_hermitian_left(c0, c1, c2, c3)
    sigma0, sigma1, sigma2, sigma3 = compute_sigma(
        e_ref,
        f,
        u,
        c0,
        c1,
        c2,
        c3,
        occupied,
        virtual,
    )

    covariant1 = 2.0 * sigma1
    covariant2 = 4.0 * sigma2 - 2.0 * sigma2.swapaxes(2, 3)
    covariant3 = _triples_metric_action(sigma3)
    energy_sigma = left0 * sigma0
    energy_sigma += np.einsum("ia,ai->", left1, covariant1, optimize=True)
    energy_sigma += 0.5 * np.einsum(
        "ijab,abij->",
        left2,
        covariant2,
        optimize=True,
    )
    energy_sigma += (1.0 / 6.0) * np.einsum(
        "ijkabc,abcijk->",
        left3,
        covariant3,
        optimize=True,
    )

    gamma = compute_one_rdm(c0, c1, c2, c3, left0, left1, left2, left3)
    gamma2 = compute_two_rdm(c0, c1, c2, c3, left0, left1, left2, left3)
    energy_rdm = np.einsum("pq,pq->", h, gamma, optimize=True)
    energy_rdm += 0.5 * np.einsum(
        "pqrs,pqrs->",
        u,
        gamma2,
        optimize=True,
    )
    np.testing.assert_allclose(energy_sigma, energy_rdm, atol=2e-9)


def test_cisdt_reduces_to_existing_cisd_when_triples_are_zero() -> None:
    rng = np.random.default_rng(1621)
    nocc, nvir = 2, 3
    e_ref, f, u, _h, occupied, virtual = _hermitian_spatial_system(
        rng,
        nocc,
        nvir,
    )
    c0, c1, c2, c3 = _amplitudes(rng, nocc, nvir)
    c3.fill(0)

    actual = compute_sigma(
        e_ref,
        f,
        u,
        c0,
        c1,
        c2,
        c3,
        occupied,
        virtual,
    )
    expected = cisd_sigma(e_ref, f, u, c0, c1, c2, occupied, virtual)
    for actual_component, expected_component in zip(
        actual[:3],
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(actual_component, expected_component, atol=2e-10)
    np.testing.assert_allclose(
        actual[3],
        symmetrize_triples_columns(actual[3]),
        atol=2e-10,
    )
    np.testing.assert_allclose(
        triples_null_component(actual[3]),
        0.0,
        atol=2e-10,
    )
    np.testing.assert_allclose(compute_norm(c0, c1, c2, c3), cisd_norm(c0, c1, c2))
    np.testing.assert_allclose(
        compute_one_rdm(c0, c1, c2, c3),
        cisd_one_rdm(c0, c1, c2),
        atol=2e-10,
    )
    np.testing.assert_allclose(
        compute_two_rdm(c0, c1, c2, c3),
        cisd_two_rdm(c0, c1, c2),
        atol=2e-10,
    )


def test_cisdt_is_invariant_to_triples_null_gauge() -> None:
    rng = np.random.default_rng(1637)
    nocc, nvir = 2, 3
    e_ref, f, u, _h, occupied, virtual = _hermitian_spatial_system(
        rng,
        nocc,
        nvir,
    )
    c0, c1, c2, c3 = _amplitudes(rng, nocc, nvir)
    raw_shift = symmetrize_triples_columns(
        _complex_random(rng, c3.shape)
    )
    shifted_c3 = c3 + triples_null_component(raw_shift)

    sigma = compute_sigma(
        e_ref, f, u, c0, c1, c2, c3, occupied, virtual
    )
    shifted_sigma = compute_sigma(
        e_ref, f, u, c0, c1, c2, shifted_c3, occupied, virtual
    )
    for actual, shifted in zip(sigma, shifted_sigma, strict=True):
        np.testing.assert_allclose(actual, shifted, atol=5e-10)
    np.testing.assert_allclose(triples_null_component(sigma[3]), 0.0, atol=2e-10)
    np.testing.assert_allclose(
        compute_one_rdm(c0, c1, c2, c3),
        compute_one_rdm(c0, c1, c2, shifted_c3),
        atol=5e-10,
    )
    np.testing.assert_allclose(
        compute_two_rdm(c0, c1, c2, c3),
        compute_two_rdm(c0, c1, c2, shifted_c3),
        atol=5e-10,
    )


def test_cisdt_supports_independent_left_state_and_complex_linearity() -> None:
    rng = np.random.default_rng(1657)
    nocc, nvir = 2, 3
    e_ref, f, u, h, occupied, virtual = _hermitian_spatial_system(
        rng,
        nocc,
        nvir,
    )
    c0, c1, c2, c3 = _amplitudes(rng, nocc, nvir)
    alpha = np.complex128(-0.7 + 0.2j)
    sigma = compute_sigma(
        e_ref, f, u, c0, c1, c2, c3, occupied, virtual
    )
    scaled = compute_sigma(
        e_ref,
        f,
        u,
        alpha * c0,
        alpha * c1,
        alpha * c2,
        alpha * c3,
        occupied,
        virtual,
    )
    for actual, scaled_component in zip(sigma, scaled, strict=True):
        np.testing.assert_allclose(scaled_component, alpha * actual, atol=5e-10)

    left0 = np.complex128(-0.2 + 0.6j)
    left1 = _complex_random(rng, (nocc, nvir))
    raw_left2 = _complex_random(rng, (nocc, nocc, nvir, nvir))
    left2 = _column_symmetric_doubles(raw_left2.transpose(2, 3, 0, 1)).transpose(
        2,
        3,
        0,
        1,
    )
    raw_left3 = _complex_random(
        rng,
        (nocc, nocc, nocc, nvir, nvir, nvir),
    )
    left3 = symmetrize_triples_columns(
        raw_left3.transpose(3, 4, 5, 0, 1, 2)
    ).transpose(3, 4, 5, 0, 1, 2)

    norm = compute_norm(c0, c1, c2, c3, left0, left1, left2, left3)
    gamma = compute_one_rdm(c0, c1, c2, c3, left0, left1, left2, left3)
    gamma2 = compute_two_rdm(c0, c1, c2, c3, left0, left1, left2, left3)
    np.testing.assert_allclose(np.trace(gamma), 2 * nocc * norm, atol=3e-10)
    np.testing.assert_allclose(
        np.einsum("pqrq->pr", gamma2, optimize=True),
        (2 * nocc - 1) * gamma,
        atol=5e-10,
    )

    sigma0, sigma1, sigma2, sigma3 = sigma
    energy_sigma = left0 * sigma0
    energy_sigma += np.einsum(
        "ia,ai->",
        left1,
        2.0 * sigma1,
        optimize=True,
    )
    energy_sigma += 0.5 * np.einsum(
        "ijab,abij->",
        left2,
        4.0 * sigma2 - 2.0 * sigma2.swapaxes(2, 3),
        optimize=True,
    )
    energy_sigma += (1.0 / 6.0) * np.einsum(
        "ijkabc,abcijk->",
        left3,
        _triples_metric_action(sigma3),
        optimize=True,
    )
    energy_rdm = np.einsum("pq,pq->", h, gamma, optimize=True)
    energy_rdm += 0.5 * np.einsum(
        "pqrs,pqrs->",
        u,
        gamma2,
        optimize=True,
    )
    np.testing.assert_allclose(energy_sigma, energy_rdm, atol=2e-9)

    with pytest.raises(ValueError, match="supplied together"):
        compute_one_rdm(c0, c1, c2, c3, left_c0=left0)
