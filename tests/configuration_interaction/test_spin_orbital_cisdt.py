"""Numerical invariant tests for spin-orbital CISDT."""

from __future__ import annotations

import numpy as np

from manybody_lib.configuration_interaction.spin_orbital.cisd import (
    compute_norm as cisd_norm,
)
from manybody_lib.configuration_interaction.spin_orbital.cisd import (
    compute_one_rdm as cisd_one_rdm,
)
from manybody_lib.configuration_interaction.spin_orbital.cisd import (
    compute_sigma as cisd_sigma,
)
from manybody_lib.configuration_interaction.spin_orbital.cisd import (
    compute_two_rdm as cisd_two_rdm,
)
from manybody_lib.configuration_interaction.spin_orbital.cisdt import (
    antisymmetrize_doubles,
    antisymmetrize_triples,
    compute_norm,
    compute_one_rdm,
    compute_sigma,
    compute_two_rdm,
)
from manybody_lib.configuration_interaction.spin_orbital.cisdt._contractions import (
    TERM_COUNTS,
)


def _complex_random(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _amplitudes(
    rng: np.random.Generator,
    nocc: int,
    nvir: int,
) -> tuple[np.complex128, np.ndarray, np.ndarray, np.ndarray]:
    c0 = np.complex128(0.5 - 0.2j)
    c1 = _complex_random(rng, (nvir, nocc))
    c2 = antisymmetrize_doubles(
        _complex_random(rng, (nvir, nvir, nocc, nocc))
    )
    c3 = antisymmetrize_triples(
        _complex_random(
            rng,
            (nvir, nvir, nvir, nocc, nocc, nocc),
        )
    )
    return c0, c1, c2, c3


def _hermitian_system(
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
    u = antisymmetrize_doubles(raw_u)
    u = 0.5 * (u + u.transpose(2, 3, 0, 1).conj())
    f = h + np.einsum(
        "piqi->pq",
        u[:, occupied, :, occupied],
        optimize=True,
    )
    e_ref = np.trace(h[occupied, occupied])
    e_ref += 0.5 * np.einsum(
        "ijij->",
        u[occupied, occupied, occupied, occupied],
        optimize=True,
    )
    return e_ref, f, u, h, occupied, virtual


def test_generated_contraction_counts_match_symbolic_expressions() -> None:
    assert TERM_COUNTS == {
        "sigma": (3, 9, 25, 58),
        "norm": 4,
        "one_rdm": (7, 3, 3, 3),
        "two_rdm": (22, 8, 2, 8, 6, 2, 2, 2, 2),
    }


def test_spin_orbital_cisdt_rdms_obey_invariants() -> None:
    rng = np.random.default_rng(1709)
    nocc, nvir = 3, 4
    c0, c1, c2, c3 = _amplitudes(rng, nocc, nvir)
    norm = compute_norm(c0, c1, c2, c3)
    gamma = compute_one_rdm(c0, c1, c2, c3)
    gamma2 = compute_two_rdm(c0, c1, c2, c3)

    np.testing.assert_allclose(gamma, gamma.T.conj(), atol=3e-10)
    np.testing.assert_allclose(
        gamma2,
        gamma2.transpose(2, 3, 0, 1).conj(),
        atol=3e-10,
    )
    np.testing.assert_allclose(gamma2, -gamma2.swapaxes(0, 1), atol=3e-10)
    np.testing.assert_allclose(gamma2, -gamma2.swapaxes(2, 3), atol=3e-10)
    np.testing.assert_allclose(np.trace(gamma), nocc * norm, atol=3e-10)
    np.testing.assert_allclose(
        np.einsum("pqrq->pr", gamma2, optimize=True),
        (nocc - 1) * gamma,
        atol=5e-10,
    )


def test_spin_orbital_cisdt_sigma_and_rdms_have_consistent_energy() -> None:
    rng = np.random.default_rng(1721)
    nocc, nvir = 3, 4
    e_ref, f, u, h, occupied, virtual = _hermitian_system(rng, nocc, nvir)
    c0, c1, c2, c3 = _amplitudes(rng, nocc, nvir)
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
    energy_sigma = np.conjugate(c0) * sigma0
    energy_sigma += np.einsum("ai,ai->", c1.conj(), sigma1, optimize=True)
    energy_sigma += 0.25 * np.einsum(
        "abij,abij->",
        c2.conj(),
        sigma2,
        optimize=True,
    )
    energy_sigma += (1.0 / 36.0) * np.einsum(
        "abcijk,abcijk->",
        c3.conj(),
        sigma3,
        optimize=True,
    )

    gamma = compute_one_rdm(c0, c1, c2, c3)
    gamma2 = compute_two_rdm(c0, c1, c2, c3)
    energy_rdm = np.einsum("pq,pq->", h, gamma, optimize=True)
    energy_rdm += 0.25 * np.einsum(
        "pqrs,pqrs->",
        u,
        gamma2,
        optimize=True,
    )
    np.testing.assert_allclose(energy_sigma, energy_rdm, atol=2e-9)


def test_spin_orbital_cisdt_reduces_to_cisd_for_shared_outputs() -> None:
    rng = np.random.default_rng(1741)
    nocc, nvir = 3, 4
    e_ref, f, u, _h, occupied, virtual = _hermitian_system(
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
        np.testing.assert_allclose(actual_component, expected_component, atol=3e-10)
    assert np.linalg.norm(actual[3]) > 1e-8
    np.testing.assert_allclose(compute_norm(c0, c1, c2, c3), cisd_norm(c0, c1, c2))
    np.testing.assert_allclose(
        compute_one_rdm(c0, c1, c2, c3),
        cisd_one_rdm(c0, c1, c2),
        atol=3e-10,
    )
    np.testing.assert_allclose(
        compute_two_rdm(c0, c1, c2, c3),
        cisd_two_rdm(c0, c1, c2),
        atol=3e-10,
    )


def test_spin_orbital_cisdt_sigma_is_complex_linear_and_antisymmetric() -> None:
    rng = np.random.default_rng(1753)
    nocc, nvir = 3, 4
    e_ref, f, u, _h, occupied, virtual = _hermitian_system(
        rng,
        nocc,
        nvir,
    )
    c0, c1, c2, c3 = _amplitudes(rng, nocc, nvir)
    alpha = np.complex128(-0.4 + 0.7j)
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

    sigma2, sigma3 = sigma[2], sigma[3]
    np.testing.assert_allclose(sigma2, -sigma2.swapaxes(0, 1), atol=3e-10)
    np.testing.assert_allclose(sigma2, -sigma2.swapaxes(2, 3), atol=3e-10)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        np.testing.assert_allclose(
            sigma3,
            -sigma3.swapaxes(first, second),
            atol=5e-10,
        )
        np.testing.assert_allclose(
            sigma3,
            -sigma3.swapaxes(3 + first, 3 + second),
            atol=5e-10,
        )
