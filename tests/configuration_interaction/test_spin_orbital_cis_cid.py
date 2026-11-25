"""Invariant and energy-closure tests for spin-orbital CIS and CID."""

from __future__ import annotations

import numpy as np
import pytest

from manybody_lib.configuration_interaction.spin_orbital.cis import (
    compute_norm as cis_norm,
)
from manybody_lib.configuration_interaction.spin_orbital.cis import (
    compute_one_rdm as cis_one_rdm,
)
from manybody_lib.configuration_interaction.spin_orbital.cis import (
    compute_sigma as cis_sigma,
)
from manybody_lib.configuration_interaction.spin_orbital.cis import (
    compute_two_rdm as cis_two_rdm,
)
from manybody_lib.configuration_interaction.spin_orbital.cid import (
    compute_norm as cid_norm,
)
from manybody_lib.configuration_interaction.spin_orbital.cid import (
    compute_one_rdm as cid_one_rdm,
)
from manybody_lib.configuration_interaction.spin_orbital.cid import (
    compute_sigma as cid_sigma,
)
from manybody_lib.configuration_interaction.spin_orbital.cid import (
    compute_two_rdm as cid_two_rdm,
)
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


def _complex_random(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _antisymmetric_doubles(raw: np.ndarray) -> np.ndarray:
    return (
        raw
        - raw.swapaxes(0, 1)
        - raw.swapaxes(2, 3)
        + raw.swapaxes(0, 1).swapaxes(2, 3)
    ) / 4.0


def _hermitian_spin_orbital_system(
    rng: np.random.Generator, nocc: int, nvir: int
) -> tuple[complex, np.ndarray, np.ndarray, np.ndarray, slice, slice]:
    norb = nocc + nvir
    o, v = slice(0, nocc), slice(nocc, norb)
    raw_h = _complex_random(rng, (norb, norb))
    h = 0.5 * (raw_h + raw_h.T.conj())

    raw_u = _complex_random(rng, (norb, norb, norb, norb))
    u = (
        raw_u
        - raw_u.swapaxes(0, 1)
        - raw_u.swapaxes(2, 3)
        + raw_u.swapaxes(0, 1).swapaxes(2, 3)
    ) / 4.0
    u = 0.5 * (u + u.transpose(2, 3, 0, 1).conj())

    f = h + np.einsum("piqi->pq", u[:, o, :, o], optimize=True)
    e_ref = np.trace(h[o, o]) + 0.5 * np.einsum(
        "ijij->", u[o, o, o, o], optimize=True
    )
    return e_ref, f, u, h, o, v


def _assert_spin_rdm_invariants(
    gamma: np.ndarray, gamma2: np.ndarray, norm: complex, nelec: int
) -> None:
    np.testing.assert_allclose(gamma, gamma.T.conj(), atol=2e-11)
    np.testing.assert_allclose(
        gamma2, gamma2.transpose(2, 3, 0, 1).conj(), atol=2e-11
    )
    np.testing.assert_allclose(gamma2, -gamma2.swapaxes(0, 1), atol=2e-11)
    np.testing.assert_allclose(gamma2, -gamma2.swapaxes(2, 3), atol=2e-11)
    np.testing.assert_allclose(np.trace(gamma), nelec * norm, atol=2e-11)
    np.testing.assert_allclose(
        np.einsum("pqrq->pr", gamma2, optimize=True),
        (nelec - 1) * gamma,
        atol=2e-11,
    )


@pytest.mark.parametrize("method", ["cis", "cid", "cisd"])
def test_spin_orbital_rdm_invariants(method: str) -> None:
    rng = np.random.default_rng(714)
    nocc, nvir = 3, 4
    c0 = np.complex128(0.4 - 0.2j)

    if method == "cis":
        amplitudes = _complex_random(rng, (nvir, nocc))
        norm = cis_norm(c0, amplitudes)
        gamma = cis_one_rdm(c0, amplitudes)
        gamma2 = cis_two_rdm(c0, amplitudes)
    elif method == "cid":
        amplitudes = _antisymmetric_doubles(
            _complex_random(rng, (nvir, nvir, nocc, nocc))
        )
        norm = cid_norm(c0, amplitudes)
        gamma = cid_one_rdm(c0, amplitudes)
        gamma2 = cid_two_rdm(c0, amplitudes)
    else:
        singles = _complex_random(rng, (nvir, nocc))
        doubles = _antisymmetric_doubles(
            _complex_random(rng, (nvir, nvir, nocc, nocc))
        )
        norm = cisd_norm(c0, singles, doubles)
        gamma = cisd_one_rdm(c0, singles, doubles)
        gamma2 = cisd_two_rdm(c0, singles, doubles)

    _assert_spin_rdm_invariants(gamma, gamma2, norm, nocc)


@pytest.mark.parametrize("method", ["cis", "cid", "cisd"])
def test_spin_orbital_sigma_energy_closure(method: str) -> None:
    rng = np.random.default_rng(812)
    nocc, nvir = 3, 4
    e_ref, f, u, h, o, v = _hermitian_spin_orbital_system(rng, nocc, nvir)
    c0 = np.complex128(-0.3 + 0.7j)

    if method == "cis":
        amplitudes = _complex_random(rng, (nvir, nocc))
        sigma0, sigma = cis_sigma(e_ref, f, u, c0, amplitudes, o, v)
        energy_sigma = np.conjugate(c0) * sigma0 + np.einsum(
            "ai,ai->", amplitudes.conj(), sigma, optimize=True
        )
        gamma = cis_one_rdm(c0, amplitudes)
        gamma2 = cis_two_rdm(c0, amplitudes)
    elif method == "cid":
        amplitudes = _antisymmetric_doubles(
            _complex_random(rng, (nvir, nvir, nocc, nocc))
        )
        sigma0, sigma = cid_sigma(e_ref, f, u, c0, amplitudes, o, v)
        energy_sigma = np.conjugate(c0) * sigma0 + 0.25 * np.einsum(
            "abij,abij->", amplitudes.conj(), sigma, optimize=True
        )
        gamma = cid_one_rdm(c0, amplitudes)
        gamma2 = cid_two_rdm(c0, amplitudes)
    else:
        singles = _complex_random(rng, (nvir, nocc))
        doubles = _antisymmetric_doubles(
            _complex_random(rng, (nvir, nvir, nocc, nocc))
        )
        sigma0, sigma1, sigma2 = cisd_sigma(
            e_ref, f, u, c0, singles, doubles, o, v
        )
        energy_sigma = np.conjugate(c0) * sigma0
        energy_sigma += np.einsum(
            "ai,ai->", singles.conj(), sigma1, optimize=True
        )
        energy_sigma += 0.25 * np.einsum(
            "abij,abij->", doubles.conj(), sigma2, optimize=True
        )
        gamma = cisd_one_rdm(c0, singles, doubles)
        gamma2 = cisd_two_rdm(c0, singles, doubles)

    energy_rdm = np.einsum("pq,pq->", h, gamma, optimize=True)
    energy_rdm += 0.25 * np.einsum(
        "pqrs,pqrs->", u, gamma2, optimize=True
    )
    np.testing.assert_allclose(energy_sigma, energy_rdm, atol=3e-10)


@pytest.mark.parametrize("method", ["cis", "cid"])
def test_spin_orbital_sigma_is_complex_linear(method: str) -> None:
    rng = np.random.default_rng(917)
    nocc, nvir = 2, 3
    e_ref, f, u, _, o, v = _hermitian_spin_orbital_system(rng, nocc, nvir)
    alpha = np.complex128(0.6 - 1.3j)
    c0 = np.complex128(0.2 + 0.9j)

    if method == "cis":
        amplitudes = _complex_random(rng, (nvir, nocc))
        evaluator = cis_sigma
    else:
        amplitudes = _antisymmetric_doubles(
            _complex_random(rng, (nvir, nvir, nocc, nocc))
        )
        evaluator = cid_sigma

    sigma0, sigma = evaluator(e_ref, f, u, c0, amplitudes, o, v)
    scaled0, scaled = evaluator(
        e_ref, f, u, alpha * c0, alpha * amplitudes, o, v
    )
    np.testing.assert_allclose(scaled0, alpha * sigma0, atol=2e-11)
    np.testing.assert_allclose(scaled, alpha * sigma, atol=2e-11)

    if method == "cid":
        np.testing.assert_allclose(sigma, -sigma.swapaxes(0, 1), atol=2e-11)
        np.testing.assert_allclose(sigma, -sigma.swapaxes(2, 3), atol=2e-11)


def test_spin_orbital_cisd_reduces_to_cis_and_cid() -> None:
    rng = np.random.default_rng(991)
    nocc, nvir = 2, 3
    e_ref, f, u, _, o, v = _hermitian_spin_orbital_system(rng, nocc, nvir)
    c0 = np.complex128(0.1 - 0.7j)
    singles = _complex_random(rng, (nvir, nocc))
    doubles = _antisymmetric_doubles(
        _complex_random(rng, (nvir, nvir, nocc, nocc))
    )
    zero1 = np.zeros_like(singles)
    zero2 = np.zeros_like(doubles)

    cisd0, cisd1, cisd2 = cisd_sigma(e_ref, f, u, c0, singles, zero2, o, v)
    cis0, cis1 = cis_sigma(e_ref, f, u, c0, singles, o, v)
    np.testing.assert_allclose(cisd0, cis0, atol=2e-11)
    np.testing.assert_allclose(cisd1, cis1, atol=2e-11)
    np.testing.assert_allclose(
        cisd_one_rdm(c0, singles, zero2), cis_one_rdm(c0, singles), atol=2e-11
    )
    np.testing.assert_allclose(
        cisd_two_rdm(c0, singles, zero2), cis_two_rdm(c0, singles), atol=2e-11
    )

    cisd0, _, cisd2 = cisd_sigma(e_ref, f, u, c0, zero1, doubles, o, v)
    cid0, cid2 = cid_sigma(e_ref, f, u, c0, doubles, o, v)
    np.testing.assert_allclose(cisd0, cid0, atol=2e-11)
    np.testing.assert_allclose(cisd2, cid2, atol=2e-11)
    np.testing.assert_allclose(
        cisd_one_rdm(c0, zero1, doubles), cid_one_rdm(c0, doubles), atol=2e-11
    )
    np.testing.assert_allclose(
        cisd_two_rdm(c0, zero1, doubles), cid_two_rdm(c0, doubles), atol=2e-11
    )
