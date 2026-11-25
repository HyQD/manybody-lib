"""Invariant and energy-closure tests for HJO closed-shell CIS and CID."""

from __future__ import annotations

import numpy as np
import pytest

from manybody_lib.configuration_interaction.closed_shell.cis import (
    compute_hermitian_left as cis_left,
)
from manybody_lib.configuration_interaction.closed_shell.cis import (
    compute_norm as cis_norm,
)
from manybody_lib.configuration_interaction.closed_shell.cis import (
    compute_one_rdm as cis_one_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cis import (
    compute_sigma as cis_sigma,
)
from manybody_lib.configuration_interaction.closed_shell.cis import (
    compute_two_rdm as cis_two_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cid import (
    compute_hermitian_left as cid_left,
)
from manybody_lib.configuration_interaction.closed_shell.cid import (
    compute_norm as cid_norm,
)
from manybody_lib.configuration_interaction.closed_shell.cid import (
    compute_one_rdm as cid_one_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cid import (
    compute_sigma as cid_sigma,
)
from manybody_lib.configuration_interaction.closed_shell.cid import (
    compute_two_rdm as cid_two_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cisd import (
    compute_hermitian_left as cisd_left,
)
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


def _complex_random(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _column_symmetric_doubles(raw: np.ndarray) -> np.ndarray:
    return 0.5 * (raw + raw.transpose(1, 0, 3, 2))


def _hermitian_spatial_system(
    rng: np.random.Generator, nocc: int, nvir: int
) -> tuple[complex, np.ndarray, np.ndarray, np.ndarray, slice, slice]:
    norb = nocc + nvir
    o, v = slice(0, nocc), slice(nocc, norb)
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
    f += 2.0 * np.einsum("piqi->pq", u[:, o, :, o], optimize=True)
    f -= np.einsum("piiq->pq", u[:, o, o, :], optimize=True)
    e_ref = 2.0 * np.trace(h[o, o])
    e_ref += 2.0 * np.einsum("ijij->", u[o, o, o, o], optimize=True)
    e_ref -= np.einsum("ijji->", u[o, o, o, o], optimize=True)
    return e_ref, f, u, h, o, v


def _assert_closed_shell_rdm_invariants(
    gamma: np.ndarray, gamma2: np.ndarray, norm: complex, nelec: int
) -> None:
    np.testing.assert_allclose(gamma, gamma.T.conj(), atol=3e-11)
    np.testing.assert_allclose(
        gamma2, gamma2.transpose(2, 3, 0, 1).conj(), atol=3e-11
    )
    np.testing.assert_allclose(
        gamma2, gamma2.transpose(1, 0, 3, 2), atol=3e-11
    )
    np.testing.assert_allclose(np.trace(gamma), nelec * norm, atol=3e-11)
    np.testing.assert_allclose(
        np.einsum("pqrq->pr", gamma2, optimize=True),
        (nelec - 1) * gamma,
        atol=3e-11,
    )


@pytest.mark.parametrize("method", ["cis", "cid", "cisd"])
def test_closed_shell_rdm_invariants(method: str) -> None:
    rng = np.random.default_rng(1031)
    nocc, nvir = 3, 4
    c0 = np.complex128(0.8 + 0.1j)

    if method == "cis":
        amplitudes = _complex_random(rng, (nvir, nocc))
        norm = cis_norm(c0, amplitudes)
        gamma = cis_one_rdm(c0, amplitudes)
        gamma2 = cis_two_rdm(c0, amplitudes)
    elif method == "cid":
        amplitudes = _column_symmetric_doubles(
            _complex_random(rng, (nvir, nvir, nocc, nocc))
        )
        norm = cid_norm(c0, amplitudes)
        gamma = cid_one_rdm(c0, amplitudes)
        gamma2 = cid_two_rdm(c0, amplitudes)
    else:
        singles = _complex_random(rng, (nvir, nocc))
        doubles = _column_symmetric_doubles(
            _complex_random(rng, (nvir, nvir, nocc, nocc))
        )
        norm = cisd_norm(c0, singles, doubles)
        gamma = cisd_one_rdm(c0, singles, doubles)
        gamma2 = cisd_two_rdm(c0, singles, doubles)

    _assert_closed_shell_rdm_invariants(gamma, gamma2, norm, 2 * nocc)


@pytest.mark.parametrize("method", ["cis", "cid", "cisd"])
def test_closed_shell_sigma_energy_closure(method: str) -> None:
    rng = np.random.default_rng(1129)
    nocc, nvir = 3, 4
    e_ref, f, u, h, o, v = _hermitian_spatial_system(rng, nocc, nvir)
    c0 = np.complex128(-0.5 + 0.4j)

    if method == "cis":
        amplitudes = _complex_random(rng, (nvir, nocc))
        left0, left = cis_left(c0, amplitudes)
        sigma0, sigma = cis_sigma(e_ref, f, u, c0, amplitudes, o, v)
        energy_sigma = left0 * sigma0 + np.einsum(
            "ia,ai->", left, sigma, optimize=True
        )
        gamma = cis_one_rdm(c0, amplitudes, left0, left)
        gamma2 = cis_two_rdm(c0, amplitudes, left0, left)
    elif method == "cid":
        amplitudes = _column_symmetric_doubles(
            _complex_random(rng, (nvir, nvir, nocc, nocc))
        )
        left0, left = cid_left(c0, amplitudes)
        sigma0, sigma = cid_sigma(e_ref, f, u, c0, amplitudes, o, v)
        energy_sigma = left0 * sigma0 + 0.5 * np.einsum(
            "ijab,abij->", left, sigma, optimize=True
        )
        gamma = cid_one_rdm(c0, amplitudes, left0, left)
        gamma2 = cid_two_rdm(c0, amplitudes, left0, left)
    else:
        singles = _complex_random(rng, (nvir, nocc))
        doubles = _column_symmetric_doubles(
            _complex_random(rng, (nvir, nvir, nocc, nocc))
        )
        left0, left1, left2 = cisd_left(c0, singles, doubles)
        sigma0, sigma1, sigma2 = cisd_sigma(
            e_ref, f, u, c0, singles, doubles, o, v
        )
        energy_sigma = left0 * sigma0
        energy_sigma += np.einsum("ia,ai->", left1, sigma1, optimize=True)
        energy_sigma += 0.5 * np.einsum(
            "ijab,abij->", left2, sigma2, optimize=True
        )
        gamma = cisd_one_rdm(c0, singles, doubles, left0, left1, left2)
        gamma2 = cisd_two_rdm(c0, singles, doubles, left0, left1, left2)

    energy_rdm = np.einsum("pq,pq->", h, gamma, optimize=True)
    energy_rdm += 0.5 * np.einsum(
        "pqrs,pqrs->", u, gamma2, optimize=True
    )
    np.testing.assert_allclose(energy_sigma, energy_rdm, atol=5e-10)


@pytest.mark.parametrize("method", ["cis", "cid"])
def test_closed_shell_sigma_is_complex_linear(method: str) -> None:
    rng = np.random.default_rng(1223)
    nocc, nvir = 2, 3
    e_ref, f, u, _, o, v = _hermitian_spatial_system(rng, nocc, nvir)
    alpha = np.complex128(-0.8 + 0.3j)
    c0 = np.complex128(0.7 - 0.2j)

    if method == "cis":
        amplitudes = _complex_random(rng, (nvir, nocc))
        evaluator = cis_sigma
    else:
        amplitudes = _column_symmetric_doubles(
            _complex_random(rng, (nvir, nvir, nocc, nocc))
        )
        evaluator = cid_sigma

    sigma0, sigma = evaluator(e_ref, f, u, c0, amplitudes, o, v)
    scaled0, scaled = evaluator(
        e_ref, f, u, alpha * c0, alpha * amplitudes, o, v
    )
    np.testing.assert_allclose(scaled0, alpha * sigma0, atol=3e-11)
    np.testing.assert_allclose(scaled, alpha * sigma, atol=3e-11)
    if method == "cid":
        np.testing.assert_allclose(
            sigma, sigma.transpose(1, 0, 3, 2), atol=3e-11
        )


def test_independent_left_amplitudes_are_supported() -> None:
    rng = np.random.default_rng(1301)
    nocc, nvir = 2, 3
    c0 = np.complex128(0.2 + 0.4j)
    c1 = _complex_random(rng, (nvir, nocc))
    left0 = np.complex128(-0.1 + 0.9j)
    left1 = _complex_random(rng, (nocc, nvir))

    norm = cis_norm(c0, c1, left0, left1)
    gamma = cis_one_rdm(c0, c1, left0, left1)
    gamma2 = cis_two_rdm(c0, c1, left0, left1)
    np.testing.assert_allclose(np.trace(gamma), 2 * nocc * norm, atol=3e-11)
    np.testing.assert_allclose(
        np.einsum("pqrq->pr", gamma2, optimize=True),
        (2 * nocc - 1) * gamma,
        atol=3e-11,
    )

    with pytest.raises(ValueError, match="supplied together"):
        cis_one_rdm(c0, c1, left_c0=left0)


def test_closed_shell_cisd_reduces_to_cis_and_cid() -> None:
    rng = np.random.default_rng(1373)
    nocc, nvir = 2, 3
    e_ref, f, u, _, o, v = _hermitian_spatial_system(rng, nocc, nvir)
    c0 = np.complex128(-0.4 + 0.2j)
    singles = _complex_random(rng, (nvir, nocc))
    doubles = _column_symmetric_doubles(
        _complex_random(rng, (nvir, nvir, nocc, nocc))
    )
    zero1 = np.zeros_like(singles)
    zero2 = np.zeros_like(doubles)

    cisd0, cisd1, _ = cisd_sigma(e_ref, f, u, c0, singles, zero2, o, v)
    cis0, cis1 = cis_sigma(e_ref, f, u, c0, singles, o, v)
    np.testing.assert_allclose(cisd0, cis0, atol=3e-11)
    np.testing.assert_allclose(cisd1, cis1, atol=3e-11)
    np.testing.assert_allclose(
        cisd_one_rdm(c0, singles, zero2), cis_one_rdm(c0, singles), atol=3e-11
    )
    np.testing.assert_allclose(
        cisd_two_rdm(c0, singles, zero2), cis_two_rdm(c0, singles), atol=3e-11
    )

    cisd0, _, cisd2 = cisd_sigma(e_ref, f, u, c0, zero1, doubles, o, v)
    cid0, cid2 = cid_sigma(e_ref, f, u, c0, doubles, o, v)
    np.testing.assert_allclose(cisd0, cid0, atol=3e-11)
    np.testing.assert_allclose(cisd2, cid2, atol=3e-11)
    np.testing.assert_allclose(
        cisd_one_rdm(c0, zero1, doubles), cid_one_rdm(c0, doubles), atol=3e-11
    )
    np.testing.assert_allclose(
        cisd_two_rdm(c0, zero1, doubles), cid_two_rdm(c0, doubles), atol=3e-11
    )


def test_closed_shell_cisd_supports_independent_left_state() -> None:
    rng = np.random.default_rng(1427)
    nocc, nvir = 2, 3
    e_ref, f, u, h, o, v = _hermitian_spatial_system(rng, nocc, nvir)
    c0 = np.complex128(0.3 - 0.8j)
    c1 = _complex_random(rng, (nvir, nocc))
    c2 = _column_symmetric_doubles(
        _complex_random(rng, (nvir, nvir, nocc, nocc))
    )
    left0 = np.complex128(-0.2 + 0.5j)
    left1 = _complex_random(rng, (nocc, nvir))
    raw_left2 = _complex_random(rng, (nocc, nocc, nvir, nvir))
    left2 = 0.5 * (raw_left2 + raw_left2.transpose(1, 0, 3, 2))

    norm = cisd_norm(c0, c1, c2, left0, left1, left2)
    gamma = cisd_one_rdm(c0, c1, c2, left0, left1, left2)
    gamma2 = cisd_two_rdm(c0, c1, c2, left0, left1, left2)
    np.testing.assert_allclose(np.trace(gamma), 2 * nocc * norm, atol=5e-11)
    np.testing.assert_allclose(
        np.einsum("pqrq->pr", gamma2, optimize=True),
        (2 * nocc - 1) * gamma,
        atol=5e-11,
    )

    sigma0, sigma1, sigma2 = cisd_sigma(e_ref, f, u, c0, c1, c2, o, v)
    energy_sigma = left0 * sigma0
    energy_sigma += np.einsum("ia,ai->", left1, sigma1, optimize=True)
    energy_sigma += 0.5 * np.einsum(
        "ijab,abij->", left2, sigma2, optimize=True
    )
    energy_rdm = np.einsum("pq,pq->", h, gamma, optimize=True)
    energy_rdm += 0.5 * np.einsum(
        "pqrs,pqrs->", u, gamma2, optimize=True
    )
    np.testing.assert_allclose(energy_sigma, energy_rdm, atol=8e-10)

    with pytest.raises(ValueError, match="supplied together"):
        cisd_one_rdm(c0, c1, c2, left_c0=left0)
