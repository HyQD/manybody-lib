"""PySCF-backed equivalence tests for spin and closed-shell CI kernels."""

from __future__ import annotations

import numpy as np
import pytest

from manybody_lib.configuration_interaction.closed_shell.cis import (
    compute_norm as closed_cis_norm,
)
from manybody_lib.configuration_interaction.closed_shell.cis import (
    compute_one_rdm as closed_cis_one_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cis import (
    compute_sigma as closed_cis_sigma,
)
from manybody_lib.configuration_interaction.closed_shell.cis import (
    compute_two_rdm as closed_cis_two_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cid import (
    compute_norm as closed_cid_norm,
)
from manybody_lib.configuration_interaction.closed_shell.cid import (
    compute_one_rdm as closed_cid_one_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cid import (
    compute_sigma as closed_cid_sigma,
)
from manybody_lib.configuration_interaction.closed_shell.cid import (
    compute_two_rdm as closed_cid_two_rdm,
)
from manybody_lib.configuration_interaction.spin_orbital.cis import (
    compute_norm as spin_cis_norm,
)
from manybody_lib.configuration_interaction.spin_orbital.cis import (
    compute_one_rdm as spin_cis_one_rdm,
)
from manybody_lib.configuration_interaction.spin_orbital.cis import (
    compute_sigma as spin_cis_sigma,
)
from manybody_lib.configuration_interaction.spin_orbital.cis import (
    compute_two_rdm as spin_cis_two_rdm,
)
from manybody_lib.configuration_interaction.spin_orbital.cid import (
    compute_norm as spin_cid_norm,
)
from manybody_lib.configuration_interaction.spin_orbital.cid import (
    compute_one_rdm as spin_cid_one_rdm,
)
from manybody_lib.configuration_interaction.spin_orbital.cid import (
    compute_sigma as spin_cid_sigma,
)
from manybody_lib.configuration_interaction.spin_orbital.cid import (
    compute_two_rdm as spin_cid_two_rdm,
)

pyscf = pytest.importorskip("pyscf")
from pyscf import ao2mo, gto, scf  # noqa: E402


def _random_unitary(rng: np.random.Generator, size: int) -> np.ndarray:
    raw = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    unitary, triangular = np.linalg.qr(raw)
    phases = np.diag(triangular)
    phases = np.where(np.abs(phases) > 0.0, phases / np.abs(phases), 1.0)
    return unitary * phases.conj()


@pytest.fixture(scope="module")
def rhf_integrals() -> tuple[complex, np.ndarray, np.ndarray, int]:
    """Return complex-rotated BeH2/STO-3G RHF MO-basis integrals."""

    molecule = gto.M(
        atom="H 0 0 -1.3; Be 0 0 0; H 0 0 1.3",
        basis="sto-3g",
        unit="Angstrom",
        spin=0,
        charge=0,
        verbose=0,
    )
    mean_field = scf.RHF(molecule).run(conv_tol=1e-12)
    assert mean_field.converged

    coefficients = mean_field.mo_coeff
    h_ao = mean_field.get_hcore()
    h = coefficients.T.conj() @ h_ao @ coefficients
    norb = coefficients.shape[1]
    eri_chemist = ao2mo.restore(
        1, ao2mo.kernel(molecule, coefficients), norb
    )
    u = eri_chemist.transpose(0, 2, 1, 3)
    nocc = molecule.nelectron // 2

    rng = np.random.default_rng(1409)
    rotation = np.zeros((norb, norb), dtype=np.complex128)
    rotation[:nocc, :nocc] = _random_unitary(rng, nocc)
    rotation[nocc:, nocc:] = _random_unitary(rng, norb - nocc)
    h = rotation.T.conj() @ h @ rotation
    u = np.einsum(
        "xp,yq,xyzw,zr,ws->pqrs",
        rotation.conj(),
        rotation.conj(),
        u,
        rotation,
        rotation,
        optimize=True,
    )

    o = slice(0, nocc)
    e_ref = 2.0 * np.trace(h[o, o])
    e_ref += 2.0 * np.einsum("ijij->", u[o, o, o, o], optimize=True)
    e_ref -= np.einsum("ijji->", u[o, o, o, o], optimize=True)
    np.testing.assert_allclose(
        e_ref.real + molecule.energy_nuc(), mean_field.e_tot, atol=2e-10
    )
    return e_ref, h, u, nocc


def _spatial_fock(h: np.ndarray, u: np.ndarray, nocc: int) -> np.ndarray:
    o = slice(0, nocc)
    f = h.copy()
    f += 2.0 * np.einsum("piqi->pq", u[:, o, :, o], optimize=True)
    f -= np.einsum("piiq->pq", u[:, o, o, :], optimize=True)
    return f


def _spin_integrals(
    h: np.ndarray, u: np.ndarray, nocc: int
) -> tuple[complex, np.ndarray, np.ndarray]:
    norb = h.shape[0]
    nso = 2 * norb
    h_spin = np.zeros((nso, nso), dtype=np.complex128)
    interaction = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
    for p in range(norb):
        for q in range(norb):
            for spin in range(2):
                h_spin[2 * p + spin, 2 * q + spin] = h[p, q]
            for r in range(norb):
                for s in range(norb):
                    for spin_p in range(2):
                        for spin_q in range(2):
                            interaction[
                                2 * p + spin_p,
                                2 * q + spin_q,
                                2 * r + spin_p,
                                2 * s + spin_q,
                            ] = u[p, q, r, s]
    antisymmetrized = interaction - interaction.swapaxes(2, 3)
    nocc_spin = 2 * nocc
    o = slice(0, nocc_spin)
    f_spin = h_spin + np.einsum(
        "piqi->pq", antisymmetrized[:, o, :, o], optimize=True
    )
    e_ref = np.trace(h_spin[o, o]) + 0.5 * np.einsum(
        "ijij->", antisymmetrized[o, o, o, o], optimize=True
    )
    return e_ref, f_spin, antisymmetrized


def _map_singles_to_spin(c1: np.ndarray) -> np.ndarray:
    nvir, nocc = c1.shape
    spin = np.zeros((2 * nvir, 2 * nocc), dtype=np.complex128)
    for a in range(nvir):
        for i in range(nocc):
            for sigma in range(2):
                spin[2 * a + sigma, 2 * i + sigma] = c1[a, i]
    return spin


def _map_doubles_to_spin(c2: np.ndarray) -> np.ndarray:
    nvir, _, nocc, _ = c2.shape
    spin = np.zeros(
        (2 * nvir, 2 * nvir, 2 * nocc, 2 * nocc), dtype=np.complex128
    )
    for a_spin in range(2 * nvir):
        a, sigma_a = divmod(a_spin, 2)
        for b_spin in range(2 * nvir):
            b, sigma_b = divmod(b_spin, 2)
            for i_spin in range(2 * nocc):
                i, sigma_i = divmod(i_spin, 2)
                for j_spin in range(2 * nocc):
                    j, sigma_j = divmod(j_spin, 2)
                    value = 0.0j
                    if sigma_a == sigma_i and sigma_b == sigma_j:
                        value += c2[a, b, i, j]
                    if sigma_a == sigma_j and sigma_b == sigma_i:
                        value -= c2[a, b, j, i]
                    spin[a_spin, b_spin, i_spin, j_spin] = value
    return spin


def _spin_trace_one_rdm(gamma: np.ndarray) -> np.ndarray:
    norb = gamma.shape[0] // 2
    traced = np.zeros((norb, norb), dtype=gamma.dtype)
    for p in range(norb):
        for q in range(norb):
            traced[p, q] = sum(
                gamma[2 * p + spin, 2 * q + spin] for spin in range(2)
            )
    return traced


def _spin_trace_two_rdm(gamma2: np.ndarray) -> np.ndarray:
    norb = gamma2.shape[0] // 2
    traced = np.zeros((norb, norb, norb, norb), dtype=gamma2.dtype)
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    traced[p, q, r, s] = sum(
                        gamma2[
                            2 * p + sigma,
                            2 * q + tau,
                            2 * r + sigma,
                            2 * s + tau,
                        ]
                        for sigma in range(2)
                        for tau in range(2)
                    )
    return traced


@pytest.mark.integration
@pytest.mark.parametrize("method", ["cis", "cid"])
def test_pyscf_spin_and_closed_shell_agree(
    rhf_integrals: tuple[complex, np.ndarray, np.ndarray, int], method: str
) -> None:
    e_ref, h, u, nocc = rhf_integrals
    norb = h.shape[0]
    nvir = norb - nocc
    closed_f = _spatial_fock(h, u, nocc)
    spin_e_ref, spin_f, spin_u = _spin_integrals(h, u, nocc)
    np.testing.assert_allclose(spin_e_ref, e_ref, atol=2e-10)

    rng = np.random.default_rng(1511 if method == "cis" else 1601)
    c0 = np.complex128(0.3 - 0.6j)
    o, v = slice(0, nocc), slice(nocc, norb)
    o_spin, v_spin = slice(0, 2 * nocc), slice(2 * nocc, 2 * norb)

    if method == "cis":
        closed = rng.standard_normal((nvir, nocc))
        closed = closed + 1j * rng.standard_normal(closed.shape)
        spin = _map_singles_to_spin(closed)
        closed_sigma0, closed_sigma = closed_cis_sigma(
            e_ref, closed_f, u, c0, closed, o, v
        )
        spin_sigma0, spin_sigma = spin_cis_sigma(
            spin_e_ref, spin_f, spin_u, c0, spin, o_spin, v_spin
        )
        expected_spin_sigma = _map_singles_to_spin(closed_sigma)
        closed_norm = closed_cis_norm(c0, closed)
        spin_norm = spin_cis_norm(c0, spin)
        closed_gamma = closed_cis_one_rdm(c0, closed)
        spin_gamma = spin_cis_one_rdm(c0, spin)
        closed_gamma2 = closed_cis_two_rdm(c0, closed)
        spin_gamma2 = spin_cis_two_rdm(c0, spin)
    else:
        raw = rng.standard_normal((nvir, nvir, nocc, nocc))
        raw = raw + 1j * rng.standard_normal(raw.shape)
        closed = 0.5 * (raw + raw.transpose(1, 0, 3, 2))
        spin = _map_doubles_to_spin(closed)
        closed_sigma0, closed_sigma = closed_cid_sigma(
            e_ref, closed_f, u, c0, closed, o, v
        )
        spin_sigma0, spin_sigma = spin_cid_sigma(
            spin_e_ref, spin_f, spin_u, c0, spin, o_spin, v_spin
        )
        expected_spin_sigma = _map_doubles_to_spin(closed_sigma)
        closed_norm = closed_cid_norm(c0, closed)
        spin_norm = spin_cid_norm(c0, spin)
        closed_gamma = closed_cid_one_rdm(c0, closed)
        spin_gamma = spin_cid_one_rdm(c0, spin)
        closed_gamma2 = closed_cid_two_rdm(c0, closed)
        spin_gamma2 = spin_cid_two_rdm(c0, spin)

    np.testing.assert_allclose(spin_norm, closed_norm, atol=3e-10)
    np.testing.assert_allclose(spin_sigma0, closed_sigma0, atol=3e-10)
    np.testing.assert_allclose(spin_sigma, expected_spin_sigma, atol=5e-10)
    np.testing.assert_allclose(
        _spin_trace_one_rdm(spin_gamma), closed_gamma, atol=3e-10
    )
    np.testing.assert_allclose(
        _spin_trace_two_rdm(spin_gamma2), closed_gamma2, atol=5e-10
    )
