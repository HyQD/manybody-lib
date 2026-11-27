"""BH/STO-3G cross-validation of spin-orbital and closed-shell CISDT."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from manybody_lib.configuration_interaction.closed_shell.cidt import (
    compute_norm as closed_cidt_norm,
)
from manybody_lib.configuration_interaction.closed_shell.cidt import (
    compute_one_rdm as closed_cidt_one_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cidt import (
    compute_sigma as closed_cidt_sigma,
)
from manybody_lib.configuration_interaction.closed_shell.cidt import (
    compute_sigma_optimized as closed_cidt_sigma_optimized,
)
from manybody_lib.configuration_interaction.closed_shell.cidt import (
    compute_two_rdm as closed_cidt_two_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt import (
    compute_norm as closed_norm,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt import (
    compute_one_rdm as closed_one_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt import (
    compute_sigma as closed_sigma,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt import (
    compute_sigma_optimized as closed_sigma_optimized,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt import (
    compute_two_rdm as closed_two_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt import (
    project_physical_triples,
    symmetrize_triples_columns,
)
from manybody_lib.configuration_interaction.spin_orbital.cidt import (
    compute_norm as spin_cidt_norm,
)
from manybody_lib.configuration_interaction.spin_orbital.cidt import (
    compute_one_rdm as spin_cidt_one_rdm,
)
from manybody_lib.configuration_interaction.spin_orbital.cidt import (
    compute_sigma as spin_cidt_sigma,
)
from manybody_lib.configuration_interaction.spin_orbital.cidt import (
    compute_two_rdm as spin_cidt_two_rdm,
)
from manybody_lib.configuration_interaction.spin_orbital.cisdt import (
    compute_norm as spin_norm,
)
from manybody_lib.configuration_interaction.spin_orbital.cisdt import (
    compute_one_rdm as spin_one_rdm,
)
from manybody_lib.configuration_interaction.spin_orbital.cisdt import (
    compute_sigma as spin_sigma,
)
from manybody_lib.configuration_interaction.spin_orbital.cisdt import (
    compute_two_rdm as spin_two_rdm,
)


pyscf = pytest.importorskip("pyscf")
from pyscf import ao2mo, gto, scf  # noqa: E402


_S3 = tuple(itertools.permutations(range(3)))


def _permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _random_unitary(rng: np.random.Generator, size: int) -> np.ndarray:
    raw = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    unitary, triangular = np.linalg.qr(raw)
    diagonal = np.diag(triangular)
    phases = np.where(np.abs(diagonal) > 0.0, diagonal / np.abs(diagonal), 1.0)
    return unitary * phases.conj()


@pytest.fixture(scope="module")
def bh_integrals() -> tuple[complex, np.ndarray, np.ndarray, int]:
    """Return complex-rotated BH/STO-3G RHF MO-basis integrals."""

    molecule = gto.M(
        atom="B 0 0 0; H 0 0 1.23",
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
        1,
        ao2mo.kernel(molecule, coefficients),
        norb,
    )
    u = eri_chemist.transpose(0, 2, 1, 3)
    nocc = molecule.nelectron // 2

    rng = np.random.default_rng(1801)
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

    occupied = slice(0, nocc)
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
    np.testing.assert_allclose(
        e_ref.real + molecule.energy_nuc(),
        mean_field.e_tot,
        atol=2e-10,
    )
    return e_ref, h, u, nocc


def _spatial_fock(h: np.ndarray, u: np.ndarray, nocc: int) -> np.ndarray:
    occupied = slice(0, nocc)
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
    return f


def _spin_integrals(
    h: np.ndarray,
    u: np.ndarray,
    nocc: int,
) -> tuple[complex, np.ndarray, np.ndarray]:
    norb = h.shape[0]
    nso = 2 * norb
    h_spin = np.zeros((nso, nso), dtype=np.complex128)
    interaction = np.zeros((nso, nso, nso, nso), dtype=np.complex128)
    for p, q in itertools.product(range(norb), repeat=2):
        for sigma in range(2):
            h_spin[2 * p + sigma, 2 * q + sigma] = h[p, q]
    for p, q, r, s in itertools.product(range(norb), repeat=4):
        for sigma in range(2):
            for tau in range(2):
                interaction[
                    2 * p + sigma,
                    2 * q + tau,
                    2 * r + sigma,
                    2 * s + tau,
                ] = u[p, q, r, s]
    antisymmetrized = interaction - interaction.swapaxes(2, 3)
    occupied = slice(0, 2 * nocc)
    f_spin = h_spin + np.einsum(
        "piqi->pq",
        antisymmetrized[:, occupied, :, occupied],
        optimize=True,
    )
    e_ref = np.trace(h_spin[occupied, occupied])
    e_ref += 0.5 * np.einsum(
        "ijij->",
        antisymmetrized[occupied, occupied, occupied, occupied],
        optimize=True,
    )
    return e_ref, f_spin, antisymmetrized


def _map_singles_to_spin(c1: np.ndarray) -> np.ndarray:
    nvir, nocc = c1.shape
    result = np.zeros((2 * nvir, 2 * nocc), dtype=np.complex128)
    for a, i, spin in itertools.product(range(nvir), range(nocc), range(2)):
        result[2 * a + spin, 2 * i + spin] = c1[a, i]
    return result


def _map_doubles_to_spin(c2: np.ndarray) -> np.ndarray:
    nvir, _, nocc, _ = c2.shape
    result = np.zeros(
        (2 * nvir, 2 * nvir, 2 * nocc, 2 * nocc),
        dtype=np.complex128,
    )
    for a_spin, b_spin, i_spin, j_spin in itertools.product(
        range(2 * nvir),
        range(2 * nvir),
        range(2 * nocc),
        range(2 * nocc),
    ):
        a, sigma_a = divmod(a_spin, 2)
        b, sigma_b = divmod(b_spin, 2)
        i, sigma_i = divmod(i_spin, 2)
        j, sigma_j = divmod(j_spin, 2)
        value = 0.0j
        if sigma_a == sigma_i and sigma_b == sigma_j:
            value += c2[a, b, i, j]
        if sigma_a == sigma_j and sigma_b == sigma_i:
            value -= c2[a, b, j, i]
        result[a_spin, b_spin, i_spin, j_spin] = value
    return result


def _map_triples_to_spin(c3: np.ndarray) -> np.ndarray:
    nvir, _, _, nocc, _, _ = c3.shape
    result = np.zeros(
        (2 * nvir, 2 * nvir, 2 * nvir, 2 * nocc, 2 * nocc, 2 * nocc),
        dtype=np.complex128,
    )
    for virtual_spins in itertools.product(range(2 * nvir), repeat=3):
        virtuals, virtual_spin_labels = zip(
            *(divmod(index, 2) for index in virtual_spins),
            strict=True,
        )
        for occupied_spins in itertools.product(range(2 * nocc), repeat=3):
            occupieds, occupied_spin_labels = zip(
                *(divmod(index, 2) for index in occupied_spins),
                strict=True,
            )
            value = 0.0j
            for permutation in _S3:
                if all(
                    virtual_spin_labels[position]
                    == occupied_spin_labels[permutation[position]]
                    for position in range(3)
                ):
                    permuted_occupied = tuple(
                        occupieds[permutation[position]] for position in range(3)
                    )
                    value += _permutation_sign(permutation) * c3[
                        virtuals + permuted_occupied
                    ]
            result[virtual_spins + occupied_spins] = value
    return result


def _spin_trace_one_rdm(gamma: np.ndarray) -> np.ndarray:
    norb = gamma.shape[0] // 2
    result = np.zeros((norb, norb), dtype=gamma.dtype)
    for p, q in itertools.product(range(norb), repeat=2):
        result[p, q] = sum(
            gamma[2 * p + spin, 2 * q + spin] for spin in range(2)
        )
    return result


def _spin_trace_two_rdm(gamma2: np.ndarray) -> np.ndarray:
    norb = gamma2.shape[0] // 2
    result = np.zeros((norb, norb, norb, norb), dtype=gamma2.dtype)
    for p, q, r, s in itertools.product(range(norb), repeat=4):
        result[p, q, r, s] = sum(
            gamma2[
                2 * p + sigma,
                2 * q + tau,
                2 * r + sigma,
                2 * s + tau,
            ]
            for sigma in range(2)
            for tau in range(2)
        )
    return result


@pytest.mark.integration
def test_bh_sto3g_spin_and_closed_shell_cisdt_agree(
    bh_integrals: tuple[complex, np.ndarray, np.ndarray, int],
) -> None:
    e_ref, h, u, nocc = bh_integrals
    norb = h.shape[0]
    nvir = norb - nocc
    closed_f = _spatial_fock(h, u, nocc)
    spin_e_ref, spin_f, spin_u = _spin_integrals(h, u, nocc)
    np.testing.assert_allclose(spin_e_ref, e_ref, atol=2e-10)

    rng = np.random.default_rng(1811)
    c0 = np.complex128(0.2 - 0.4j)
    closed1 = rng.standard_normal((nvir, nocc))
    closed1 = closed1 + 1j * rng.standard_normal(closed1.shape)
    raw2 = rng.standard_normal((nvir, nvir, nocc, nocc))
    raw2 = raw2 + 1j * rng.standard_normal(raw2.shape)
    closed2 = 0.5 * (raw2 + raw2.transpose(1, 0, 3, 2))
    raw3 = rng.standard_normal((nvir, nvir, nvir, nocc, nocc, nocc))
    raw3 = raw3 + 1j * rng.standard_normal(raw3.shape)
    closed3 = project_physical_triples(symmetrize_triples_columns(raw3))

    spin1 = _map_singles_to_spin(closed1)
    spin2 = _map_doubles_to_spin(closed2)
    spin3 = _map_triples_to_spin(closed3)
    occupied = slice(0, nocc)
    virtual = slice(nocc, norb)
    occupied_spin = slice(0, 2 * nocc)
    virtual_spin = slice(2 * nocc, 2 * norb)

    closed_args = (
        e_ref,
        closed_f,
        u,
        c0,
        closed1,
        closed2,
        closed3,
        occupied,
        virtual,
    )
    closed_values = closed_sigma(*closed_args)
    for combine_integrals, optimize_accumulation, use_gemm in (
        (False, False, False), (True, False, False), (True, True, False),
        (True, True, True),
    ):
        optimized_values = closed_sigma_optimized(
            *closed_args,
            assume_symmetry=True,
            combine_integrals=combine_integrals,
            optimize_accumulation=optimize_accumulation,
            use_gemm=use_gemm,
            gemm_block_size=1 if use_gemm else None,
        )
        for actual, expected in zip(optimized_values, closed_values, strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    spin_values = spin_sigma(
        spin_e_ref,
        spin_f,
        spin_u,
        c0,
        spin1,
        spin2,
        spin3,
        occupied_spin,
        virtual_spin,
    )
    expected_spin_values = (
        closed_values[0],
        _map_singles_to_spin(closed_values[1]),
        _map_doubles_to_spin(closed_values[2]),
        _map_triples_to_spin(closed_values[3]),
    )
    for actual, expected in zip(spin_values, expected_spin_values, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-9)

    np.testing.assert_allclose(
        spin_norm(c0, spin1, spin2, spin3),
        closed_norm(c0, closed1, closed2, closed3),
        atol=2e-9,
    )
    spin_gamma = spin_one_rdm(c0, spin1, spin2, spin3)
    closed_gamma = closed_one_rdm(c0, closed1, closed2, closed3)
    np.testing.assert_allclose(
        _spin_trace_one_rdm(spin_gamma),
        closed_gamma,
        atol=2e-9,
    )
    spin_gamma2 = spin_two_rdm(c0, spin1, spin2, spin3)
    closed_gamma2 = closed_two_rdm(c0, closed1, closed2, closed3)
    np.testing.assert_allclose(
        _spin_trace_two_rdm(spin_gamma2),
        closed_gamma2,
        atol=5e-9,
    )


@pytest.mark.integration
def test_bh_sto3g_spin_and_closed_shell_cidt_agree(
    bh_integrals: tuple[complex, np.ndarray, np.ndarray, int],
) -> None:
    e_ref, h, u, nocc = bh_integrals
    norb = h.shape[0]
    nvir = norb - nocc
    closed_f = _spatial_fock(h, u, nocc)
    spin_e_ref, spin_f, spin_u = _spin_integrals(h, u, nocc)

    rng = np.random.default_rng(1871)
    c0 = np.complex128(-0.3 + 0.2j)
    raw2 = rng.standard_normal((nvir, nvir, nocc, nocc))
    raw2 = raw2 + 1j * rng.standard_normal(raw2.shape)
    closed2 = 0.5 * (raw2 + raw2.transpose(1, 0, 3, 2))
    raw3 = rng.standard_normal((nvir, nvir, nvir, nocc, nocc, nocc))
    raw3 = raw3 + 1j * rng.standard_normal(raw3.shape)
    closed3 = project_physical_triples(symmetrize_triples_columns(raw3))
    spin2 = _map_doubles_to_spin(closed2)
    spin3 = _map_triples_to_spin(closed3)

    occupied = slice(0, nocc)
    virtual = slice(nocc, norb)
    occupied_spin = slice(0, 2 * nocc)
    virtual_spin = slice(2 * nocc, 2 * norb)
    closed_args = (
        e_ref,
        closed_f,
        u,
        c0,
        closed2,
        closed3,
        occupied,
        virtual,
    )
    closed_values = closed_cidt_sigma(*closed_args)
    for combine_integrals, optimize_accumulation, use_gemm in (
        (False, False, False), (True, False, False), (True, True, False),
        (True, True, True),
    ):
        optimized_values = closed_cidt_sigma_optimized(
            *closed_args,
            assume_symmetry=True,
            combine_integrals=combine_integrals,
            optimize_accumulation=optimize_accumulation,
            use_gemm=use_gemm,
            gemm_block_size=1 if use_gemm else None,
        )
        for actual, expected in zip(optimized_values, closed_values, strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    spin_values = spin_cidt_sigma(
        spin_e_ref,
        spin_f,
        spin_u,
        c0,
        spin2,
        spin3,
        occupied_spin,
        virtual_spin,
    )
    expected_spin_values = (
        closed_values[0],
        _map_doubles_to_spin(closed_values[1]),
        _map_triples_to_spin(closed_values[2]),
    )
    for actual, expected in zip(spin_values, expected_spin_values, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-9)

    np.testing.assert_allclose(
        spin_cidt_norm(c0, spin2, spin3),
        closed_cidt_norm(c0, closed2, closed3),
        atol=2e-9,
    )
    np.testing.assert_allclose(
        _spin_trace_one_rdm(spin_cidt_one_rdm(c0, spin2, spin3)),
        closed_cidt_one_rdm(c0, closed2, closed3),
        atol=2e-9,
    )
    np.testing.assert_allclose(
        _spin_trace_two_rdm(spin_cidt_two_rdm(c0, spin2, spin3)),
        closed_cidt_two_rdm(c0, closed2, closed3),
        atol=5e-9,
    )
