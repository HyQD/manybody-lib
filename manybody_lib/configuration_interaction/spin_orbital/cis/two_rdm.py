"""Two-body reduced density matrix for spin-orbital CIS."""

from __future__ import annotations

import numpy as np

from .amplitudes import compute_norm


def _antisymmetrize_pairs(block: np.ndarray) -> np.ndarray:
    """Apply ``P(01) P(23)`` to a rank-four array."""

    return (
        block
        - block.swapaxes(0, 1)
        - block.swapaxes(2, 3)
        + block.swapaxes(0, 1).swapaxes(2, 3)
    )


def compute_two_rdm(c0: complex, c1: np.ndarray) -> np.ndarray:
    r"""Return the unnormalized CIS 2-RDM ``Gamma[p, q, r, s]``.

    ``Gamma[p, q, r, s]`` represents
    :math:`\Gamma^{rs}_{pq}=\langle\Psi|a_p^\dagger a_q^\dagger
    a_s a_r|\Psi\rangle`.
    """

    c1 = np.asarray(c1)
    nvir, nocc = c1.shape
    norb = nocc + nvir
    o, v = slice(0, nocc), slice(nocc, norb)
    dtype = np.result_type(c0, c1, np.complex128)
    gamma2 = np.zeros((norb, norb, norb, norb), dtype=dtype)
    eye = np.eye(nocc, dtype=dtype)

    reference = np.einsum("ik,jl->ijkl", eye, eye, optimize=True)
    reference -= np.einsum("il,jk->ijkl", eye, eye, optimize=True)
    oooo = compute_norm(c0, c1) * reference

    hole_density = np.einsum(
        "ej,el->jl", c1, c1.conj(), optimize=True
    )
    base = np.einsum("ik,jl->ijkl", eye, hole_density, optimize=True)
    oooo -= _antisymmetrize_pairs(base)
    gamma2[o, o, o, o] = oooo

    ooov = np.conjugate(c0) * np.einsum(
        "ik,cj->ijkc", eye, c1, optimize=True
    )
    ooov -= ooov.swapaxes(0, 1)
    gamma2[o, o, o, v] = ooov
    gamma2[o, o, v, o] = -ooov.swapaxes(2, 3)

    ovoo = c0 * np.einsum(
        "ik,al->iakl", eye, c1.conj(), optimize=True
    )
    ovoo -= ovoo.swapaxes(2, 3)
    gamma2[o, v, o, o] = ovoo
    gamma2[v, o, o, o] = -ovoo.swapaxes(0, 1)

    ovov = np.einsum(
        "cm,am,ik->iakc", c1, c1.conj(), eye, optimize=True
    )
    ovov -= np.einsum("ci,ak->iakc", c1, c1.conj(), optimize=True)
    gamma2[o, v, o, v] = ovov
    gamma2[v, o, o, v] = -ovov.swapaxes(0, 1)
    gamma2[o, v, v, o] = -ovov.swapaxes(2, 3)
    gamma2[v, o, v, o] = ovov.swapaxes(0, 1).swapaxes(2, 3)

    return gamma2
