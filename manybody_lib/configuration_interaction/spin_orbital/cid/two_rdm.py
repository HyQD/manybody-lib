"""Two-body reduced density matrix for spin-orbital CID."""

from __future__ import annotations

import numpy as np

from .amplitudes import compute_norm


def _antisymmetrize_pairs(block: np.ndarray) -> np.ndarray:
    return (
        block
        - block.swapaxes(0, 1)
        - block.swapaxes(2, 3)
        + block.swapaxes(0, 1).swapaxes(2, 3)
    )


def compute_two_rdm(c0: complex, c2: np.ndarray) -> np.ndarray:
    r"""Return the unnormalized CID 2-RDM ``Gamma[p, q, r, s]``."""

    c2 = np.asarray(c2)
    nvir, _, nocc, _ = c2.shape
    norb = nocc + nvir
    o, v = slice(0, nocc), slice(nocc, norb)
    dtype = np.result_type(c0, c2, np.complex128)
    gamma2 = np.zeros((norb, norb, norb, norb), dtype=dtype)
    eye = np.eye(nocc, dtype=dtype)

    reference = np.einsum("ik,jl->ijkl", eye, eye, optimize=True)
    reference -= np.einsum("il,jk->ijkl", eye, eye, optimize=True)
    oooo = compute_norm(c0, c2) * reference
    oooo += 0.5 * np.einsum(
        "efkl,efij->ijkl", c2.conj(), c2, optimize=True
    )
    base = 0.5 * np.einsum(
        "eflm,efim,jk->ijkl", c2.conj(), c2, eye, optimize=True
    )
    oooo += _antisymmetrize_pairs(base)
    gamma2[o, o, o, o] = oooo

    gamma2[o, o, v, v] = np.conjugate(c0) * c2.transpose(2, 3, 0, 1)
    gamma2[v, v, o, o] = c0 * c2.conj()

    ovov = 0.5 * np.einsum(
        "aemn,cemn,ik->iakc", c2.conj(), c2, eye, optimize=True
    )
    ovov -= np.einsum(
        "aekm,ceim->iakc", c2.conj(), c2, optimize=True
    )
    gamma2[o, v, o, v] = ovov
    gamma2[v, o, o, v] = -ovov.swapaxes(0, 1)
    gamma2[o, v, v, o] = -ovov.swapaxes(2, 3)
    gamma2[v, o, v, o] = ovov.swapaxes(0, 1).swapaxes(2, 3)

    gamma2[v, v, v, v] = 0.5 * np.einsum(
        "abmn,cdmn->abcd", c2.conj(), c2, optimize=True
    )
    return gamma2
