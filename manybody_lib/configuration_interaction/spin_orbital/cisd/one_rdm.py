"""One-body reduced density matrix for spin-orbital CISD."""

from __future__ import annotations

import numpy as np

from .amplitudes import compute_norm


def compute_one_rdm(c0: complex, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    r"""Return the unnormalized CISD 1-RDM ``gamma[p, q]``."""

    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    nvir, nocc = c1.shape
    if c2.shape != (nvir, nvir, nocc, nocc):
        raise ValueError("c1 and c2 dimensions are inconsistent")
    norb = nocc + nvir
    o, v = slice(0, nocc), slice(nocc, norb)
    dtype = np.result_type(c0, c1, c2, np.complex128)
    gamma = np.zeros((norb, norb), dtype=dtype)
    eye = np.eye(nocc, dtype=dtype)

    gamma[o, o] = compute_norm(c0, c1, c2) * eye
    gamma[o, o] -= np.einsum(
        "ci,cj->ij", c1, c1.conj(), optimize=True
    )
    gamma[o, o] -= 0.5 * np.einsum(
        "cdjk,cdik->ij", c2.conj(), c2, optimize=True
    )

    gamma[o, v] = np.conjugate(c0) * c1.T
    gamma[o, v] += np.einsum(
        "ck,acik->ia", c1.conj(), c2, optimize=True
    )
    gamma[v, o] = c0 * c1.conj()
    gamma[v, o] += np.einsum(
        "ck,acik->ai", c1, c2.conj(), optimize=True
    )

    gamma[v, v] = np.einsum(
        "bk,ak->ab", c1, c1.conj(), optimize=True
    )
    gamma[v, v] += 0.5 * np.einsum(
        "aclk,bclk->ab", c2.conj(), c2, optimize=True
    )
    return gamma
