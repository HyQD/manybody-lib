"""One-body reduced density matrix for spin-orbital CID."""

from __future__ import annotations

import numpy as np

from .amplitudes import compute_norm


def compute_one_rdm(c0: complex, c2: np.ndarray) -> np.ndarray:
    r"""Return the unnormalized CID 1-RDM ``gamma[p, q]``."""

    c2 = np.asarray(c2)
    nvir, _, nocc, _ = c2.shape
    norb = nocc + nvir
    o, v = slice(0, nocc), slice(nocc, norb)
    dtype = np.result_type(c0, c2, np.complex128)
    gamma = np.zeros((norb, norb), dtype=dtype)
    eye = np.eye(nocc, dtype=dtype)

    gamma[o, o] = compute_norm(c0, c2) * eye
    gamma[o, o] -= 0.5 * np.einsum(
        "cdjk,cdik->ij", c2.conj(), c2, optimize=True
    )
    gamma[v, v] = 0.5 * np.einsum(
        "aclk,bclk->ab", c2.conj(), c2, optimize=True
    )
    return gamma
