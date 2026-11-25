"""One-body reduced density matrix for spin-orbital CIS."""

from __future__ import annotations

import numpy as np

from .amplitudes import compute_norm


def compute_one_rdm(c0: complex, c1: np.ndarray) -> np.ndarray:
    r"""Return the unnormalized CIS 1-RDM ``gamma[p, q]``.

    The array element ``gamma[p, q]`` represents
    :math:`\gamma^q_p = \langle\Psi|a_p^\dagger a_q|\Psi\rangle`.
    """

    c1 = np.asarray(c1)
    nvir, nocc = c1.shape
    norb = nocc + nvir
    o, v = slice(0, nocc), slice(nocc, norb)
    dtype = np.result_type(c0, c1, np.complex128)
    gamma = np.zeros((norb, norb), dtype=dtype)
    eye = np.eye(nocc, dtype=dtype)

    gamma[o, o] = compute_norm(c0, c1) * eye
    gamma[o, o] -= np.einsum(
        "ci,cj->ij", c1, c1.conj(), optimize=True
    )
    gamma[o, v] = np.conjugate(c0) * c1.T
    gamma[v, o] = c0 * c1.conj()
    gamma[v, v] = np.einsum(
        "bk,ak->ab", c1, c1.conj(), optimize=True
    )
    return gamma
