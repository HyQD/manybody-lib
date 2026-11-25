"""Spin-summed one-body RDM for closed-shell CIS."""

from __future__ import annotations

import numpy as np

from .amplitudes import compute_norm, resolve_left


def compute_one_rdm(
    c0: complex,
    c1: np.ndarray,
    left_c0: complex | None = None,
    left_c1: np.ndarray | None = None,
) -> np.ndarray:
    r"""Return the unnormalized spin-summed HJO CIS 1-RDM.

    Independent left amplitudes may be supplied as ``left_c1[i, a]``.
    If omitted, the Hermitian-bra HJO transformation is used.
    """

    c1 = np.asarray(c1)
    left_c0, left_c1 = resolve_left(c0, c1, left_c0, left_c1)
    nvir, nocc = c1.shape
    norb = nocc + nvir
    o, v = slice(0, nocc), slice(nocc, norb)
    dtype = np.result_type(c0, c1, left_c0, left_c1, np.complex128)
    gamma = np.zeros((norb, norb), dtype=dtype)
    eye = np.eye(nocc, dtype=dtype)

    norm = compute_norm(c0, c1, left_c0, left_c1)
    gamma[o, o] = 2.0 * norm * eye
    gamma[o, o] -= np.einsum(
        "ai,ja->ij", c1, left_c1, optimize=True
    )
    gamma[o, v] = 2.0 * left_c0 * c1.T
    gamma[v, o] = c0 * left_c1.T
    gamma[v, v] = np.einsum(
        "bi,ia->ab", c1, left_c1, optimize=True
    )
    return gamma
