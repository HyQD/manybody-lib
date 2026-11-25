"""Spin-summed one-body RDM for closed-shell CID."""

from __future__ import annotations

import numpy as np

from .amplitudes import resolve_left


def compute_one_rdm(
    c0: complex,
    c2: np.ndarray,
    left_c0: complex | None = None,
    left_c2: np.ndarray | None = None,
) -> np.ndarray:
    r"""Return the unnormalized spin-summed HJO CID 1-RDM."""

    c2 = np.asarray(c2)
    left_c0, left_c2 = resolve_left(c0, c2, left_c0, left_c2)
    nvir, _, nocc, _ = c2.shape
    norb = nocc + nvir
    o, v = slice(0, nocc), slice(nocc, norb)
    dtype = np.result_type(c0, c2, left_c0, left_c2, np.complex128)
    gamma = np.zeros((norb, norb), dtype=dtype)
    eye = np.eye(nocc, dtype=dtype)

    doubles_overlap = np.einsum(
        "klab,abkl->", left_c2, c2, optimize=True
    )
    gamma[o, o] = (2.0 * left_c0 * c0 + doubles_overlap) * eye
    gamma[o, o] -= np.einsum(
        "abik,kjba->ij", c2, left_c2, optimize=True
    )
    gamma[v, v] = np.einsum(
        "bcij,ijac->ab", c2, left_c2, optimize=True
    )
    return gamma
