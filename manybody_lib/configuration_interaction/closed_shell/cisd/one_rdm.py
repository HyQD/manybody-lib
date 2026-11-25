"""Spin-summed one-body RDM for closed-shell CISD."""

from __future__ import annotations

import numpy as np

from ..cid.one_rdm import compute_one_rdm as compute_cid_one_rdm
from ..cis.one_rdm import compute_one_rdm as compute_cis_one_rdm
from .amplitudes import resolve_left


def compute_one_rdm(
    c0: complex,
    c1: np.ndarray,
    c2: np.ndarray,
    left_c0: complex | None = None,
    left_c1: np.ndarray | None = None,
    left_c2: np.ndarray | None = None,
) -> np.ndarray:
    r"""Return the unnormalized spin-summed HJO CISD 1-RDM."""

    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    left_c0, left_c1, left_c2 = resolve_left(
        c0, c1, c2, left_c0, left_c1, left_c2
    )
    nvir, nocc = c1.shape
    if c2.shape != (nvir, nvir, nocc, nocc):
        raise ValueError("c1 and c2 dimensions are inconsistent")
    norb = nocc + nvir
    o, v = slice(0, nocc), slice(nocc, norb)
    dtype = np.result_type(c0, c1, c2, left_c0, left_c1, left_c2, np.complex128)

    gamma = compute_cis_one_rdm(c0, c1, left_c0, left_c1)
    gamma += compute_cid_one_rdm(c0, c2, left_c0, left_c2)
    gamma[o, o] -= 2.0 * left_c0 * c0 * np.eye(nocc, dtype=dtype)

    gamma[o, v] -= np.einsum(
        "jb,abji->ia", left_c1, c2, optimize=True
    )
    gamma[o, v] += 2.0 * np.einsum(
        "jb,abij->ia", left_c1, c2, optimize=True
    )
    gamma[v, o] += np.einsum(
        "bj,ijab->ai", c1, left_c2, optimize=True
    )
    return gamma
