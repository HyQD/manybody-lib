"""Right sigma vector for closed-shell CISD in the HJO basis."""

from __future__ import annotations

import numpy as np

from ..cid.sigma_vectors import compute_sigma as compute_cid_sigma
from ..cis.sigma_vectors import compute_sigma as compute_cis_sigma


def compute_sigma(
    e_ref: complex,
    f: np.ndarray,
    u: np.ndarray,
    c0: complex,
    c1: np.ndarray,
    c2: np.ndarray,
    occupied: slice,
    virtual: slice,
) -> tuple[np.complex128, np.ndarray, np.ndarray]:
    r"""Evaluate the closed-shell HJO CISD right sigma vector.

    ``u`` contains non-antisymmetrized physicists' spatial integrals and
    ``c2[a,b,i,j] == c2[b,a,j,i]``.
    """

    f = np.asarray(f)
    u = np.asarray(u)
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    o, v = occupied, virtual

    cis0, sigma1 = compute_cis_sigma(e_ref, f, u, c0, c1, o, v)
    cid0, sigma2 = compute_cid_sigma(e_ref, f, u, c0, c2, o, v)
    sigma0 = cis0 + cid0 - e_ref * c0

    sigma1 += 2.0 * np.einsum(
        "bcij,ajbc->ai", c2, u[v, o, v, v], optimize=True
    )
    sigma1 -= np.einsum(
        "bcij,ajcb->ai", c2, u[v, o, v, v], optimize=True
    )
    sigma1 -= 2.0 * np.einsum(
        "abjk,jkib->ai", c2, u[o, o, o, v], optimize=True
    )
    sigma1 += np.einsum(
        "abjk,jkbi->ai", c2, u[o, o, v, o], optimize=True
    )
    sigma1 -= np.einsum("jb,abji->ai", f[o, v], c2, optimize=True)
    sigma1 += 2.0 * np.einsum("jb,abij->ai", f[o, v], c2, optimize=True)

    term = np.einsum("ai,bj->abij", c1, f[v, o], optimize=True)
    term += np.einsum(
        "ci,abcj->abij", c1, u[v, v, v, o], optimize=True
    )
    term -= np.einsum(
        "ak,bkji->abij", c1, u[v, o, o, o], optimize=True
    )
    sigma2 += term + term.swapaxes(0, 1).swapaxes(2, 3)

    dtype = np.result_type(e_ref, f, u, c0, c1, c2, np.complex128)
    return np.asarray(sigma0, dtype=dtype)[()], sigma1, sigma2
