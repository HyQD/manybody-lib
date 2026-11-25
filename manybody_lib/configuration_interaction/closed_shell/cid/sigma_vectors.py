"""Right sigma vector for closed-shell CID in the HJO basis."""

from __future__ import annotations

import numpy as np


def compute_sigma(
    e_ref: complex,
    f: np.ndarray,
    u: np.ndarray,
    c0: complex,
    c2: np.ndarray,
    occupied: slice,
    virtual: slice,
) -> tuple[np.complex128, np.ndarray]:
    r"""Evaluate the closed-shell HJO CID right sigma vector.

    ``c2[a, b, i, j]`` obeys the redundant-doubles column symmetry
    ``c2[a,b,i,j] == c2[b,a,j,i]``.  ``u`` is not antisymmetrized.
    """

    f = np.asarray(f)
    u = np.asarray(u)
    c2 = np.asarray(c2)
    o, v = occupied, virtual
    dtype = np.result_type(e_ref, f, u, c0, c2, np.complex128)

    l_oo_vv = 2.0 * u[o, o, v, v] - u[o, o, v, v].swapaxes(2, 3)
    sigma0 = e_ref * c0 + np.einsum(
        "abij,ijab->", c2, l_oo_vv, optimize=True
    )

    sigma2 = np.asarray(e_ref * c2 + c0 * u[v, v, o, o], dtype=dtype)
    sigma2 += np.einsum(
        "abkl,klij->abij", c2, u[o, o, o, o], optimize=True
    )
    sigma2 += np.einsum(
        "cdij,abcd->abij", c2, u[v, v, v, v], optimize=True
    )

    term = np.einsum("bc,acij->abij", f[v, v], c2, optimize=True)
    term -= np.einsum("kj,abik->abij", f[o, o], c2, optimize=True)
    term += 2.0 * np.einsum(
        "acik,bkjc->abij", c2, u[v, o, o, v], optimize=True
    )
    term -= np.einsum(
        "acik,bkcj->abij", c2, u[v, o, v, o], optimize=True
    )
    term -= np.einsum(
        "acki,bkjc->abij", c2, u[v, o, o, v], optimize=True
    )
    term -= np.einsum(
        "ackj,bkci->abij", c2, u[v, o, v, o], optimize=True
    )
    sigma2 += term + term.swapaxes(0, 1).swapaxes(2, 3)

    return np.asarray(sigma0, dtype=dtype)[()], sigma2
