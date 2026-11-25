"""Right sigma vector for spin-orbital CIS."""

from __future__ import annotations

import numpy as np


def compute_sigma(
    e_ref: complex,
    f: np.ndarray,
    u: np.ndarray,
    c0: complex,
    c1: np.ndarray,
    occupied: slice,
    virtual: slice,
) -> tuple[np.complex128, np.ndarray]:
    r"""Evaluate the spin-orbital CIS right sigma vector.

    The arrays use ``c1[a, i]``, ``f[p, q]``, and antisymmetrized
    physicists' integrals ``u[p, q, r, s] = <pq||rs>``.  No complex
    conjugation is applied to right amplitudes.
    """

    f = np.asarray(f)
    u = np.asarray(u)
    c1 = np.asarray(c1)
    o, v = occupied, virtual
    dtype = np.result_type(e_ref, f, u, c0, c1, np.complex128)

    sigma0 = e_ref * c0 + np.einsum(
        "jb,bj->", f[o, v], c1, optimize=True
    )

    sigma1 = np.asarray(e_ref * c1 + c0 * f[v, o], dtype=dtype)
    sigma1 += np.einsum("ab,bi->ai", f[v, v], c1, optimize=True)
    sigma1 -= np.einsum("ji,aj->ai", f[o, o], c1, optimize=True)
    sigma1 += np.einsum(
        "ajib,bj->ai", u[v, o, o, v], c1, optimize=True
    )

    return np.asarray(sigma0, dtype=dtype)[()], sigma1
