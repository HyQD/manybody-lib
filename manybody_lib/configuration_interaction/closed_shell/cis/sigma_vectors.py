"""Right sigma vector for closed-shell CIS in the HJO basis."""

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
    r"""Evaluate the closed-shell CIS right sigma vector.

    ``u[p, q, r, s]`` contains non-antisymmetrized physicists' spatial
    integrals and ``c1[a, i]`` multiplies the spin-free generator
    :math:`E_{ai}`.
    """

    f = np.asarray(f)
    u = np.asarray(u)
    c1 = np.asarray(c1)
    o, v = occupied, virtual
    dtype = np.result_type(e_ref, f, u, c0, c1, np.complex128)

    sigma0 = e_ref * c0 + 2.0 * np.einsum(
        "ia,ai->", f[o, v], c1, optimize=True
    )

    sigma1 = np.asarray(e_ref * c1 + c0 * f[v, o], dtype=dtype)
    sigma1 += np.einsum("ab,bi->ai", f[v, v], c1, optimize=True)
    sigma1 -= np.einsum("ji,aj->ai", f[o, o], c1, optimize=True)
    sigma1 += 2.0 * np.einsum(
        "bj,ajib->ai", c1, u[v, o, o, v], optimize=True
    )
    sigma1 -= np.einsum(
        "bj,ajbi->ai", c1, u[v, o, v, o], optimize=True
    )

    return np.asarray(sigma0, dtype=dtype)[()], sigma1
