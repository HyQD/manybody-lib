"""Right sigma vector for spin-orbital CID."""

from __future__ import annotations

import numpy as np


def _antisymmetrize_pairs(block: np.ndarray) -> np.ndarray:
    return (
        block
        - block.swapaxes(0, 1)
        - block.swapaxes(2, 3)
        + block.swapaxes(0, 1).swapaxes(2, 3)
    )


def compute_sigma(
    e_ref: complex,
    f: np.ndarray,
    u: np.ndarray,
    c0: complex,
    c2: np.ndarray,
    occupied: slice,
    virtual: slice,
) -> tuple[np.complex128, np.ndarray]:
    r"""Evaluate the spin-orbital CID right sigma vector.

    ``c2[a, b, i, j]`` is separately antisymmetric in ``a,b`` and ``i,j``.
    ``u[p, q, r, s]`` contains antisymmetrized physicists' integrals.
    """

    f = np.asarray(f)
    u = np.asarray(u)
    c2 = np.asarray(c2)
    o, v = occupied, virtual
    dtype = np.result_type(e_ref, f, u, c0, c2, np.complex128)

    sigma0 = e_ref * c0 + 0.25 * np.einsum(
        "cdkl,klcd->", c2, u[o, o, v, v], optimize=True
    )

    sigma2 = np.asarray(e_ref * c2 + c0 * u[v, v, o, o], dtype=dtype)

    term = -np.einsum("kj,abik->abij", f[o, o], c2, optimize=True)
    sigma2 += term - term.swapaxes(2, 3)

    term = np.einsum("bc,acij->abij", f[v, v], c2, optimize=True)
    sigma2 += term - term.swapaxes(0, 1)

    sigma2 += 0.5 * np.einsum(
        "cdij,abcd->abij", c2, u[v, v, v, v], optimize=True
    )

    term = np.einsum(
        "acik,bkjc->abij", c2, u[v, o, o, v], optimize=True
    )
    sigma2 += _antisymmetrize_pairs(term)

    sigma2 += 0.5 * np.einsum(
        "abkl,klij->abij", c2, u[o, o, o, o], optimize=True
    )

    return np.asarray(sigma0, dtype=dtype)[()], sigma2
