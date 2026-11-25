"""Right sigma vector for spin-orbital CISD."""

from __future__ import annotations

import numpy as np

from ..cid.sigma_vectors import compute_sigma as compute_cid_sigma
from ..cis.sigma_vectors import compute_sigma as compute_cis_sigma


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
    c1: np.ndarray,
    c2: np.ndarray,
    occupied: slice,
    virtual: slice,
) -> tuple[np.complex128, np.ndarray, np.ndarray]:
    r"""Evaluate the spin-orbital CISD right sigma vector.

    ``c1[a, i]`` stores singles and ``c2[a, b, i, j]`` stores doubles
    separately antisymmetric in the virtual and occupied index pairs.
    ``u`` contains antisymmetrized physicists' integrals.
    """

    f = np.asarray(f)
    u = np.asarray(u)
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    o, v = occupied, virtual

    cis0, sigma1 = compute_cis_sigma(e_ref, f, u, c0, c1, o, v)
    cid0, sigma2 = compute_cid_sigma(e_ref, f, u, c0, c2, o, v)
    sigma0 = cis0 + cid0 - e_ref * c0

    sigma1 += np.einsum("kc,acik->ai", f[o, v], c2, optimize=True)
    sigma1 += 0.5 * np.einsum(
        "cdik,akcd->ai", c2, u[v, o, v, v], optimize=True
    )
    sigma1 -= 0.5 * np.einsum(
        "ackl,klic->ai", c2, u[o, o, o, v], optimize=True
    )

    term = np.einsum(
        "cj,abic->abij", c1, u[v, v, o, v], optimize=True
    )
    sigma2 += term - term.swapaxes(2, 3)

    term = np.einsum("ai,bj->abij", c1, f[v, o], optimize=True)
    sigma2 += _antisymmetrize_pairs(term)

    term = -np.einsum(
        "bk,akij->abij", c1, u[v, o, o, o], optimize=True
    )
    sigma2 += term - term.swapaxes(0, 1)

    dtype = np.result_type(e_ref, f, u, c0, c1, c2, np.complex128)
    return np.asarray(sigma0, dtype=dtype)[()], sigma1, sigma2
