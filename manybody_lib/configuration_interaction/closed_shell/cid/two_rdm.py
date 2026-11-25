"""Spin-summed two-body RDM for closed-shell CID."""

from __future__ import annotations

import numpy as np

from .amplitudes import compute_norm, resolve_left


def _simultaneous_exchange(block: np.ndarray) -> np.ndarray:
    return block + block.swapaxes(0, 1).swapaxes(2, 3)


def compute_two_rdm(
    c0: complex,
    c2: np.ndarray,
    left_c0: complex | None = None,
    left_c2: np.ndarray | None = None,
) -> np.ndarray:
    r"""Return the unnormalized spin-summed HJO CID 2-RDM."""

    c2 = np.asarray(c2)
    left_c0, left_c2 = resolve_left(c0, c2, left_c0, left_c2)
    nvir, _, nocc, _ = c2.shape
    norb = nocc + nvir
    o, v = slice(0, nocc), slice(nocc, norb)
    dtype = np.result_type(c0, c2, left_c0, left_c2, np.complex128)
    gamma2 = np.zeros((norb, norb, norb, norb), dtype=dtype)
    eye = np.eye(nocc, dtype=dtype)
    norm = compute_norm(c0, c2, left_c0, left_c2)

    oooo = 4.0 * norm * np.einsum(
        "ik,jl->ijkl", eye, eye, optimize=True
    )
    oooo -= 2.0 * norm * np.einsum(
        "il,jk->ijkl", eye, eye, optimize=True
    )
    oooo += np.einsum(
        "abij,klab->ijkl", c2, left_c2, optimize=True
    )
    base = np.einsum(
        "il,abmj,mkab->ijkl", eye, c2, left_c2, optimize=True
    )
    base -= 2.0 * np.einsum(
        "ik,abmj,mlab->ijkl", eye, c2, left_c2, optimize=True
    )
    gamma2[o, o, o, o] = oooo + _simultaneous_exchange(base)

    gamma2[o, o, v, v] = (
        -2.0 * left_c0 * c2.swapaxes(2, 3).transpose(2, 3, 0, 1)
        + 4.0 * left_c0 * c2.transpose(2, 3, 0, 1)
    )
    gamma2[v, v, o, o] = c0 * left_c2.transpose(2, 3, 0, 1)

    vovo = 2.0 * np.einsum(
        "ij,bckl,klac->aibj", eye, c2, left_c2, optimize=True
    )
    vovo -= np.einsum(
        "bcik,kjca->aibj", c2, left_c2, optimize=True
    )
    vovo -= np.einsum(
        "bcki,kjac->aibj", c2, left_c2, optimize=True
    )
    gamma2[v, o, v, o] = vovo
    gamma2[o, v, o, v] = vovo.swapaxes(0, 1).swapaxes(2, 3)

    ovvo = -np.einsum(
        "ij,bckl,klac->iabj", eye, c2, left_c2, optimize=True
    )
    ovvo -= np.einsum(
        "bcki,kjca->iabj", c2, left_c2, optimize=True
    )
    ovvo += 2.0 * np.einsum(
        "bcik,kjca->iabj", c2, left_c2, optimize=True
    )
    gamma2[o, v, v, o] = ovvo
    gamma2[v, o, o, v] = ovvo.swapaxes(0, 1).swapaxes(2, 3)

    gamma2[v, v, v, v] = np.einsum(
        "cdij,ijab->abcd", c2, left_c2, optimize=True
    )
    return gamma2
