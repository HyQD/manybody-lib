"""Spin-summed two-body RDM for closed-shell CIS."""

from __future__ import annotations

import numpy as np

from .amplitudes import compute_norm, resolve_left


def _simultaneous_exchange(block: np.ndarray) -> np.ndarray:
    """Apply the unnormalized HJO column symmetrizer."""

    return block + block.swapaxes(0, 1).swapaxes(2, 3)


def compute_two_rdm(
    c0: complex,
    c1: np.ndarray,
    left_c0: complex | None = None,
    left_c1: np.ndarray | None = None,
) -> np.ndarray:
    r"""Return the unnormalized spin-summed HJO CIS 2-RDM.

    ``Gamma[p, q, r, s]`` represents
    :math:`\langle\widetilde\Psi|e_{pqrs}|\Psi\rangle`.
    """

    c1 = np.asarray(c1)
    left_c0, left_c1 = resolve_left(c0, c1, left_c0, left_c1)
    nvir, nocc = c1.shape
    norb = nocc + nvir
    o, v = slice(0, nocc), slice(nocc, norb)
    dtype = np.result_type(c0, c1, left_c0, left_c1, np.complex128)
    gamma2 = np.zeros((norb, norb, norb, norb), dtype=dtype)
    eye = np.eye(nocc, dtype=dtype)
    norm = compute_norm(c0, c1, left_c0, left_c1)

    oooo = 4.0 * norm * np.einsum(
        "ik,jl->ijkl", eye, eye, optimize=True
    )
    oooo -= 2.0 * norm * np.einsum(
        "il,jk->ijkl", eye, eye, optimize=True
    )
    base = np.einsum(
        "il,aj,ka->ijkl", eye, c1, left_c1, optimize=True
    )
    base -= 2.0 * np.einsum(
        "ik,aj,la->ijkl", eye, c1, left_c1, optimize=True
    )
    gamma2[o, o, o, o] = oooo + _simultaneous_exchange(base)

    oovo = -2.0 * left_c0 * np.einsum(
        "ik,aj->ijak", eye, c1, optimize=True
    )
    oovo += 4.0 * left_c0 * np.einsum(
        "jk,ai->ijak", eye, c1, optimize=True
    )
    gamma2[o, o, v, o] = oovo
    gamma2[o, o, o, v] = oovo.swapaxes(0, 1).swapaxes(2, 3)

    vooo = -c0 * np.einsum(
        "ij,ka->aijk", eye, left_c1, optimize=True
    )
    vooo += 2.0 * c0 * np.einsum(
        "ik,ja->aijk", eye, left_c1, optimize=True
    )
    gamma2[v, o, o, o] = vooo
    gamma2[o, v, o, o] = vooo.swapaxes(0, 1).swapaxes(2, 3)

    vovo = -np.einsum(
        "bi,ja->aibj", c1, left_c1, optimize=True
    )
    vovo += 2.0 * np.einsum(
        "ij,bk,ka->aibj", eye, c1, left_c1, optimize=True
    )
    gamma2[v, o, v, o] = vovo
    gamma2[o, v, o, v] = vovo.swapaxes(0, 1).swapaxes(2, 3)

    ovvo = 2.0 * np.einsum(
        "bi,ja->iabj", c1, left_c1, optimize=True
    )
    ovvo -= np.einsum(
        "ij,bk,ka->iabj", eye, c1, left_c1, optimize=True
    )
    gamma2[o, v, v, o] = ovvo
    gamma2[v, o, o, v] = ovvo.swapaxes(0, 1).swapaxes(2, 3)
    return gamma2
