"""Spin-summed two-body RDM for closed-shell CISD."""

from __future__ import annotations

import numpy as np

from ..cid.two_rdm import compute_two_rdm as compute_cid_two_rdm
from ..cis.two_rdm import compute_two_rdm as compute_cis_two_rdm
from .amplitudes import resolve_left


def compute_two_rdm(
    c0: complex,
    c1: np.ndarray,
    c2: np.ndarray,
    left_c0: complex | None = None,
    left_c1: np.ndarray | None = None,
    left_c2: np.ndarray | None = None,
) -> np.ndarray:
    r"""Return the unnormalized spin-summed HJO CISD 2-RDM."""

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
    eye = np.eye(nocc, dtype=dtype)

    gamma2 = compute_cis_two_rdm(c0, c1, left_c0, left_c1)
    gamma2 += compute_cid_two_rdm(c0, c2, left_c0, left_c2)
    reference = 4.0 * np.einsum("ik,jl->ijkl", eye, eye, optimize=True)
    reference -= 2.0 * np.einsum("il,jk->ijkl", eye, eye, optimize=True)
    gamma2[o, o, o, o] -= left_c0 * c0 * reference

    oovo = np.einsum(
        "ik,lb,ablj->ijak", eye, left_c1, c2, optimize=True
    )
    oovo -= 2.0 * np.einsum(
        "ik,lb,abjl->ijak", eye, left_c1, c2, optimize=True
    )
    oovo -= 2.0 * np.einsum(
        "jk,lb,abli->ijak", eye, left_c1, c2, optimize=True
    )
    oovo += 4.0 * np.einsum(
        "jk,lb,abil->ijak", eye, left_c1, c2, optimize=True
    )
    oovo += np.einsum("kb,abji->ijak", left_c1, c2, optimize=True)
    oovo -= 2.0 * np.einsum("kb,abij->ijak", left_c1, c2, optimize=True)
    gamma2[o, o, v, o] += oovo
    gamma2[o, o, o, v] += oovo.swapaxes(0, 1).swapaxes(2, 3)

    vooo = -np.einsum(
        "ij,bl,lkba->aijk", eye, c1, left_c2, optimize=True
    )
    vooo += 2.0 * np.einsum(
        "ik,bl,ljba->aijk", eye, c1, left_c2, optimize=True
    )
    vooo -= np.einsum("bi,jkab->aijk", c1, left_c2, optimize=True)
    gamma2[v, o, o, o] += vooo
    gamma2[o, v, o, o] += vooo.swapaxes(0, 1).swapaxes(2, 3)

    vovv = -np.einsum(
        "ja,bcij->aibc", left_c1, c2, optimize=True
    )
    vovv += 2.0 * np.einsum(
        "ja,bcji->aibc", left_c1, c2, optimize=True
    )
    gamma2[v, o, v, v] += vovv
    gamma2[o, v, v, v] += vovv.swapaxes(0, 1).swapaxes(2, 3)

    vvvo = np.einsum("cj,ijba->abci", c1, left_c2, optimize=True)
    gamma2[v, v, v, o] += vvvo
    gamma2[v, v, o, v] += vvvo.swapaxes(0, 1).swapaxes(2, 3)
    return gamma2
