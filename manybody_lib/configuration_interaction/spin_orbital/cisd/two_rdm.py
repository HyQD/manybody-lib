"""Two-body reduced density matrix for spin-orbital CISD."""

from __future__ import annotations

import numpy as np

from ..cid.two_rdm import compute_two_rdm as compute_cid_two_rdm
from ..cis.two_rdm import compute_two_rdm as compute_cis_two_rdm


def compute_two_rdm(c0: complex, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    r"""Return the unnormalized CISD 2-RDM ``Gamma[p, q, r, s]``."""

    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    nvir, nocc = c1.shape
    if c2.shape != (nvir, nvir, nocc, nocc):
        raise ValueError("c1 and c2 dimensions are inconsistent")
    norb = nocc + nvir
    o, v = slice(0, nocc), slice(nocc, norb)
    dtype = np.result_type(c0, c1, c2, np.complex128)
    eye = np.eye(nocc, dtype=dtype)

    gamma2 = compute_cis_two_rdm(c0, c1) + compute_cid_two_rdm(c0, c2)
    reference = np.einsum("ik,jl->ijkl", eye, eye, optimize=True)
    reference -= np.einsum("il,jk->ijkl", eye, eye, optimize=True)
    gamma2[o, o, o, o] -= np.conjugate(c0) * c0 * reference

    ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=dtype)
    base = np.einsum(
        "em,jk,ceim->ijkc", c1.conj(), eye, c2, optimize=True
    )
    ooov += -base + base.swapaxes(0, 1)
    ooov += np.einsum("ek,ceij->ijkc", c1.conj(), c2, optimize=True)
    gamma2[o, o, o, v] += ooov
    gamma2[o, o, v, o] -= ooov.swapaxes(2, 3)

    ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=dtype)
    base = np.einsum(
        "em,aelm,ik->iakl", c1, c2.conj(), eye, optimize=True
    )
    ovoo += base - base.swapaxes(2, 3)
    ovoo += np.einsum("ei,aekl->iakl", c1, c2.conj(), optimize=True)
    gamma2[o, v, o, o] += ovoo
    gamma2[v, o, o, o] -= ovoo.swapaxes(0, 1)

    ovvv = np.einsum("am,cdim->iacd", c1.conj(), c2, optimize=True)
    gamma2[o, v, v, v] += ovvv
    gamma2[v, o, v, v] -= ovvv.swapaxes(0, 1)

    vvov = np.einsum("cm,abkm->abkc", c1, c2.conj(), optimize=True)
    gamma2[v, v, o, v] += vvov
    gamma2[v, v, v, o] -= vvov.swapaxes(2, 3)
    return gamma2
