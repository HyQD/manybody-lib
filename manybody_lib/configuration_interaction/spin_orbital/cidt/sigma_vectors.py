"""Right sigma vector for spin-orbital CIDT."""

from __future__ import annotations

import numpy as np

from ..cisdt._engine import ContractionEvaluator, result_dtype
from ._specs import SIGMA_SPECS
from .amplitudes import validate_amplitudes


def compute_sigma(
    e_ref: complex,
    f: np.ndarray,
    u: np.ndarray,
    c0: complex,
    c2: np.ndarray,
    c3: np.ndarray,
    occupied: slice,
    virtual: slice,
) -> tuple[np.complex128, np.ndarray, np.ndarray]:
    r"""Evaluate the spin-orbital CIDT sigma components ``(0, 2, 3)``."""

    if not isinstance(occupied, slice) or not isinstance(virtual, slice):
        raise TypeError("occupied and virtual must be slices")
    f = np.asarray(f)
    u = np.asarray(u)
    c2, c3, nocc, nvir = validate_amplitudes(c2, c3)
    if f.ndim != 2 or f.shape[0] != f.shape[1]:
        raise ValueError("f must be a square rank-two array")
    if u.shape != (f.shape[0],) * 4:
        raise ValueError("u must have shape (norb, norb, norb, norb)")
    if f[occupied, occupied].shape != (nocc, nocc):
        raise ValueError("occupied slice is inconsistent with the amplitudes")
    if f[virtual, virtual].shape != (nvir, nvir):
        raise ValueError("virtual slice is inconsistent with the amplitudes")

    dtype = result_dtype(e_ref, f, u, c0, c2, c3)
    data = {
        "e_ref": e_ref,
        "f": f,
        "u": u,
        "c0": c0,
        "c2": c2,
        "c3": c3,
        "nocc": nocc,
        "nvir": nvir,
    }
    evaluator = ContractionEvaluator(
        data,
        dtype=dtype,
        occupied=occupied,
        virtual=virtual,
    )
    values = tuple(
        evaluator.evaluate(SIGMA_SPECS[rank]["terms"], SIGMA_SPECS[rank]["spaces"])
        for rank in (0, 2, 3)
    )
    return np.asarray(values[0], dtype=dtype)[()], values[1], values[2]
