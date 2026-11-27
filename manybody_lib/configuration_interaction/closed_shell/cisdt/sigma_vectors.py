"""Right sigma vector for closed-shell Matthews CISDT."""

from __future__ import annotations

import numpy as np

from ._contractions import SIGMA_SPECS
from ._engine import ContractionEvaluator, result_dtype
from .amplitudes import project_physical_triples, validate_right_amplitudes


def compute_sigma(
    e_ref: complex,
    f: np.ndarray,
    u: np.ndarray,
    c0: complex,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    occupied: slice,
    virtual: slice,
) -> tuple[np.complex128, np.ndarray, np.ndarray, np.ndarray]:
    r"""Evaluate the metric-resolved closed-shell Matthews CISDT sigma vector.

    ``u[p, q, r, s]`` contains non-antisymmetrized physicists' spatial
    integrals.  Doubles and triples are symmetric under simultaneous
    permutations of their virtual/occupied columns.  The triples result is
    the Moore--Penrose representative with a zero wavefunction-null part.
    """

    if not isinstance(occupied, slice) or not isinstance(virtual, slice):
        raise TypeError("occupied and virtual must be slices")
    f = np.asarray(f)
    u = np.asarray(u)
    c1, c2, c3, nocc, nvir = validate_right_amplitudes(c1, c2, c3)
    if f.ndim != 2 or f.shape[0] != f.shape[1]:
        raise ValueError("f must be a square rank-two array")
    if u.shape != (f.shape[0],) * 4:
        raise ValueError("u must have shape (norb, norb, norb, norb)")
    if f[occupied, occupied].shape != (nocc, nocc):
        raise ValueError("occupied slice is inconsistent with the amplitudes")
    if f[virtual, virtual].shape != (nvir, nvir):
        raise ValueError("virtual slice is inconsistent with the amplitudes")

    dtype = result_dtype(e_ref, f, u, c0, c1, c2, c3)
    data = {
        "e_ref": e_ref,
        "f": f,
        "u": u,
        "c0": c0,
        "c1": c1,
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
    sigma0 = evaluator.evaluate_sigma(SIGMA_SPECS[0])
    sigma1 = evaluator.evaluate_sigma(SIGMA_SPECS[1])
    sigma2 = evaluator.evaluate_sigma(SIGMA_SPECS[2])
    sigma3 = evaluator.evaluate_sigma(SIGMA_SPECS[3])
    sigma3 = project_physical_triples(sigma3)
    return np.asarray(sigma0, dtype=dtype)[()], sigma1, sigma2, sigma3
