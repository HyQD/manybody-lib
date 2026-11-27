"""Closed-shell Matthews CIDT sigma vector with exact contraction reuse."""

from __future__ import annotations

import numpy as np

from ..cisdt._engine import ContractionEvaluator, result_dtype
from ..cisdt._gemm import validate_gemm_options
from ..cisdt._schedule import compile_sigma_schedule
from ..cisdt.amplitudes import project_physical_triples
from ._specs import SIGMA_SPECS
from .amplitudes import validate_right_amplitudes


_SIGMA_SCHEDULES = {
    (symmetry, combine): {
        rank: compile_sigma_schedule(
            spec, assume_symmetry=symmetry, combine_integrals=combine
        )
        for rank, spec in SIGMA_SPECS.items()
    }
    for symmetry in (False, True)
    for combine in (False, True)
}


def compute_sigma(
    e_ref: complex,
    f: np.ndarray,
    u: np.ndarray,
    c0: complex,
    c2: np.ndarray,
    c3: np.ndarray,
    occupied: slice,
    virtual: slice,
    *,
    assume_symmetry: bool = False,
    combine_integrals: bool = False,
    optimize_accumulation: bool = False,
    use_gemm: bool = False,
    gemm_block_size: int | None = None,
) -> tuple[np.complex128, np.ndarray, np.ndarray]:
    """Evaluate the original CIDT equations with redundant contractions removed.

    The schedule is compiled from the singles-free CIDT table; no singles
    amplitudes or singles-dependent contractions are introduced.

    With ``assume_symmetry=True``, also exploit the simultaneous-column
    symmetries of ``c2`` and ``c3`` and the physicists' integral symmetry
    ``u[p,q,r,s] == u[q,p,s,r]``. The caller must ensure these equalities;
    they are not checked at runtime. Complex tensors and arbitrary triples
    null-space components are supported. The default makes no additional
    symmetry assumptions.

    With ``combine_integrals=True``, replace matching direct/exchange pairs
    by contractions with ``L[p,q,r,s] = 2*u[p,q,r,s] - u[p,q,s,r]``.
    Only needed blocks are formed and cached during this call. This option
    is algebraically valid even when ``assume_symmetry=False``.

    With ``optimize_accumulation=True``, fuse weighted triples permutations
    without full-size scaling temporaries and remove the null component in
    place on the new sigma tensor. Large complex128 outputs use lazily JIT-
    compiled Numba loops; small and unsupported arrays use NumPy. Numba's
    thread count is controlled independently of BLAS by NUMBA_NUM_THREADS.

    With ``use_gemm=True``, binary C3 contractions in the triples sigma use
    explicit matrix products. An optional positive ``gemm_block_size``
    bounds the number of free virtual indices per tile, streaming each tile
    to its permutation destinations rather than allocating a full output.
    With ``assume_symmetry=True``, equivalent amplitude labelings may expose
    contiguous matrix views. The default GEMM path is unblocked.
    """

    if not isinstance(occupied, slice) or not isinstance(virtual, slice):
        raise TypeError("occupied and virtual must be slices")
    gemm_block_size = validate_gemm_options(use_gemm, gemm_block_size)
    f = np.asarray(f)
    u = np.asarray(u)
    c2, c3, nocc, nvir = validate_right_amplitudes(c2, c3)
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
    schedules = _SIGMA_SCHEDULES[assume_symmetry, combine_integrals]
    sigma0 = evaluator.evaluate_sigma_schedule(schedules[0])
    sigma2 = evaluator.evaluate_sigma_schedule(schedules[2])
    sigma3 = evaluator.evaluate_sigma_schedule(
        schedules[3],
        optimize_accumulation=optimize_accumulation,
        use_gemm=use_gemm,
        gemm_block_size=gemm_block_size,
    )
    if optimize_accumulation:
        from ..cisdt._accumulation import project_triples_inplace

        project_triples_inplace(sigma3)
    else:
        sigma3 = project_physical_triples(sigma3)
    return np.asarray(sigma0, dtype=dtype)[()], sigma2, sigma3
