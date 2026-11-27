"""Amplitude helpers for spin-orbital CISDT."""

from __future__ import annotations

import itertools

import numpy as np

from ._contractions import NORM_TERMS
from ._engine import ContractionEvaluator, result_dtype


_S3 = tuple(itertools.permutations(range(3)))


def _permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def antisymmetrize_doubles(c2: np.ndarray) -> np.ndarray:
    """Project a rank-four array onto separate virtual/occupied antisymmetry."""

    c2 = np.asarray(c2)
    if c2.ndim != 4:
        raise ValueError("c2 must be a rank-four array")
    return 0.25 * (
        c2
        - c2.swapaxes(0, 1)
        - c2.swapaxes(2, 3)
        + c2.swapaxes(0, 1).swapaxes(2, 3)
    )


def antisymmetrize_triples(c3: np.ndarray) -> np.ndarray:
    """Project a rank-six array onto separate virtual/occupied antisymmetry."""

    c3 = np.asarray(c3)
    if c3.ndim != 6:
        raise ValueError("c3 must be a rank-six array")
    result = np.zeros_like(c3, dtype=result_dtype(c3))
    for virtual_permutation in _S3:
        virtual_sign = _permutation_sign(virtual_permutation)
        for occupied_permutation in _S3:
            axes = virtual_permutation + tuple(
                3 + position for position in occupied_permutation
            )
            sign = virtual_sign * _permutation_sign(occupied_permutation)
            result += sign * c3.transpose(axes)
    return result / 36.0


def validate_amplitudes(
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Validate and return amplitudes and their orbital dimensions."""

    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    c3 = np.asarray(c3)
    if c1.ndim != 2:
        raise ValueError("c1 must have shape (nvir, nocc)")
    nvir, nocc = c1.shape
    if c2.shape != (nvir, nvir, nocc, nocc):
        raise ValueError("c2 must have shape (nvir, nvir, nocc, nocc)")
    if c3.shape != (nvir, nvir, nvir, nocc, nocc, nocc):
        raise ValueError(
            "c3 must have shape (nvir, nvir, nvir, nocc, nocc, nocc)"
        )
    return c1, c2, c3, nocc, nvir


def amplitude_data(
    c0: complex,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
) -> tuple[dict[str, object], np.dtype]:
    """Build the validated data mapping shared by norm and RDM kernels."""

    c1, c2, c3, nocc, nvir = validate_amplitudes(c1, c2, c3)
    dtype = result_dtype(c0, c1, c2, c3)
    data = {
        "c0": c0,
        "c0_conj": np.conjugate(c0),
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "nocc": nocc,
        "nvir": nvir,
    }
    return data, dtype


def compute_norm(
    c0: complex,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
) -> np.complex128:
    """Return ``<Psi|Psi>`` without normalizing the CISDT state."""

    data, dtype = amplitude_data(c0, c1, c2, c3)
    value = ContractionEvaluator(data, dtype=dtype).evaluate(NORM_TERMS, "")
    return np.asarray(value, dtype=dtype)[()]
