"""Amplitude helpers for closed-shell Matthews CIDT."""

from __future__ import annotations

import numpy as np

from ..cisdt._engine import ContractionEvaluator, result_dtype
from ..cisdt.amplitudes import (
    project_physical_triples,
    symmetrize_triples_columns,
    triples_null_component,
)
from ._specs import NORM_TERMS


def validate_right_amplitudes(
    c2: np.ndarray,
    c3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Validate and return doubles/triples and their orbital dimensions."""

    c2 = np.asarray(c2)
    c3 = np.asarray(c3)
    if c2.ndim != 4:
        raise ValueError("c2 must have shape (nvir, nvir, nocc, nocc)")
    nvir, second_nvir, nocc, second_nocc = c2.shape
    if second_nvir != nvir or second_nocc != nocc:
        raise ValueError("c2 must have shape (nvir, nvir, nocc, nocc)")
    if c3.shape != (nvir, nvir, nvir, nocc, nocc, nocc):
        raise ValueError(
            "c3 must have shape (nvir, nvir, nvir, nocc, nocc, nocc)"
        )
    return c2, c3, nocc, nvir


def compute_hermitian_left(
    c0: complex,
    c2: np.ndarray,
    c3: np.ndarray,
) -> tuple[np.complex128, np.ndarray, np.ndarray]:
    """Construct the literal Hermitian-adjoint Matthews CIDT left state."""

    c2, c3, _nocc, _nvir = validate_right_amplitudes(c2, c3)
    dtype = result_dtype(c0, c2, c3)
    left_c0 = np.asarray(np.conjugate(c0), dtype=dtype)[()]
    left_c2 = np.asarray(c2.transpose(2, 3, 0, 1).conj(), dtype=dtype)
    left_c3 = np.asarray(c3.transpose(3, 4, 5, 0, 1, 2).conj(), dtype=dtype)
    return left_c0, left_c2, left_c3


def resolve_left(
    c0: complex,
    c2: np.ndarray,
    c3: np.ndarray,
    left_c0: complex | None,
    left_c2: np.ndarray | None,
    left_c3: np.ndarray | None,
) -> tuple[np.complex128, np.ndarray, np.ndarray]:
    """Resolve and validate optional independent Matthews left amplitudes."""

    c2, c3, nocc, nvir = validate_right_amplitudes(c2, c3)
    supplied = tuple(value is not None for value in (left_c0, left_c2, left_c3))
    if not any(supplied):
        return compute_hermitian_left(c0, c2, c3)
    if not all(supplied):
        raise ValueError("left_c0, left_c2, and left_c3 must be supplied together")
    resolved_c2 = np.asarray(left_c2)
    resolved_c3 = np.asarray(left_c3)
    if resolved_c2.shape != (nocc, nocc, nvir, nvir):
        raise ValueError("left_c2 must have shape (nocc, nocc, nvir, nvir)")
    if resolved_c3.shape != (nocc, nocc, nocc, nvir, nvir, nvir):
        raise ValueError(
            "left_c3 must have shape (nocc, nocc, nocc, nvir, nvir, nvir)"
        )
    return np.asarray(left_c0)[()], resolved_c2, resolved_c3


def amplitude_data(
    c0: complex,
    c2: np.ndarray,
    c3: np.ndarray,
    left_c0: complex | None = None,
    left_c2: np.ndarray | None = None,
    left_c3: np.ndarray | None = None,
) -> tuple[dict[str, object], np.dtype]:
    """Build the validated data mapping shared by norm and RDM kernels."""

    c2, c3, nocc, nvir = validate_right_amplitudes(c2, c3)
    left_c0, left_c2, left_c3 = resolve_left(
        c0,
        c2,
        c3,
        left_c0,
        left_c2,
        left_c3,
    )
    dtype = result_dtype(c0, c2, c3, left_c0, left_c2, left_c3)
    data = {
        "c0": c0,
        "c2": c2,
        "c3": c3,
        "left_c0": left_c0,
        "left_c2": left_c2,
        "left_c3": left_c3,
        "nocc": nocc,
        "nvir": nvir,
    }
    return data, dtype


def compute_norm(
    c0: complex,
    c2: np.ndarray,
    c3: np.ndarray,
    left_c0: complex | None = None,
    left_c2: np.ndarray | None = None,
    left_c3: np.ndarray | None = None,
) -> np.complex128:
    """Return the unnormalized Matthews CIDT overlap."""

    data, dtype = amplitude_data(c0, c2, c3, left_c0, left_c2, left_c3)
    value = ContractionEvaluator(data, dtype=dtype).evaluate(NORM_TERMS, "")
    return np.asarray(value, dtype=dtype)[()]


__all__ = [
    "compute_hermitian_left",
    "compute_norm",
    "project_physical_triples",
    "symmetrize_triples_columns",
    "triples_null_component",
]
