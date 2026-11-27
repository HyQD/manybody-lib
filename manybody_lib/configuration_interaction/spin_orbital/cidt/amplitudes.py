"""Amplitude helpers for spin-orbital CIDT."""

from __future__ import annotations

import numpy as np

from ..cisdt._engine import ContractionEvaluator, result_dtype
from ..cisdt.amplitudes import antisymmetrize_doubles, antisymmetrize_triples
from ._specs import NORM_TERMS


def validate_amplitudes(
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


def amplitude_data(
    c0: complex,
    c2: np.ndarray,
    c3: np.ndarray,
) -> tuple[dict[str, object], np.dtype]:
    """Build the validated data mapping shared by norm and RDM kernels."""

    c2, c3, nocc, nvir = validate_amplitudes(c2, c3)
    dtype = result_dtype(c0, c2, c3)
    data = {
        "c0": c0,
        "c0_conj": np.conjugate(c0),
        "c2": c2,
        "c3": c3,
        "nocc": nocc,
        "nvir": nvir,
    }
    return data, dtype


def compute_norm(c0: complex, c2: np.ndarray, c3: np.ndarray) -> np.complex128:
    """Return ``<Psi|Psi>`` without normalizing the CIDT state."""

    data, dtype = amplitude_data(c0, c2, c3)
    value = ContractionEvaluator(data, dtype=dtype).evaluate(NORM_TERMS, "")
    return np.asarray(value, dtype=dtype)[()]


__all__ = [
    "antisymmetrize_doubles",
    "antisymmetrize_triples",
    "compute_norm",
]
