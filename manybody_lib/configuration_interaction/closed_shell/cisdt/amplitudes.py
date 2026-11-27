"""Amplitude helpers for closed-shell Matthews CISDT."""

from __future__ import annotations

import itertools

import numpy as np

from ._contractions import NORM_TERMS
from ._engine import ContractionEvaluator, result_dtype


_S3 = tuple(itertools.permutations(range(3)))


def validate_right_amplitudes(
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Validate and return right amplitudes and their orbital dimensions."""

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


def symmetrize_triples_columns(c3: np.ndarray) -> np.ndarray:
    r"""Average over simultaneous permutations of the three ``(a, i)`` columns."""

    c3 = np.asarray(c3)
    if c3.ndim != 6:
        raise ValueError("c3 must be a rank-six array")
    result = np.zeros_like(c3, dtype=result_dtype(c3))
    for permutation in _S3:
        inverse = tuple(np.argsort(permutation))
        axes = inverse + tuple(3 + position for position in inverse)
        result += c3.transpose(axes)
    return result / 6.0


def triples_null_component(c3: np.ndarray) -> np.ndarray:
    """Return the wavefunction-null trivial-S3 component of triples."""

    c3 = np.asarray(c3)
    if c3.ndim != 6:
        raise ValueError("c3 must be a rank-six array")
    result = np.zeros_like(c3, dtype=result_dtype(c3))
    for permutation in _S3:
        axes = (0, 1, 2) + tuple(3 + position for position in permutation)
        result += c3.transpose(axes)
    return result / 6.0


def project_physical_triples(c3: np.ndarray) -> np.ndarray:
    """Return the Matthews Moore--Penrose zero-null triples representative."""

    c3 = np.asarray(c3)
    return c3 - triples_null_component(c3)


def compute_hermitian_left(
    c0: complex,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
) -> tuple[np.complex128, np.ndarray, np.ndarray, np.ndarray]:
    r"""Construct the literal Hermitian-adjoint Matthews left state."""

    c1, c2, c3, _nocc, _nvir = validate_right_amplitudes(c1, c2, c3)
    dtype = result_dtype(c0, c1, c2, c3)
    left_c0 = np.asarray(np.conjugate(c0), dtype=dtype)[()]
    left_c1 = np.asarray(c1.T.conj(), dtype=dtype)
    left_c2 = np.asarray(c2.transpose(2, 3, 0, 1).conj(), dtype=dtype)
    left_c3 = np.asarray(c3.transpose(3, 4, 5, 0, 1, 2).conj(), dtype=dtype)
    return left_c0, left_c1, left_c2, left_c3


def resolve_left(
    c0: complex,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    left_c0: complex | None,
    left_c1: np.ndarray | None,
    left_c2: np.ndarray | None,
    left_c3: np.ndarray | None,
) -> tuple[np.complex128, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve and validate optional independent Matthews left amplitudes."""

    c1, c2, c3, nocc, nvir = validate_right_amplitudes(c1, c2, c3)
    supplied = tuple(
        value is not None
        for value in (left_c0, left_c1, left_c2, left_c3)
    )
    if not any(supplied):
        return compute_hermitian_left(c0, c1, c2, c3)
    if not all(supplied):
        raise ValueError(
            "left_c0, left_c1, left_c2, and left_c3 must be supplied together"
        )
    resolved_c1 = np.asarray(left_c1)
    resolved_c2 = np.asarray(left_c2)
    resolved_c3 = np.asarray(left_c3)
    if resolved_c1.shape != (nocc, nvir):
        raise ValueError("left_c1 must have shape (nocc, nvir)")
    if resolved_c2.shape != (nocc, nocc, nvir, nvir):
        raise ValueError("left_c2 must have shape (nocc, nocc, nvir, nvir)")
    if resolved_c3.shape != (nocc, nocc, nocc, nvir, nvir, nvir):
        raise ValueError(
            "left_c3 must have shape (nocc, nocc, nocc, nvir, nvir, nvir)"
        )
    return (
        np.asarray(left_c0)[()],
        resolved_c1,
        resolved_c2,
        resolved_c3,
    )


def amplitude_data(
    c0: complex,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    left_c0: complex | None = None,
    left_c1: np.ndarray | None = None,
    left_c2: np.ndarray | None = None,
    left_c3: np.ndarray | None = None,
) -> tuple[dict[str, object], np.dtype]:
    """Build the validated data mapping shared by norm and RDM kernels."""

    c1, c2, c3, nocc, nvir = validate_right_amplitudes(c1, c2, c3)
    left_c0, left_c1, left_c2, left_c3 = resolve_left(
        c0,
        c1,
        c2,
        c3,
        left_c0,
        left_c1,
        left_c2,
        left_c3,
    )
    dtype = result_dtype(c0, c1, c2, c3, left_c0, left_c1, left_c2, left_c3)
    data = {
        "c0": c0,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "left_c0": left_c0,
        "left_c1": left_c1,
        "left_c2": left_c2,
        "left_c3": left_c3,
        "nocc": nocc,
        "nvir": nvir,
    }
    return data, dtype


def compute_norm(
    c0: complex,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    left_c0: complex | None = None,
    left_c1: np.ndarray | None = None,
    left_c2: np.ndarray | None = None,
    left_c3: np.ndarray | None = None,
) -> np.complex128:
    r"""Return the unnormalized Matthews overlap ``<left Psi|right Psi>``."""

    data, dtype = amplitude_data(
        c0,
        c1,
        c2,
        c3,
        left_c0,
        left_c1,
        left_c2,
        left_c3,
    )
    value = ContractionEvaluator(data, dtype=dtype).evaluate(NORM_TERMS, "")
    return np.asarray(value, dtype=dtype)[()]
