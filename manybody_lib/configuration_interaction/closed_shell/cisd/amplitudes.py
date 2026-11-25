"""Amplitude helpers for closed-shell CISD in the HJO basis."""

from __future__ import annotations

import numpy as np


def compute_hermitian_left(
    c0: complex, c1: np.ndarray, c2: np.ndarray
) -> tuple[np.complex128, np.ndarray, np.ndarray]:
    r"""Construct HJO left amplitudes representing ``<Psi|``."""

    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    dtype = np.result_type(c0, c1, c2, np.complex128)
    left_c0 = np.asarray(np.conjugate(c0), dtype=dtype)[()]
    left_c1 = np.asarray(2.0 * c1.T.conj(), dtype=dtype)
    transformed = 4.0 * c2 - 2.0 * c2.swapaxes(2, 3)
    left_c2 = np.asarray(transformed.transpose(2, 3, 0, 1).conj(), dtype=dtype)
    return left_c0, left_c1, left_c2


def resolve_left(
    c0: complex,
    c1: np.ndarray,
    c2: np.ndarray,
    left_c0: complex | None,
    left_c1: np.ndarray | None,
    left_c2: np.ndarray | None,
) -> tuple[np.complex128, np.ndarray, np.ndarray]:
    """Resolve optional independent left amplitudes."""

    supplied = (left_c0 is not None, left_c1 is not None, left_c2 is not None)
    if not any(supplied):
        return compute_hermitian_left(c0, c1, c2)
    if not all(supplied):
        raise ValueError("left_c0, left_c1, and left_c2 must be supplied together")
    return np.asarray(left_c0)[()], np.asarray(left_c1), np.asarray(left_c2)


def compute_norm(
    c0: complex,
    c1: np.ndarray,
    c2: np.ndarray,
    left_c0: complex | None = None,
    left_c1: np.ndarray | None = None,
    left_c2: np.ndarray | None = None,
) -> np.complex128:
    """Return the unnormalized HJO overlap ``<left Psi|right Psi>``."""

    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    left_c0, left_c1, left_c2 = resolve_left(
        c0, c1, c2, left_c0, left_c1, left_c2
    )
    value = left_c0 * c0
    value += np.einsum("ia,ai->", left_c1, c1, optimize=True)
    value += 0.5 * np.einsum("ijab,abij->", left_c2, c2, optimize=True)
    dtype = np.result_type(c0, c1, c2, left_c0, left_c1, left_c2, np.complex128)
    return np.asarray(value, dtype=dtype)[()]
