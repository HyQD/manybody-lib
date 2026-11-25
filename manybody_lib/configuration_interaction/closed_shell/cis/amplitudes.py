"""Amplitude helpers for closed-shell CIS in the HJO basis."""

from __future__ import annotations

import numpy as np


def compute_hermitian_left(
    c0: complex, c1: np.ndarray
) -> tuple[np.complex128, np.ndarray]:
    r"""Construct HJO left amplitudes representing ``<Psi|``.

    The right singles have shape ``c1[a, i]`` and the left singles have
    shape ``left_c1[i, a]``.  HJO biorthogonality gives
    :math:`\widetilde C_a^i=2(C_i^a)^*`.
    """

    c1 = np.asarray(c1)
    dtype = np.result_type(c0, c1, np.complex128)
    left_c0 = np.asarray(np.conjugate(c0), dtype=dtype)[()]
    left_c1 = np.asarray(2.0 * c1.T.conj(), dtype=dtype)
    return left_c0, left_c1


def resolve_left(
    c0: complex,
    c1: np.ndarray,
    left_c0: complex | None,
    left_c1: np.ndarray | None,
) -> tuple[np.complex128, np.ndarray]:
    """Resolve optional independent left amplitudes."""

    if left_c0 is None and left_c1 is None:
        return compute_hermitian_left(c0, c1)
    if left_c0 is None or left_c1 is None:
        raise ValueError("left_c0 and left_c1 must be supplied together")
    return np.asarray(left_c0)[()], np.asarray(left_c1)


def compute_norm(
    c0: complex,
    c1: np.ndarray,
    left_c0: complex | None = None,
    left_c1: np.ndarray | None = None,
) -> np.complex128:
    """Return the unnormalized HJO overlap ``<left Psi|right Psi>``."""

    c1 = np.asarray(c1)
    left_c0, left_c1 = resolve_left(c0, c1, left_c0, left_c1)
    value = left_c0 * c0 + np.einsum(
        "ia,ai->", left_c1, c1, optimize=True
    )
    dtype = np.result_type(c0, c1, left_c0, left_c1, np.complex128)
    return np.asarray(value, dtype=dtype)[()]
