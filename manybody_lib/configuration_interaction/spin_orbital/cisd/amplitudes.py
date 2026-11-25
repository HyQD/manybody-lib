"""Amplitude helpers for spin-orbital CISD."""

from __future__ import annotations

import numpy as np


def compute_norm(c0: complex, c1: np.ndarray, c2: np.ndarray) -> np.complex128:
    """Return ``<Psi|Psi>`` without normalizing the CISD state."""

    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    value = np.conjugate(c0) * c0
    value += np.einsum("ai,ai->", c1.conj(), c1, optimize=True)
    value += 0.25 * np.einsum(
        "abij,abij->", c2.conj(), c2, optimize=True
    )
    return np.asarray(value, dtype=np.result_type(c0, c1, c2, np.complex128))[()]
