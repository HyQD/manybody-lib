"""Amplitude helpers for spin-orbital CIS."""

from __future__ import annotations

import numpy as np


def compute_norm(c0: complex, c1: np.ndarray) -> np.complex128:
    """Return ``<Psi|Psi>`` without normalizing the CIS state.

    ``c1[a, i]`` stores the coefficient :math:`c_i^a`.
    """

    c1 = np.asarray(c1)
    value = np.conjugate(c0) * c0 + np.einsum(
        "ai,ai->", c1.conj(), c1, optimize=True
    )
    return np.asarray(value, dtype=np.result_type(c0, c1, np.complex128))[()]
