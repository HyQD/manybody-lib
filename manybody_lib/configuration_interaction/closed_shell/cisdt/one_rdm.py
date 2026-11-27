"""Spin-summed one-body RDM for closed-shell Matthews CISDT."""

from __future__ import annotations

import numpy as np

from ._contractions import ONE_RDM_SPECS
from ._engine import ContractionEvaluator
from .amplitudes import amplitude_data


def compute_one_rdm(
    c0: complex,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    left_c0: complex | None = None,
    left_c1: np.ndarray | None = None,
    left_c2: np.ndarray | None = None,
    left_c3: np.ndarray | None = None,
) -> np.ndarray:
    r"""Return the unnormalized spin-summed Matthews CISDT 1-RDM."""

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
    nocc = int(data["nocc"])
    nvir = int(data["nvir"])
    norb = nocc + nvir
    selectors = {"o": slice(0, nocc), "v": slice(nocc, norb)}
    gamma = np.zeros((norb, norb), dtype=dtype)
    evaluator = ContractionEvaluator(data, dtype=dtype)
    for spaces, spec in ONE_RDM_SPECS.items():
        block = evaluator.evaluate(spec["terms"], spec["spaces"])
        gamma[tuple(selectors[space] for space in spaces)] = block
    return gamma
