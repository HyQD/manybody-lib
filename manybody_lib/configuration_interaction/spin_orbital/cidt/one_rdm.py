"""One-body reduced density matrix for spin-orbital CIDT."""

from __future__ import annotations

import numpy as np

from ..cisdt._engine import ContractionEvaluator
from ._specs import ONE_RDM_SPECS
from .amplitudes import amplitude_data


def compute_one_rdm(c0: complex, c2: np.ndarray, c3: np.ndarray) -> np.ndarray:
    r"""Return the unnormalized CIDT 1-RDM ``gamma[p, q]``."""

    data, dtype = amplitude_data(c0, c2, c3)
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
