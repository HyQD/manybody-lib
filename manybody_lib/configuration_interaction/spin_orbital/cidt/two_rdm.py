"""Two-body reduced density matrix for spin-orbital CIDT."""

from __future__ import annotations

import numpy as np

from ..cisdt._engine import ContractionEvaluator
from ..cisdt.two_rdm import _PAIR_ACTIONS, _permuted_spaces
from ._specs import TWO_RDM_SPECS
from .amplitudes import amplitude_data


def compute_two_rdm(c0: complex, c2: np.ndarray, c3: np.ndarray) -> np.ndarray:
    r"""Return the unnormalized CIDT 2-RDM ``Gamma[p, q, r, s]``."""

    data, dtype = amplitude_data(c0, c2, c3)
    nocc = int(data["nocc"])
    nvir = int(data["nvir"])
    norb = nocc + nvir
    selectors = {"o": slice(0, nocc), "v": slice(nocc, norb)}
    gamma2 = np.zeros((norb, norb, norb, norb), dtype=dtype)
    evaluator = ContractionEvaluator(data, dtype=dtype)
    for spaces, spec in TWO_RDM_SPECS.items():
        block = evaluator.evaluate(spec["terms"], spec["spaces"])
        assigned = set()
        for sign, axes in _PAIR_ACTIONS:
            target_spaces = _permuted_spaces(spaces, axes)
            if target_spaces in assigned:
                continue
            assigned.add(target_spaces)
            target = tuple(selectors[space] for space in target_spaces)
            gamma2[target] = sign * block.transpose(axes)
    return gamma2
