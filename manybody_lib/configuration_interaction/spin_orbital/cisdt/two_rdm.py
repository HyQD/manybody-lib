"""Two-body reduced density matrix for spin-orbital CISDT."""

from __future__ import annotations

import numpy as np

from ._contractions import TWO_RDM_SPECS
from ._engine import ContractionEvaluator
from .amplitudes import amplitude_data


_PAIR_ACTIONS = (
    (1, (0, 1, 2, 3)),
    (-1, (1, 0, 2, 3)),
    (-1, (0, 1, 3, 2)),
    (1, (1, 0, 3, 2)),
)


def _permuted_spaces(spaces: str, axes: tuple[int, ...]) -> str:
    inverse = np.argsort(axes)
    return "".join(spaces[int(position)] for position in inverse)


def compute_two_rdm(
    c0: complex,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
) -> np.ndarray:
    r"""Return the unnormalized CISDT 2-RDM ``Gamma[p, q, r, s]``."""

    data, dtype = amplitude_data(c0, c1, c2, c3)
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
