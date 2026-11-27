"""Internal evaluator for generated Matthews CISDT contractions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


TermSpec = tuple[
    tuple[int, int, int, int],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
    str,
]


class ContractionEvaluator:
    """Evaluate frozen tensor contractions against one set of arrays."""

    def __init__(
        self,
        data: Mapping[str, object],
        *,
        dtype: np.dtype,
        occupied: slice | None = None,
        virtual: slice | None = None,
    ) -> None:
        self.data = data
        self.dtype = np.dtype(dtype)
        self.occupied = occupied
        self.virtual = virtual
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def _shape(self, spaces: str) -> tuple[int, ...]:
        dimensions = {"o": self.data["nocc"], "v": self.data["nvir"]}
        return tuple(int(dimensions[space]) for space in spaces)

    def _operand(self, source: str, spaces: str) -> np.ndarray:
        key = source, spaces
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if source == "eye_o":
            value = np.eye(int(self.data["nocc"]), dtype=self.dtype)
        elif source == "eye_v":
            value = np.eye(int(self.data["nvir"]), dtype=self.dtype)
        elif source in ("f", "u"):
            if self.occupied is None or self.virtual is None:
                raise ValueError("orbital slices are required for f and u contractions")
            selectors = {
                "o": self.occupied,
                "v": self.virtual,
            }
            tensor = np.asarray(self.data[source])
            value = tensor[tuple(selectors[space] for space in spaces)]
        else:
            value = np.asarray(self.data[source])

        self._cache[key] = value
        return value

    @staticmethod
    def _coefficient(value: tuple[int, int, int, int]) -> complex:
        real_numerator, real_denominator, imag_numerator, imag_denominator = value
        return complex(
            real_numerator / real_denominator,
            imag_numerator / imag_denominator,
        )

    def evaluate(
        self,
        terms: Sequence[TermSpec],
        spaces: str,
    ) -> np.ndarray:
        """Evaluate a sum of generated terms with the requested free spaces."""

        result = np.zeros(self._shape(spaces), dtype=self.dtype)
        for coefficient_spec, scalar_keys, operand_specs, equation in terms:
            coefficient = self._coefficient(coefficient_spec)
            for key in scalar_keys:
                coefficient *= self.data[key]
            if operand_specs:
                operands = tuple(
                    self._operand(source, operand_spaces)
                    for source, operand_spaces in operand_specs
                )
                contribution = np.einsum(
                    equation,
                    *operands,
                    optimize=True,
                )
            else:
                contribution = 1
            result += coefficient * contribution
        return result

    def evaluate_sigma(self, spec: Mapping[str, object]) -> np.ndarray:
        """Evaluate explicit terms and factored column-permutation groups."""

        spaces = str(spec["spaces"])
        result = self.evaluate(spec["explicit"], spaces)
        for axes_group, representative_terms in spec["groups"]:
            representative = self.evaluate(representative_terms, spaces)
            for axes in axes_group:
                result += representative.transpose(axes)
        return result


def result_dtype(*values: object) -> np.dtype:
    """Return the common complex dtype used by the numerical kernels."""

    return np.dtype(np.result_type(*values, np.complex128))
