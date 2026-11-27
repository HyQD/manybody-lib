"""Explicit GEMM plans for binary, rank-six triples contractions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from math import prod
from operator import index

import numpy as np


@dataclass(frozen=True, slots=True)
class GemmPlan:
    """Map an Einstein contraction to two matrices and output transpose views."""

    amplitude_axes: tuple[int, ...]
    integral_axes: tuple[int, ...]
    amplitude_free_axes: tuple[int, ...]
    integral_free_axes: tuple[int, ...]
    contraction_axes: tuple[int, ...]
    output_axes: tuple[int, ...]
    amplitude_tile_axis: int
    output_tile_axis: int
    contract_first: bool

    @property
    def amplitude_layout(self):
        """Identity of the amplitude packing, independent of the integrals."""

        return (
            self.amplitude_axes,
            self.amplitude_free_axes,
            self.contraction_axes,
            self.amplitude_tile_axis,
            self.contract_first,
        )

    def prepare_integral(self, amplitude, integral):
        contracted = prod(amplitude.shape[axis] for axis in self.contraction_axes)
        integral_free_shape = tuple(
            integral.shape[axis] for axis in self.integral_free_axes
        )
        integral_free = prod(integral_free_shape)
        integral_shape = (
            (integral_free, contracted)
            if self.contract_first
            else (contracted, integral_free)
        )
        # Pack the (usually smaller) integral matrix once for all amplitude tiles.
        integral_matrix = np.ascontiguousarray(
            integral.transpose(self.integral_axes).reshape(integral_shape)
        )
        return integral_matrix, integral_free_shape

    def amplitude_tiles(self, amplitude, block_size=None):
        """Yield a packed matrix (or view) shared by compatible GEMMs."""

        extent = amplitude.shape[self.amplitude_tile_axis]
        if not extent:
            return
        block_size = extent if block_size is None else min(extent, block_size)
        for start in range(0, extent, block_size):
            stop = min(start + block_size, extent)
            # Do not retain a reference in the generator frame: the evaluator
            # releases the pack before accumulating its last GEMM output.
            yield start, stop, *self._pack_amplitude_tile(amplitude, start, stop)

    def _pack_amplitude_tile(self, amplitude, start, stop):
        contracted = prod(amplitude.shape[axis] for axis in self.contraction_axes)
        selection = [slice(None)] * amplitude.ndim
        selection[self.amplitude_tile_axis] = slice(start, stop)
        amplitude_tile = amplitude[tuple(selection)]
        amplitude_free_shape = tuple(
            amplitude_tile.shape[axis] for axis in self.amplitude_free_axes
        )
        amplitude_free = prod(amplitude_free_shape)
        # Only pack the selected tile, and preserve a matrix view whenever
        # its contracted/free axes can be reshaped without copying.
        amplitude_shape = (
            (contracted, amplitude_free)
            if self.contract_first
            else (amplitude_free, contracted)
        )
        matrix = amplitude_tile.transpose(self.amplitude_axes).reshape(amplitude_shape)
        return matrix, amplitude_free_shape

    def contract_tile(self, matrix, amplitude_free_shape, prepared_integral):
        integral_matrix, integral_free_shape = prepared_integral
        if self.contract_first:
            product = integral_matrix @ matrix
            shape = integral_free_shape + amplitude_free_shape
        else:
            product = matrix @ integral_matrix
            shape = amplitude_free_shape + integral_free_shape
        return product.reshape(shape).transpose(self.output_axes)

    def tiles(self, amplitude, integral, block_size=None):
        """Yield (start, stop, contribution) tiles in the original output order."""

        prepared = self.prepare_integral(amplitude, integral)
        for start, stop, matrix, free_shape in self.amplitude_tiles(
            amplitude, block_size
        ):
            contribution = self.contract_tile(matrix, free_shape, prepared)
            del matrix
            yield start, stop, contribution
            del contribution


def _contiguous_contraction_sides(labels, contracted):
    positions = tuple(i for i, label in enumerate(labels) if label in contracted)
    prefix = positions == tuple(range(len(positions)))
    suffix = positions == tuple(range(len(labels) - len(positions), len(labels)))
    return prefix, suffix


@lru_cache(maxsize=None)
def compile_gemm_plan(equation: str, assume_symmetry: bool = False) -> GemmPlan | None:
    """Compile a two-operand C3 contraction, without assuming real integrals.

    Under the already-enabled column symmetry, a different labeling of the
    *amplitude* axes may expose a contiguous matrix view. The integral slots
    and the requested output order remain unchanged. Unsupported expressions
    use the original einsum implementation.
    """

    inputs, output = equation.split("->")
    operands = inputs.split(",")
    if len(operands) != 2 or len(operands[0]) != 6 or len(output) != 6:
        return None
    left, right = operands
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        return None
    contracted = set(left) & set(right)
    if not contracted or set(output) != set(left) ^ set(right):
        return None
    if len(set(output)) != len(output):
        return None

    prefix, suffix = _contiguous_contraction_sides(left, contracted)
    if assume_symmetry and not (prefix or suffix):
        for permutation in permutations(range(3)):
            axes = permutation + tuple(3 + p for p in permutation)
            candidate = "".join(left[axis] for axis in axes)
            candidate_prefix, candidate_suffix = _contiguous_contraction_sides(
                candidate, contracted
            )
            if candidate_prefix or candidate_suffix:
                left = candidate
                prefix, suffix = candidate_prefix, candidate_suffix
                break

    contraction_labels = tuple(label for label in left if label in contracted)
    amplitude_free = tuple(label for label in left if label not in contracted)
    integral_free = tuple(label for label in right if label not in contracted)
    # Choose a free virtual axis (one of the first three amplitude slots).
    tile_axis = next((i for i in range(3) if left[i] not in contracted), None)
    if tile_axis is None:
        return None

    contract_first = prefix and not suffix
    if contract_first:
        amplitude_order = contraction_labels + amplitude_free
        integral_order = integral_free + contraction_labels
        native_output = integral_free + amplitude_free
    else:
        amplitude_order = amplitude_free + contraction_labels
        integral_order = contraction_labels + integral_free
        native_output = amplitude_free + integral_free

    return GemmPlan(
        amplitude_axes=tuple(left.index(label) for label in amplitude_order),
        integral_axes=tuple(right.index(label) for label in integral_order),
        amplitude_free_axes=tuple(left.index(label) for label in amplitude_free),
        integral_free_axes=tuple(right.index(label) for label in integral_free),
        contraction_axes=tuple(left.index(label) for label in contraction_labels),
        output_axes=tuple(native_output.index(label) for label in output),
        amplitude_tile_axis=tile_axis,
        output_tile_axis=output.index(left[tile_axis]),
        contract_first=contract_first,
    )


def validate_gemm_options(use_gemm: bool, gemm_block_size: int | None) -> int | None:
    """Validate an optional number of free virtual indices per GEMM tile."""

    if gemm_block_size is None:
        return None
    if not use_gemm:
        raise ValueError("gemm_block_size requires use_gemm=True")
    if isinstance(gemm_block_size, bool):
        raise TypeError("gemm_block_size must be a positive integer")
    block_size = index(gemm_block_size)
    if block_size <= 0:
        raise ValueError("gemm_block_size must be positive")
    return block_size
