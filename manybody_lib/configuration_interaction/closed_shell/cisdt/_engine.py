"""Internal evaluator for generated Matthews CISDT contractions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ._schedule import Destination, SigmaSchedule


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
        self._l_cache: dict[str, np.ndarray] = {}

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

    def _l_integral(self, spaces: str) -> np.ndarray:
        """Cache only the needed direct/exchange block for this evaluation."""

        cached = self._l_cache.get(spaces)
        if cached is None:
            exchange_spaces = spaces[:2] + spaces[3] + spaces[2]
            direct = self._operand("u", spaces)
            exchange = self._operand("u", exchange_spaces).swapaxes(2, 3)
            cached = 2 * direct - exchange
            self._l_cache[spaces] = cached
        return cached

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

    def evaluate_sigma_schedule(
        self,
        schedule: SigmaSchedule,
        *,
        optimize_accumulation: bool = False,
        use_gemm: bool = False,
        gemm_block_size: int | None = None,
    ) -> np.ndarray:
        """Evaluate each unique contraction once and stream its destinations."""

        shape = self._shape(schedule.spaces)
        optimize_accumulation = optimize_accumulation and len(shape) == 6
        if optimize_accumulation:
            from ._accumulation import accumulate_permutations
        use_gemm = use_gemm and len(shape) == 6 and all(shape)
        if use_gemm:
            from ._gemm import compile_gemm_plan

        result = np.zeros(shape, dtype=self.dtype)
        group_buffers = [result] + [None] * len(schedule.permutation_groups)
        remaining = list(schedule.group_term_counts)

        def group_target(group):
            target = group_buffers[group]
            if target is None:
                target = np.zeros(shape, dtype=self.dtype)
                group_buffers[group] = target
            return target

        def add_views(target, destinations, contribution, *, strided_target=False):
            if optimize_accumulation:
                weighted_views = []
            for destination in destinations:
                coefficient = self._coefficient(destination.coefficient)
                for key in destination.scalar_keys:
                    coefficient *= self.data[key]
                if optimize_accumulation:
                    weighted_views.append((destination.axes, coefficient))
                else:
                    target += coefficient * contribution.transpose(destination.axes)
            if optimize_accumulation:
                accumulate_permutations(
                    target, contribution, weighted_views,
                    allow_strided_target=strided_target,
                )

        def finish_group(group, count):
            remaining[group] -= count
            # Expand a completed group only once, then release its accumulator.
            # No rank-six contraction results are cached between kernels.
            if group and remaining[group] == 0:
                target = group_buffers[group]
                axes_group = schedule.permutation_groups[group - 1]
                if optimize_accumulation:
                    accumulate_permutations(
                        result, target, [(axes, 1.0) for axes in axes_group]
                    )
                else:
                    for axes in axes_group:
                        result[...] += target.transpose(axes)
                group_buffers[group] = None

        def accumulate(destinations: Sequence[Destination], contribution: np.ndarray):
            group = destinations[0].group
            add_views(group_target(group), destinations, contribution)
            finish_group(group, len(destinations))

        def grouped_destinations(destinations):
            groups = {}
            for destination in destinations:
                groups.setdefault(destination.group, []).append(destination)
            return groups

        def accumulate_gemms(batch):
            # Consecutive kernels with the same amplitude layout reuse its
            # packed matrix tile. No tile or full pack survives the batch.
            amplitude = batch[0][2][0]
            prepared = []
            for plan, kernel, operands in batch:
                groups = grouped_destinations(kernel.destinations)
                tiles_by_group = {}
                # A source-output tile can map to different target axes
                # under permutations. Partition by group and target axis.
                for group, group_destinations in groups.items():
                    by_axis = {}
                    for destination in group_destinations:
                        axis = destination.axes.index(plan.output_tile_axis)
                        by_axis.setdefault(axis, []).append(destination)
                    tiles_by_group[group] = by_axis
                integral = plan.prepare_integral(amplitude, operands[1])
                prepared.append((plan, integral, groups, tiles_by_group))

            for start, stop, matrix, free_shape in batch[0][0].amplitude_tiles(
                amplitude, gemm_block_size
            ):
                for item, (plan, integral, _, tiles_by_group) in enumerate(prepared):
                    contribution = plan.contract_tile(matrix, free_shape, integral)
                    if item + 1 == len(prepared):
                        # The pack is no longer needed while accumulating the
                        # last product of this tile.
                        del matrix
                    for group, by_axis in tiles_by_group.items():
                        target = group_target(group)
                        for axis, axis_destinations in by_axis.items():
                            selection = [slice(None)] * target.ndim
                            selection[axis] = slice(start, stop)
                            add_views(
                                target[tuple(selection)],
                                axis_destinations,
                                contribution,
                                strided_target=True,
                            )
                    del contribution
            # A group completes only once all tiles have been incorporated.
            for _, _, groups, _ in prepared:
                for group, group_destinations in groups.items():
                    finish_group(group, len(group_destinations))

        def kernel_operands(kernel):
            if kernel.use_l_integral:
                return (
                    self._operand(*kernel.operands[0]),
                    self._l_integral(kernel.operands[1][1]),
                )
            return tuple(
                self._operand(source, spaces) for source, spaces in kernel.operands
            )

        def plan_for(kernel):
            if (
                use_gemm and len(kernel.operands) == 2
                and kernel.operands[0] == ("c3", "")
            ):
                return compile_gemm_plan(kernel.equation, schedule.assume_symmetry)
            return None

        for destination in schedule.scalars:
            accumulate((destination,), np.asarray(1))

        position = 0
        while position < len(schedule.kernels):
            kernel = schedule.kernels[position]
            position += 1
            operands = kernel_operands(kernel)
            plan = plan_for(kernel)
            if plan is not None:
                batch = [(plan, kernel, operands)]
                groups = {d.group for d in kernel.destinations}
                # Only batch adjacent kernels feeding the same groups; this
                # cannot delay the release of an otherwise completed group.
                while position < len(schedule.kernels):
                    other = schedule.kernels[position]
                    other_plan = plan_for(other)
                    if (
                        other_plan is None
                        or plan.amplitude_layout != other_plan.amplitude_layout
                        or groups != {d.group for d in other.destinations}
                    ):
                        break
                    batch.append((other_plan, other, kernel_operands(other)))
                    position += 1
                accumulate_gemms(batch)
                continue

            contribution = np.asarray(
                np.einsum(kernel.equation, *operands, optimize=True)
            )
            if optimize_accumulation:
                # All destinations of one contraction feeding the same
                # permutation group are fused into one target traversal.
                for destinations in grouped_destinations(kernel.destinations).values():
                    accumulate(destinations, contribution)
            else:
                for destination in kernel.destinations:
                    accumulate((destination,), contribution)
            del contribution

        return result


def result_dtype(*values: object) -> np.dtype:
    """Return the common complex dtype used by the numerical kernels."""

    return np.dtype(np.result_type(*values, np.complex128))
