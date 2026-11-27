"""Compile exact contraction reuse for the generated Matthews sigma tables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import permutations, product


CoefficientSpec = tuple[int, int, int, int]
OperandSpec = tuple[str, str]
Axes = tuple[int, ...]


# Only exact equalities, valid for complex orbitals and amplitudes, are used.
# In particular, no Hermitian conjugation or real-orbital integral symmetry
# is applied. A permutation simultaneously rearranges both input indices and
# (for integral blocks) their occupied/virtual domains.
_OPERAND_SYMMETRIES = {
    "c2": ((0, 1, 2, 3), (1, 0, 3, 2)),
    "c3": tuple(
        permutation + tuple(3 + axis for axis in permutation)
        for permutation in permutations(range(3))
    ),
    "u": ((0, 1, 2, 3), (1, 0, 3, 2)),
}


@dataclass(frozen=True, slots=True)
class Destination:
    """A scaled, permuted contraction contributing to a group sum."""

    group: int
    coefficient: CoefficientSpec
    scalar_keys: tuple[str, ...]
    axes: Axes


@dataclass(frozen=True, slots=True)
class ContractionKernel:
    """One contraction and all of its consumers in a sigma component."""

    operands: tuple[OperandSpec, ...]
    equation: str
    destinations: tuple[Destination, ...]
    use_l_integral: bool = False


@dataclass(frozen=True, slots=True)
class SigmaSchedule:
    """Immutable, shape-independent schedule of the original factored terms."""

    spaces: str
    permutation_groups: tuple[tuple[Axes, ...], ...]
    group_term_counts: tuple[int, ...]
    scalars: tuple[Destination, ...]
    kernels: tuple[ContractionKernel, ...]
    assume_symmetry: bool = False


def _normalized_subscripts(equation: str) -> tuple[tuple[tuple[int, ...], ...], Axes]:
    """Identify input index equalities independently of the label names."""

    inputs, output = equation.split("->")
    labels = {
        label: position
        for position, label in enumerate(dict.fromkeys(inputs.replace(",", "")))
    }
    normalized_inputs = tuple(
        tuple(labels[label] for label in subscript)
        for subscript in inputs.split(",")
    )
    normalized_output = tuple(labels[label] for label in output)
    return normalized_inputs, normalized_output


def _operand_variants(operand: OperandSpec, subscript: str):
    source, spaces = operand
    for axes in _OPERAND_SYMMETRIES.get(source, (tuple(range(len(subscript))),)):
        indices = "".join(subscript[axis] for axis in axes)
        domains = "".join(spaces[axis] for axis in axes) if spaces else ""
        yield (source, domains), indices


def _contraction_signature(
    operands: tuple[OperandSpec, ...],
    equation: str,
    assume_symmetry: bool,
):
    """Identify equivalent contractions and the corresponding output labels."""

    if not assume_symmetry:
        inputs, output = _normalized_subscripts(equation)
        return (operands, inputs, tuple(sorted(output))), output

    inputs, output = equation.split("->")
    variants = (
        _operand_variants(operand, subscript)
        for operand, subscript in zip(operands, inputs.split(","), strict=True)
    )
    candidates = []
    for combination in product(*variants):
        permuted_operands, permuted_inputs = zip(*combination, strict=True)
        normalized_inputs, normalized_output = _normalized_subscripts(
            ",".join(permuted_inputs) + "->" + output
        )
        key = permuted_operands, normalized_inputs, tuple(sorted(normalized_output))
        candidates.append((key, normalized_output))

    # Choose a common canonical representation, breaking ties deterministically
    # by the output ordering. The calculated kernel will still use the first
    # original equation and its original operands, not the canonical variant.
    return min(candidates)


def _integral_signature(kernel: ContractionKernel, *, exchange: bool):
    """Identify a binary amplitude/integral contraction, optionally exchanging r,s."""

    operands = kernel.operands
    if len(operands) != 2 or operands[0][0] not in ("c1", "c2", "c3"):
        return None
    if operands[1][0] != "u":
        return None

    equation = kernel.equation
    if exchange:
        inputs, output = equation.split("->")
        amplitude, integral = inputs.split(",")
        spaces = operands[1][1]
        integral = integral[:2] + integral[3] + integral[2]
        spaces = spaces[:2] + spaces[3] + spaces[2]
        operands = (operands[0], ("u", spaces))
        equation = f"{amplitude},{integral}->{output}"
    inputs, output = _normalized_subscripts(equation)
    return (operands, inputs, tuple(sorted(output))), output


def _scale_coefficient(value: CoefficientSpec, factor: Fraction) -> CoefficientSpec:
    real = Fraction(value[0], value[1]) * factor
    imag = Fraction(value[2], value[3]) * factor
    return real.numerator, real.denominator, imag.numerator, imag.denominator


def _destination_weights(kernel: ContractionKernel, axes: Axes):
    """Identify the exact weights in a common contraction-output ordering."""

    weights = {}
    for destination in kernel.destinations:
        key = (
            destination.group,
            destination.scalar_keys,
            tuple(axes[axis] for axis in destination.axes),
        )
        if key in weights:
            # The generated schedules have distinct destinations. Avoid
            # combining an unexpected duplicate without collecting it first.
            return None
        weights[key] = destination.coefficient
    return weights


def _half_l_kernel(kernel: ContractionKernel) -> ContractionKernel:
    """Replace u by L = 2u - u_exchange, scaling every destination by 1/2."""

    return replace(
        kernel,
        use_l_integral=True,
        destinations=tuple(
            replace(d, coefficient=_scale_coefficient(d.coefficient, Fraction(1, 2)))
            for d in kernel.destinations
        ),
    )


def _combine_direct_exchange_kernels(
    kernels: tuple[ContractionKernel, ...],
) -> tuple[ContractionKernel, ...]:
    """Combine exact direct/exchange pairs into a single L contraction.

    Compare the whole destination vectors after aligning integral and output
    axes. A combination is valid only if one vector is exactly -1/2 of
    the other, including every permutation group and scalar factor. No
    additional integral or amplitude symmetry is assumed for this step.
    """

    exchange_candidates = {}
    signatures = []
    for position, kernel in enumerate(kernels):
        signatures.append(_integral_signature(kernel, exchange=False))
        exchanged = _integral_signature(kernel, exchange=True)
        if exchanged is not None:
            key, output = exchanged
            exchange_candidates.setdefault(key, []).append((position, output))

    consumed = set()
    result = []
    for position, kernel in enumerate(kernels):
        if position in consumed:
            continue
        signature = signatures[position]
        if signature is not None:
            key, output = signature
            direct_weights = _destination_weights(kernel, tuple(range(len(output))))
            for other_position, other_output in exchange_candidates.get(key, ()):
                if other_position <= position or other_position in consumed:
                    continue
                other = kernels[other_position]
                axes = tuple(output.index(label) for label in other_output)
                exchange_weights = _destination_weights(other, axes)
                if direct_weights is None or exchange_weights is None:
                    continue
                if direct_weights.keys() != exchange_weights.keys():
                    continue

                if all(
                    exchange_weights[target]
                    == _scale_coefficient(weight, Fraction(-1, 2))
                    for target, weight in direct_weights.items()
                ):
                    kernel = _half_l_kernel(kernel)
                elif all(
                    direct_weights[target]
                    == _scale_coefficient(weight, Fraction(-1, 2))
                    for target, weight in exchange_weights.items()
                ):
                    kernel = _half_l_kernel(other)
                else:
                    continue
                consumed.add(other_position)
                break
        result.append(kernel)
    return tuple(result)


def compile_sigma_schedule(
    spec: Mapping[str, object],
    *,
    assume_symmetry: bool = False,
    combine_integrals: bool = False,
) -> SigmaSchedule:
    """Reuse contractions differing only in index names or output-axis order.

    By default, operands, their order, and every input index equality must
    agree exactly. With ``assume_symmetry=True``, simultaneous column
    permutations of doubles and triples and simultaneous pair exchange of
    physicists' two-electron integrals are also recognized. The first
    occurrence of each kernel supplies its original equation and layout;
    other destinations receive transpose views of that same result.

    With ``combine_integrals=True``, exactly proportional direct/exchange
    destination vectors are replaced by contractions with L = 2u - u_exchange.
    """

    groups = spec["groups"]
    term_groups = (spec["explicit"],) + tuple(terms for _, terms in groups)
    scalars = []
    builders = {}
    for group, terms in enumerate(term_groups):
        for coefficient, scalar_keys, operands, equation in terms:
            if not operands:
                scalars.append(Destination(group, coefficient, scalar_keys, ()))
                continue

            key, output = _contraction_signature(operands, equation, assume_symmetry)
            if key not in builders:
                builders[key] = (operands, equation, output, [])

            _, _, first_output, destinations = builders[key]
            axes = tuple(first_output.index(label) for label in output)
            destinations.append(Destination(group, coefficient, scalar_keys, axes))

    kernels = tuple(
        ContractionKernel(operands, equation, tuple(destinations))
        for operands, equation, _, destinations in builders.values()
    )
    if combine_integrals:
        kernels = _combine_direct_exchange_kernels(kernels)

    group_term_counts = [0] * len(term_groups)
    for destination in scalars:
        group_term_counts[destination.group] += 1
    for kernel in kernels:
        for destination in kernel.destinations:
            group_term_counts[destination.group] += 1
    return SigmaSchedule(
        spaces=str(spec["spaces"]),
        permutation_groups=tuple(axes for axes, _ in groups),
        group_term_counts=tuple(group_term_counts),
        scalars=tuple(scalars),
        kernels=kernels,
        assume_symmetry=assume_symmetry,
    )
