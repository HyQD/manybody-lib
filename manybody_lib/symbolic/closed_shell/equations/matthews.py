"""Matthews-manifold closed-shell CISDT sigma vectors and RDMs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time
import tracemalloc

from ..commutators import multiply
from ..expressions import Expression, Term
from ..hamiltonian import build_normal_ordered_hamiltonian
from ..indices import Index, OrbitalSpace, occupied, virtual
from ..manifolds.matthews import (
    apply_matthews_pseudoinverse,
    build_matthews_left_state,
    build_matthews_right_state,
    factor_matthews_metric_resolution,
    matthews_covariant_projector,
)
from ..operators import OneBodyOperator, OperatorWord, TwoBodyOperator
from ..printers import expression_to_latex
from ..presentation import (
    ColumnPermutationPresentation,
    TriplesColumnPermutationPresentation,
    factor_column_permutation,
    factor_triples_column_permutations,
    triples_column_permutation_definitions,
)
from ..reference import reference_expectation
from .ci import RightSigmaVector
from .rdm import RDMBlock


@dataclass(frozen=True, slots=True)
class DerivationMeasurement:
    """Wall time, Python peak allocation, and final term count."""

    label: str
    seconds: float
    peak_bytes: int
    term_count: int


@dataclass(frozen=True, slots=True)
class MatthewsSigmaVector:
    """Metric-resolved and raw covariant Matthews CISDT sigma components."""

    resolved: RightSigmaVector
    covariant: tuple[tuple[int, Expression], ...]
    measurements: tuple[DerivationMeasurement, ...]

    def __getitem__(self, rank: int) -> Expression:
        return self.resolved[rank]

    def presentation(self, rank: int):
        """Return a verified metric-permutation presentation for one rank."""

        covariant = dict(self.covariant)[rank]
        occupied_indices = tuple(
            occupied("ijk"[position])
            for position in range(rank)
        )
        return factor_matthews_metric_resolution(
            covariant,
            occupied_indices,
        )

    def column_presentation(self, rank: int):
        """Return a verified external-column presentation where applicable."""

        if rank == 2:
            return factor_column_permutation(
                self[rank],
                first_upper=virtual("a"),
                second_upper=virtual("b"),
                first_lower=occupied("i"),
                second_lower=occupied("j"),
                operator_latex=r"\widehat{\mathcal P}^{ab}_{ij}",
            )
        if rank == 3:
            return factor_triples_column_permutations(
                self[rank],
                upper_indices=tuple(map(virtual, "abc")),
                lower_indices=tuple(map(occupied, "ijk")),
            )
        return None


@dataclass(frozen=True, slots=True)
class MatthewsRDMs:
    """Unnormalized Matthews CISDT norm and nonzero RDM blocks."""

    method: str
    norm: Expression
    one_body: tuple[RDMBlock, ...]
    two_body: tuple[RDMBlock, ...]
    measurements: tuple[DerivationMeasurement, ...]

    def block(self, rank: int, label: str) -> RDMBlock:
        blocks = self.one_body if rank == 1 else self.two_body
        for block in blocks:
            if block.label == label:
                return block
        raise KeyError((rank, label))


def _measure(label: str, function):
    tracemalloc.start()
    started = time.perf_counter()
    try:
        expression = function()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    measurement = DerivationMeasurement(
        label,
        time.perf_counter() - started,
        peak,
        expression.term_count,
    )
    return expression, measurement


def _sector_matrix_element(left, operator: OperatorWord, right) -> Expression:
    return reference_expectation(
        multiply(left, multiply(Term(operators=operator), right))
    )


def _operator_grade(indices: tuple[Index, ...]) -> int:
    half = len(indices) // 2
    creators = indices[:half]
    annihilators = indices[half:]
    return sum(
        int(index.space is OrbitalSpace.VIRTUAL)
        for index in creators
    ) - sum(
        int(index.space is OrbitalSpace.VIRTUAL)
        for index in annihilators
    )


@lru_cache(maxsize=None)
def derive_matthews_one_rdm_block(p: Index, q: Index) -> Expression:
    """Derive one Matthews CISDT ``gamma_p^q`` block."""

    grade = _operator_grade((p, q))
    left = build_matthews_left_state()
    right = build_matthews_right_state()
    terms = []
    operator = OperatorWord((OneBodyOperator(p, q),))
    for right_sector in right.sectors:
        left_rank = right_sector.rank + grade
        if not 0 <= left_rank <= 3:
            continue
        left_sector = left.sectors[left_rank]
        terms.extend(
            _sector_matrix_element(
                left_sector.expression,
                operator,
                right_sector.expression,
            )
        )
    return Expression.from_terms(terms)


@lru_cache(maxsize=None)
def derive_matthews_two_rdm_block(
    p: Index,
    q: Index,
    r: Index,
    s: Index,
) -> Expression:
    """Derive one Matthews CISDT ``Gamma_pq^rs`` block."""

    grade = _operator_grade((p, q, r, s))
    left = build_matthews_left_state()
    right = build_matthews_right_state()
    terms = []
    operator = OperatorWord((TwoBodyOperator(p, q, r, s),))
    for right_sector in right.sectors:
        left_rank = right_sector.rank + grade
        if not 0 <= left_rank <= 3:
            continue
        left_sector = left.sectors[left_rank]
        terms.extend(
            _sector_matrix_element(
                left_sector.expression,
                operator,
                right_sector.expression,
            )
        )
    return Expression.from_terms(terms)


_ONE_BODY_SPECS = (
    (r"\gamma^{j}_{i}", (occupied("i"), occupied("j"))),
    (r"\gamma^{a}_{i}", (occupied("i"), virtual("a"))),
    (r"\gamma^{i}_{a}", (virtual("a"), occupied("i"))),
    (r"\gamma^{b}_{a}", (virtual("a"), virtual("b"))),
)
_TWO_BODY_SPECS = (
    (r"\Gamma^{kl}_{ij}", tuple(map(occupied, "ijkl"))),
    (
        r"\Gamma^{ak}_{ij}",
        (occupied("i"), occupied("j"), virtual("a"), occupied("k")),
    ),
    (
        r"\Gamma^{jk}_{ai}",
        (virtual("a"), occupied("i"), occupied("j"), occupied("k")),
    ),
    (
        r"\Gamma^{ab}_{ij}",
        (occupied("i"), occupied("j"), virtual("a"), virtual("b")),
    ),
    (
        r"\Gamma^{ij}_{ab}",
        (virtual("a"), virtual("b"), occupied("i"), occupied("j")),
    ),
    (
        r"\Gamma^{bj}_{ai}",
        (virtual("a"), occupied("i"), virtual("b"), occupied("j")),
    ),
    (
        r"\Gamma^{bj}_{ia}",
        (occupied("i"), virtual("a"), virtual("b"), occupied("j")),
    ),
    (
        r"\Gamma^{bc}_{ai}",
        (virtual("a"), occupied("i"), virtual("b"), virtual("c")),
    ),
    (
        r"\Gamma^{ci}_{ab}",
        (virtual("a"), virtual("b"), virtual("c"), occupied("i")),
    ),
    (r"\Gamma^{cd}_{ab}", tuple(map(virtual, "abcd"))),
)


@lru_cache(maxsize=1)
def derive_matthews_rdms() -> MatthewsRDMs:
    """Derive the unnormalized Matthews CISDT norm and all unique RDM blocks."""

    left = build_matthews_left_state()
    right = build_matthews_right_state()

    def norm_function():
        terms = []
        for rank in range(4):
            terms.extend(
                _sector_matrix_element(
                    left.sectors[rank].expression,
                    OperatorWord(),
                    right.sectors[rank].expression,
                )
            )
        return Expression.from_terms(terms)

    norm, norm_measurement = _measure("norm", norm_function)
    measurements = [norm_measurement]
    one_body = []
    for label, indices in _ONE_BODY_SPECS:
        expression, measurement = _measure(
            label,
            lambda indices=indices: derive_matthews_one_rdm_block(*indices),
        )
        measurements.append(measurement)
        if expression:
            one_body.append(RDMBlock(1, label, indices, expression))

    two_body = []
    for label, indices in _TWO_BODY_SPECS:
        expression, measurement = _measure(
            label,
            lambda indices=indices: derive_matthews_two_rdm_block(*indices),
        )
        measurements.append(measurement)
        if expression:
            two_body.append(RDMBlock(2, label, indices, expression))

    return MatthewsRDMs(
        "cisdt",
        norm,
        tuple(one_body),
        tuple(two_body),
        tuple(measurements),
    )


def _derive_covariant_sigma_rank(rank: int) -> Expression:
    projector = matthews_covariant_projector(rank)
    right = build_matthews_right_state()
    hamiltonian = build_normal_ordered_hamiltonian()
    terms = []
    for ket_sector in right.sectors:
        if abs(rank - ket_sector.rank) > 2:
            continue
        matrix_element = reference_expectation(
            multiply(
                projector,
                multiply(hamiltonian, ket_sector.expression),
            )
        )
        terms.extend(matrix_element)
    return Expression.from_terms(terms)


def _derive_profiled_sigma_rank(
    rank: int,
) -> tuple[Expression, Expression, DerivationMeasurement]:
    """Derive and metric-resolve one sigma rank under one profile window."""

    tracemalloc.start()
    started = time.perf_counter()
    try:
        raw = _derive_covariant_sigma_rank(rank)
        if rank == 0:
            sigma = raw
        else:
            occupied_indices = tuple(
                occupied("ijk"[position])
                for position in range(rank)
            )
            sigma = apply_matthews_pseudoinverse(raw, occupied_indices)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    measurement = DerivationMeasurement(
        f"sigma_{rank}",
        time.perf_counter() - started,
        peak,
        sigma.term_count,
    )
    return raw, sigma, measurement


@lru_cache(maxsize=1)
def derive_matthews_sigma() -> MatthewsSigmaVector:
    """Derive all metric-resolved Matthews CISDT right sigma components."""

    covariant = []
    resolved = []
    measurements = []
    for rank in range(4):
        raw, sigma, measurement = _derive_profiled_sigma_rank(rank)
        covariant.append((rank, raw))
        resolved.append((rank, sigma))
        measurements.append(measurement)
    return MatthewsSigmaVector(
        RightSigmaVector("cisdt", tuple(resolved)),
        tuple(covariant),
        tuple(measurements),
    )


def _expression_latex_lines(
    expression: Expression,
    *,
    terms_per_line: int = 4,
) -> list[str]:
    """Render a canonical expression as sign-preserving readable chunks."""

    if not expression:
        return ["0"]
    lines = []
    for start in range(0, expression.term_count, terms_per_line):
        terms = expression.terms[start : start + terms_per_line]
        rendered = expression_to_latex(Expression(terms))
        if start and not rendered.startswith("-"):
            rendered = "{}+ " + rendered
        lines.append(rendered)
    return lines


def _operator_application_latex(operator: str, expression: Expression) -> str:
    lines = _expression_latex_lines(expression)
    if len(lines) == 1:
        return rf"{operator}\left[ {lines[0]} \right]"
    body = r" \\ &\quad ".join(lines)
    return (
        rf"{operator}\left[\begin{{aligned}}"
        rf"& {body}"
        rf"\end{{aligned}}\right]"
    )


def _presentation_latex_lines(presentation) -> list[str]:
    lines = _expression_latex_lines(presentation.explicit)
    if not presentation.explicit:
        lines = []
    if isinstance(presentation, ColumnPermutationPresentation):
        groups = (
            (
                presentation.operator_latex,
                presentation.representative,
            ),
        )
    elif isinstance(presentation, TriplesColumnPermutationPresentation):
        groups = tuple(
            (group.operator_latex, group.representatives)
            for group in presentation.groups
        )
    else:
        raise TypeError(type(presentation).__name__)
    for operator, representatives in groups:
        if not representatives:
            continue
        rendered = _operator_application_latex(operator, representatives)
        if lines:
            rendered = "{}+ " + rendered
        lines.append(rendered)
    return lines or ["0"]


def _align_latex(equations: list[tuple[str, list[str]]]) -> str:
    rows = []
    for label, bodies in equations:
        rows.append(rf"{label} &= {bodies[0]}")
        rows.extend(rf"&\quad {body}" for body in bodies[1:])
    lines = [r"\begin{align}"]
    for position, row in enumerate(rows):
        suffix = r" \\" if position + 1 < len(rows) else ""
        lines.append(row + suffix)
    lines.append(r"\end{align}")
    return "\n".join(lines)


def matthews_sigma_to_latex(
    sigma: MatthewsSigmaVector,
    *,
    factor_permutations: bool = True,
) -> str:
    """Render metric-resolved CISDT sigma components as an align environment."""

    labels = (
        r"\sigma_{0}",
        r"\sigma^{a}_{i}",
        r"\sigma^{ab}_{ij}",
        r"\sigma^{abc}_{ijk}",
    )
    equations = []
    for position, label in enumerate(labels):
        presentation = (
            sigma.column_presentation(position)
            if factor_permutations
            else None
        )
        bodies = (
            _presentation_latex_lines(presentation)
            if presentation is not None
            else _expression_latex_lines(sigma[position])
        )
        equations.append((label, bodies))
    return _align_latex(equations)


def _equations_to_latex(equations) -> str:
    rendered = [
        (label, _expression_latex_lines(expression))
        for label, expression in equations
    ]
    return _align_latex(rendered)


def matthews_one_rdms_to_latex(
    rdms: MatthewsRDMs,
    *,
    include_norm: bool = True,
) -> str:
    """Render the Matthews norm and nonzero 1-RDM blocks."""

    equations = []
    if include_norm:
        equations.append((r"\langle\widetilde\Psi\vert\Psi\rangle", rdms.norm))
    equations.extend((block.label, block.expression) for block in rdms.one_body)
    return _equations_to_latex(equations)


def matthews_two_rdms_to_latex(rdms: MatthewsRDMs) -> str:
    """Render all independent nonzero Matthews 2-RDM blocks."""

    equations = [
        (block.label, block.expression)
        for block in rdms.two_body
    ]
    return _equations_to_latex(equations)


def matthews_rdms_to_latex(rdms: MatthewsRDMs) -> str:
    """Render the Matthews norm and RDMs as an expanded align environment."""

    equations = [(r"\langle\widetilde\Psi\vert\Psi\rangle", rdms.norm)]
    equations.extend((block.label, block.expression) for block in rdms.one_body)
    equations.extend((block.label, block.expression) for block in rdms.two_body)
    return _equations_to_latex(equations)


def matthews_triples_permutation_definitions() -> str:
    """Render definitions used by the factored triples sigma presentation."""

    return triples_column_permutation_definitions(
        tuple(map(virtual, "abc")),
        tuple(map(occupied, "ijk")),
    )
