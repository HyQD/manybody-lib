"""Norm, one-body RDM, and two-body RDM generation for truncated CI."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ..commutators import multiply
from ..expressions import Expression, Term
from ..indices import Index, occupied, virtual
from ..manifolds.hjo import build_hjo_left_state, build_hjo_right_state
from ..operators import OneBodyOperator, OperatorWord, TwoBodyOperator
from ..presentation import factor_column_permutation
from ..printers import expression_to_latex
from ..reference import reference_expectation


@dataclass(frozen=True, slots=True)
class RDMBlock:
    """One externally indexed RDM block."""

    rank: int
    label: str
    indices: tuple[Index, ...]
    expression: Expression


@dataclass(frozen=True, slots=True)
class CIRDMs:
    """Unnormalized norm and nonzero CI RDM blocks."""

    method: str
    norm: Expression
    one_body: tuple[RDMBlock, ...]
    two_body: tuple[RDMBlock, ...]

    def block(self, rank: int, label: str) -> RDMBlock:
        blocks = self.one_body if rank == 1 else self.two_body
        for block in blocks:
            if block.label == label:
                return block
        raise KeyError((rank, label))


@dataclass(frozen=True, slots=True)
class SymbolicQuotient:
    """An explicit numerator/denominator normalization request."""

    numerator: Expression
    denominator: Expression

    def to_latex(self) -> str:
        numerator = expression_to_latex(self.numerator)
        denominator = expression_to_latex(self.denominator)
        return rf"\frac{{{numerator}}}{{{denominator}}}"


def normalize_by_norm(expression: Expression, norm: Expression) -> SymbolicQuotient:
    """Represent explicit normalization without changing the raw expression."""

    if not norm:
        raise ZeroDivisionError("cannot normalize by the zero expression")
    return SymbolicQuotient(expression, norm)


_ONE_BODY_SPECS = (
    (r"\gamma^{j}_{i}", (occupied("i"), occupied("j"))),
    (r"\gamma^{a}_{i}", (occupied("i"), virtual("a"))),
    (r"\gamma^{i}_{a}", (virtual("a"), occupied("i"))),
    (r"\gamma^{b}_{a}", (virtual("a"), virtual("b"))),
)

_TWO_BODY_SPECS = (
    (
        r"\Gamma^{kl}_{ij}",
        (occupied("i"), occupied("j"), occupied("k"), occupied("l")),
    ),
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
    (
        r"\Gamma^{cd}_{ab}",
        (virtual("a"), virtual("b"), virtual("c"), virtual("d")),
    ),
)


def _matrix_element(
    left: Expression,
    operator: OperatorWord,
    right: Expression,
) -> Expression:
    return reference_expectation(
        multiply(left, multiply(Term(operators=operator), right))
    )


@lru_cache(maxsize=None)
def derive_one_rdm_block(
    method: str,
    p: Index,
    q: Index,
) -> Expression:
    """Derive ``gamma_p^q`` for a caller-selected occupied/virtual block."""

    left = build_hjo_left_state(method).expression
    right = build_hjo_right_state(method).expression
    return _matrix_element(
        left,
        OperatorWord((OneBodyOperator(p, q),)),
        right,
    )


@lru_cache(maxsize=None)
def derive_two_rdm_block(
    method: str,
    p: Index,
    q: Index,
    r: Index,
    s: Index,
) -> Expression:
    """Derive ``Gamma_pq^rs`` for a selected occupied/virtual block."""

    left = build_hjo_left_state(method).expression
    right = build_hjo_right_state(method).expression
    return _matrix_element(
        left,
        OperatorWord((TwoBodyOperator(p, q, r, s),)),
        right,
    )


@lru_cache(maxsize=3)
def derive_ci_rdms(method: str) -> CIRDMs:
    """Derive the unnormalized norm and every nonzero unique RDM block."""

    left_state = build_hjo_left_state(method)
    right_state = build_hjo_right_state(method)
    norm = reference_expectation(
        multiply(left_state.expression, right_state.expression)
    )

    one_body = []
    for label, indices in _ONE_BODY_SPECS:
        expression = derive_one_rdm_block(method, *indices)
        if expression:
            one_body.append(RDMBlock(1, label, indices, expression))

    two_body = []
    for label, indices in _TWO_BODY_SPECS:
        expression = derive_two_rdm_block(method, *indices)
        if expression:
            two_body.append(RDMBlock(2, label, indices, expression))

    return CIRDMs(
        left_state.method,
        norm,
        tuple(one_body),
        tuple(two_body),
    )


def rdms_to_latex(
    rdms: CIRDMs,
    *,
    factor_permutations: bool = True,
) -> str:
    """Render norm, 1-RDM, and 2-RDM equations as one align environment."""

    equations = [
        (r"\langle\widetilde{\Psi}\vert\Psi\rangle", rdms.norm)
    ]
    equations.extend((block.label, block.expression) for block in rdms.one_body)
    equations.extend((block.label, block.expression) for block in rdms.two_body)

    lines = [r"\begin{align}"]
    for position, (label, expression) in enumerate(equations):
        if factor_permutations and label == r"\Gamma^{kl}_{ij}":
            rhs = factor_column_permutation(
                expression,
                first_upper=occupied("k"),
                second_upper=occupied("l"),
                first_lower=occupied("i"),
                second_lower=occupied("j"),
                operator_latex=r"\mathcal{P}^{kl}_{ij}",
            ).to_latex()
        else:
            rhs = expression_to_latex(expression)
        suffix = r" \\" if position + 1 < len(equations) else ""
        lines.append(rf"{label} &= {rhs}{suffix}")
    lines.append(r"\end{align}")
    return "\n".join(lines)
