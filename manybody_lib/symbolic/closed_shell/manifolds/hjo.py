"""Helgaker--Jorgensen--Olsen singles and redundant-doubles manifold."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from ..commutators import multiply
from ..expressions import Expression, Term
from ..indices import Index, occupied, virtual
from ..operators import OneBodyOperator, OperatorWord
from ..reference import reference_expectation
from ..scalars import ExactScalar
from ..tensors import Tensor, TensorSymbol, simultaneous_column_symmetries


REFERENCE_COEFFICIENT = TensorSymbol("C0", rank=0, latex_name=r"C_{0}")
SINGLES_AMPLITUDE = TensorSymbol("C", rank=2, upper_count=1)
DOUBLES_AMPLITUDE = TensorSymbol(
    "C",
    rank=4,
    upper_count=2,
    symmetries=simultaneous_column_symmetries(2),
)
LEFT_REFERENCE_COEFFICIENT = TensorSymbol(
    "C0tilde",
    rank=0,
    latex_name=r"\widetilde{C}_{0}",
)
LEFT_SINGLES_AMPLITUDE = TensorSymbol(
    "Ctilde",
    rank=2,
    upper_count=1,
    latex_name=r"\widetilde{C}",
)
LEFT_DOUBLES_AMPLITUDE = TensorSymbol(
    "Ctilde",
    rank=4,
    upper_count=2,
    symmetries=simultaneous_column_symmetries(2),
    latex_name=r"\widetilde{C}",
)


@dataclass(frozen=True, slots=True)
class CIStateSector:
    """One excitation-rank contribution to a right CI state."""

    rank: int
    prefactor: ExactScalar
    amplitude: TensorSymbol
    expression: Expression


@dataclass(frozen=True, slots=True)
class CIState:
    """A truncated right CI state kept as independent rank sectors."""

    method: str
    sectors: tuple[CIStateSector, ...]
    expression: Expression

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(sector.rank for sector in self.sectors)


@dataclass(frozen=True, slots=True)
class CILeftState:
    """An independent HJO left state, without an implicit adjoint relation."""

    method: str
    sectors: tuple[CIStateSector, ...]
    expression: Expression

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(sector.rank for sector in self.sectors)


def right_excitation(rank: int, indices: tuple[Index, ...]) -> OperatorWord:
    """Return the HJO right excitation at rank zero, one, or two."""

    if rank == 0:
        if indices:
            raise ValueError("the reference excitation has no indices")
        return OperatorWord()
    if rank == 1:
        if len(indices) != 2:
            raise ValueError("a single excitation needs (a, i)")
        a, i = indices
        return OperatorWord((OneBodyOperator(a, i),))
    if rank == 2:
        if len(indices) != 4:
            raise ValueError("a double excitation needs (a, b, i, j)")
        a, b, i, j = indices
        return OperatorWord(
            (OneBodyOperator(a, i), OneBodyOperator(b, j))
        )
    raise ValueError("the HJO prototype supports excitation ranks 0, 1, and 2")


def _state_sector(rank: int) -> CIStateSector:
    if rank == 0:
        expression = Expression.from_terms(
            (Term(factors=(REFERENCE_COEFFICIENT(),)),)
        )
        return CIStateSector(
            rank,
            ExactScalar(1),
            REFERENCE_COEFFICIENT,
            expression,
        )

    a = virtual("a", dummy=True)
    i = occupied("i", dummy=True)
    if rank == 1:
        expression = Expression.from_terms(
            (
                Term(
                    factors=(SINGLES_AMPLITUDE(a, i),),
                    operators=right_excitation(1, (a, i)),
                ),
            )
        )
        return CIStateSector(rank, ExactScalar(1), SINGLES_AMPLITUDE, expression)

    if rank != 2:
        raise ValueError("the HJO prototype supports state ranks 0, 1, and 2")

    b = virtual("b", dummy=True)
    j = occupied("j", dummy=True)
    expression = Expression.from_terms(
        (
            Term(
                Fraction(1, 2),
                factors=(DOUBLES_AMPLITUDE(a, b, i, j),),
                operators=right_excitation(2, (a, b, i, j)),
            ),
        )
    )
    return CIStateSector(
        rank,
        ExactScalar(Fraction(1, 2)),
        DOUBLES_AMPLITUDE,
        expression,
    )


def _left_state_sector(rank: int) -> CIStateSector:
    if rank == 0:
        expression = Expression.from_terms(
            (Term(factors=(LEFT_REFERENCE_COEFFICIENT(),)),)
        )
        return CIStateSector(
            rank,
            ExactScalar(1),
            LEFT_REFERENCE_COEFFICIENT,
            expression,
        )

    a = virtual("a", dummy=True)
    i = occupied("i", dummy=True)
    if rank == 1:
        expression = Expression.from_terms(
            (
                Term(
                    Fraction(1, 2),
                    factors=(LEFT_SINGLES_AMPLITUDE(i, a),),
                    operators=OperatorWord((OneBodyOperator(i, a),)),
                ),
            )
        )
        return CIStateSector(
            rank,
            ExactScalar(1),
            LEFT_SINGLES_AMPLITUDE,
            expression,
        )

    if rank != 2:
        raise ValueError("the HJO prototype supports state ranks 0, 1, and 2")
    b = virtual("b", dummy=True)
    j = occupied("j", dummy=True)
    expression = Expression.from_terms(
        (
            Term(
                Fraction(1, 6),
                factors=(LEFT_DOUBLES_AMPLITUDE(i, j, a, b),),
                operators=OperatorWord(
                    (OneBodyOperator(j, b), OneBodyOperator(i, a))
                ),
            ),
            Term(
                Fraction(1, 12),
                factors=(LEFT_DOUBLES_AMPLITUDE(i, j, a, b),),
                operators=OperatorWord(
                    (OneBodyOperator(i, b), OneBodyOperator(j, a))
                ),
            ),
        )
    )
    return CIStateSector(
        rank,
        ExactScalar(Fraction(1, 2)),
        LEFT_DOUBLES_AMPLITUDE,
        expression,
    )


@lru_cache(maxsize=3)
def build_hjo_right_state(method: str) -> CIState:
    """Build the right CIS, CID, or CISD state by excitation-rank sectors."""

    method = method.lower()
    ranks_by_method = {
        "cis": (0, 1),
        "cid": (0, 2),
        "cisd": (0, 1, 2),
    }
    try:
        ranks = ranks_by_method[method]
    except KeyError as error:
        raise ValueError("method must be 'cis', 'cid', or 'cisd'") from error
    sectors = tuple(_state_sector(rank) for rank in ranks)
    expression = Expression.from_terms(
        term
        for sector in sectors
        for term in sector.expression
    )
    return CIState(method, sectors, expression)


@lru_cache(maxsize=3)
def build_hjo_left_state(method: str) -> CILeftState:
    """Build an independent HJO left CIS, CID, or CISD state."""

    right = build_hjo_right_state(method)
    sectors = tuple(_left_state_sector(rank) for rank in right.ranks)
    expression = Expression.from_terms(
        term
        for sector in sectors
        for term in sector.expression
    )
    return CILeftState(right.method, sectors, expression)


def _hermitian_factor_options(factor):
    if not isinstance(factor, Tensor):
        return ((ExactScalar(1), (factor,)),)
    if factor.symbol == LEFT_REFERENCE_COEFFICIENT:
        return (
            (
                ExactScalar(1),
                (REFERENCE_COEFFICIENT(conjugated=True),),
            ),
        )
    if factor.symbol == LEFT_SINGLES_AMPLITUDE:
        i, a = factor.indices
        return (
            (
                ExactScalar(2),
                (SINGLES_AMPLITUDE(a, i, conjugated=True),),
            ),
        )
    if factor.symbol == LEFT_DOUBLES_AMPLITUDE:
        i, j, a, b = factor.indices
        return (
            (
                ExactScalar(4),
                (DOUBLES_AMPLITUDE(a, b, i, j, conjugated=True),),
            ),
            (
                ExactScalar(-2),
                (DOUBLES_AMPLITUDE(a, b, j, i, conjugated=True),),
            ),
        )
    return ((ExactScalar(1), (factor,)),)


def hermitian_bra_substitution(expression: Expression) -> Expression:
    """Apply the explicit HJO Hermitian-bra coefficient relations."""

    terms = []
    for term in expression:
        branches = ((term.coefficient, ()),)
        for factor in term.factors:
            updated = []
            for coefficient, factors in branches:
                for option_coefficient, option_factors in (
                    _hermitian_factor_options(factor)
                ):
                    updated.append(
                        (
                            coefficient * option_coefficient,
                            factors + option_factors,
                        )
                    )
            branches = tuple(updated)
        terms.extend(
            Term(coefficient, factors, term.operators)
            for coefficient, factors in branches
        )
    return Expression.from_terms(terms)


def build_hjo_dual_projector(
    rank: int,
    indices: tuple[Index, ...] | None = None,
) -> Expression:
    """Return the HJO biorthogonal projector for one external component."""

    if rank == 0:
        if indices not in (None, ()):
            raise ValueError("the reference projector has no indices")
        return Expression.from_terms((Term(),))

    if indices is None:
        if rank == 1:
            indices = (virtual("a"), occupied("i"))
        elif rank == 2:
            indices = (
                virtual("a"),
                virtual("b"),
                occupied("i"),
                occupied("j"),
            )
    if indices is None:
        raise ValueError(
            "the HJO prototype supports projector ranks 0, 1, and 2"
        )

    if rank == 1:
        if len(indices) != 2:
            raise ValueError("a single projector needs (a, i)")
        a, i = indices
        return Expression.from_terms(
            (
                Term(
                    Fraction(1, 2),
                    operators=OperatorWord((OneBodyOperator(i, a),)),
                ),
            )
        )

    if rank == 2:
        if len(indices) != 4:
            raise ValueError("a double projector needs (a, b, i, j)")
        a, b, i, j = indices
        return Expression.from_terms(
            (
                Term(
                    Fraction(1, 3),
                    operators=OperatorWord(
                        (OneBodyOperator(j, b), OneBodyOperator(i, a))
                    ),
                ),
                Term(
                    Fraction(1, 6),
                    operators=OperatorWord(
                        (OneBodyOperator(i, b), OneBodyOperator(j, a))
                    ),
                ),
            )
        )

    raise ValueError("the HJO prototype supports projector ranks 0, 1, and 2")


def hjo_metric(
    rank: int,
    bra_indices: tuple[Index, ...],
    ket_indices: tuple[Index, ...],
) -> Expression:
    """Return ``<Phi0|Y_mu X_nu|Phi0>`` for an HJO rank sector."""

    projector = build_hjo_dual_projector(rank, bra_indices)
    excitation = Term(operators=right_excitation(rank, ket_indices))
    return reference_expectation(multiply(projector, excitation))
