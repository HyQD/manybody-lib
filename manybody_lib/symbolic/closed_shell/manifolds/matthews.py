"""Matthews covariant singles, doubles, and singular triples manifold."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import itertools

from ..expressions import Expression, Term
from ..indices import Index, occupied, virtual
from ..operators import OneBodyOperator, OperatorWord
from ..presentation import permute_expression
from ..scalars import ExactScalar
from ..tensors import Tensor, TensorSymbol, simultaneous_column_symmetries
from .hjo import (
    CILeftState,
    CIState,
    CIStateSector,
    LEFT_REFERENCE_COEFFICIENT,
    REFERENCE_COEFFICIENT,
)


MATTHEWS_SINGLES = TensorSymbol(
    "C1",
    rank=2,
    upper_count=1,
    latex_name="C",
)
MATTHEWS_DOUBLES = TensorSymbol(
    "C2",
    rank=4,
    upper_count=2,
    symmetries=simultaneous_column_symmetries(2),
    latex_name="C",
)
MATTHEWS_TRIPLES = TensorSymbol(
    "C3",
    rank=6,
    upper_count=3,
    symmetries=simultaneous_column_symmetries(3),
    latex_name="C",
)
MATTHEWS_LEFT_SINGLES = TensorSymbol(
    "Ctilde1",
    rank=2,
    upper_count=1,
    latex_name=r"\widetilde{C}",
)
MATTHEWS_LEFT_DOUBLES = TensorSymbol(
    "Ctilde2",
    rank=4,
    upper_count=2,
    symmetries=simultaneous_column_symmetries(2),
    latex_name=r"\widetilde{C}",
)
MATTHEWS_LEFT_TRIPLES = TensorSymbol(
    "Ctilde3",
    rank=6,
    upper_count=3,
    symmetries=simultaneous_column_symmetries(3),
    latex_name=r"\widetilde{C}",
)


_RIGHT_AMPLITUDES = {
    0: REFERENCE_COEFFICIENT,
    1: MATTHEWS_SINGLES,
    2: MATTHEWS_DOUBLES,
    3: MATTHEWS_TRIPLES,
}
_LEFT_AMPLITUDES = {
    0: LEFT_REFERENCE_COEFFICIENT,
    1: MATTHEWS_LEFT_SINGLES,
    2: MATTHEWS_LEFT_DOUBLES,
    3: MATTHEWS_LEFT_TRIPLES,
}
_RANK_PREFACTORS = {
    0: Fraction(1),
    1: Fraction(1),
    2: Fraction(1, 2),
    3: Fraction(1, 6),
}


def _rank_indices(rank: int, *, dummy: bool) -> tuple[Index, ...]:
    virtual_indices = tuple(
        virtual(name, dummy=dummy)
        for name in "abc"[:rank]
    )
    occupied_indices = tuple(
        occupied(name, dummy=dummy)
        for name in "ijk"[:rank]
    )
    return virtual_indices + occupied_indices


def matthews_right_excitation(
    rank: int,
    indices: tuple[Index, ...],
) -> OperatorWord:
    """Return ``E_ai E_bj ...`` for a Matthews rank sector."""

    if rank not in range(4):
        raise ValueError("Matthews excitation rank must be 0, 1, 2, or 3")
    if len(indices) != 2 * rank:
        raise ValueError(f"rank {rank} requires {2 * rank} indices")
    virtual_indices = indices[:rank]
    occupied_indices = indices[rank:]
    return OperatorWord(
        tuple(
            OneBodyOperator(a, i)
            for a, i in zip(virtual_indices, occupied_indices, strict=True)
        )
    )


def matthews_covariant_projector(
    rank: int,
    indices: tuple[Index, ...] | None = None,
) -> Expression:
    """Return the unscaled covariant ``E_ia E_jb ...`` projector."""

    if rank not in range(4):
        raise ValueError("Matthews projector rank must be 0, 1, 2, or 3")
    if indices is None:
        indices = _rank_indices(rank, dummy=False)
    if len(indices) != 2 * rank:
        raise ValueError(f"rank {rank} requires {2 * rank} indices")
    virtual_indices = indices[:rank]
    occupied_indices = indices[rank:]
    word = OperatorWord(
        tuple(
            OneBodyOperator(i, a)
            for a, i in zip(virtual_indices, occupied_indices, strict=True)
        )
    )
    return Expression.from_terms((Term(operators=word),))


def _right_sector(rank: int) -> CIStateSector:
    indices = _rank_indices(rank, dummy=True)
    amplitude = _RIGHT_AMPLITUDES[rank]
    expression = Expression.from_terms(
        (
            Term(
                _RANK_PREFACTORS[rank],
                factors=(amplitude(*indices),),
                operators=matthews_right_excitation(rank, indices),
            ),
        )
    )
    return CIStateSector(
        rank,
        ExactScalar(_RANK_PREFACTORS[rank]),
        amplitude,
        expression,
    )


def _left_sector(rank: int) -> CIStateSector:
    right_indices = _rank_indices(rank, dummy=True)
    virtual_indices = right_indices[:rank]
    occupied_indices = right_indices[rank:]
    left_indices = occupied_indices + virtual_indices
    amplitude = _LEFT_AMPLITUDES[rank]
    word = OperatorWord(
        tuple(
            OneBodyOperator(i, a)
            for a, i in zip(virtual_indices, occupied_indices, strict=True)
        )
    )
    expression = Expression.from_terms(
        (
            Term(
                _RANK_PREFACTORS[rank],
                factors=(amplitude(*left_indices),),
                operators=word,
            ),
        )
    )
    return CIStateSector(
        rank,
        ExactScalar(_RANK_PREFACTORS[rank]),
        amplitude,
        expression,
    )


@lru_cache(maxsize=1)
def build_matthews_right_state() -> CIState:
    """Build the rank-sectorized Matthews CISDT right state."""

    sectors = tuple(_right_sector(rank) for rank in range(4))
    expression = Expression.from_terms(
        term for sector in sectors for term in sector.expression
    )
    return CIState("cisdt", sectors, expression)


@lru_cache(maxsize=1)
def build_matthews_left_state() -> CILeftState:
    """Build the independent covariant Matthews CISDT left state."""

    sectors = tuple(_left_sector(rank) for rank in range(4))
    expression = Expression.from_terms(
        term for sector in sectors for term in sector.expression
    )
    return CILeftState("cisdt", sectors, expression)


def _cycles(permutation: tuple[int, ...]) -> int:
    seen = [False] * len(permutation)
    count = 0
    for start in range(len(permutation)):
        if seen[start]:
            continue
        count += 1
        position = start
        while not seen[position]:
            seen[position] = True
            position = permutation[position]
    return count


def _compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left[right[position]] for position in range(len(left)))


def _inverse(permutation: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(permutation)
    for position, image in enumerate(permutation):
        result[image] = position
    return tuple(result)


def _class_matrix(
    rank: int,
    weights: dict[int, Fraction],
) -> tuple[tuple[Fraction, ...], ...]:
    permutations = tuple(itertools.permutations(range(rank)))
    positions = {
        permutation: position
        for position, permutation in enumerate(permutations)
    }
    matrix = [[Fraction(0) for _ in permutations] for _ in permutations]
    for row, target in enumerate(permutations):
        for action in permutations:
            source = _compose(_inverse(action), target)
            matrix[row][positions[source]] += weights[_cycles(action)]
    return tuple(tuple(row) for row in matrix)


def _matrix_product(left, right):
    return tuple(
        tuple(
            sum(
                (
                    left[row][inner] * right[inner][column]
                    for inner in range(len(right))
                ),
                Fraction(0),
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


@dataclass(frozen=True, slots=True)
class MatthewsMetric:
    """Exact regular-representation metric data for one excitation rank."""

    rank: int
    metric: tuple[tuple[Fraction, ...], ...]
    pseudoinverse: tuple[tuple[Fraction, ...], ...]
    null_projector: tuple[tuple[Fraction, ...], ...]
    physical_projector: tuple[tuple[Fraction, ...], ...]
    eigenvalues: tuple[Fraction, ...]

    @property
    def matrix_rank(self) -> int:
        """Return the exact rank implied by the validated spectrum."""

        return sum(value != 0 for value in self.eigenvalues)

    @property
    def nullity(self) -> int:
        """Return the exact nullity of the regular-representation matrix."""

        return len(self.metric) - self.matrix_rank


_METRIC_WEIGHTS = {
    1: {1: Fraction(2)},
    2: {2: Fraction(4), 1: Fraction(-2)},
    3: {3: Fraction(8), 2: Fraction(-4), 1: Fraction(2)},
}
_PSEUDOINVERSE_WEIGHTS = {
    1: {1: Fraction(1, 2)},
    2: {2: Fraction(1, 3), 1: Fraction(1, 6)},
    3: {
        3: Fraction(17, 144),
        2: Fraction(-1, 144),
        1: Fraction(-7, 144),
    },
}
_EIGENVALUES = {
    1: (Fraction(2),),
    2: (Fraction(2), Fraction(6)),
    3: (
        Fraction(0),
        Fraction(6),
        Fraction(6),
        Fraction(6),
        Fraction(6),
        Fraction(24),
    ),
}


def _factorless_scalar(expression: Expression) -> Fraction:
    value = ExactScalar(0)
    for term in expression:
        if not term.factors and not term.operators:
            value += term.coefficient
    if not value.is_real:
        raise AssertionError("a Matthews overlap coefficient must be real")
    return value.real


@lru_cache(maxsize=3)
def derive_matthews_gram(rank: int) -> tuple[tuple[Fraction, ...], ...]:
    """Construct the exact Matthews Gram matrix with the operator engine.

    The factorless part of each overlap is its value for pairwise-distinct
    occupied and virtual external orbitals.  The remaining delta-containing
    terms describe coincident external labels and do not belong to the
    regular S_n representation used for metric resolution.
    """

    if rank not in (1, 2, 3):
        raise ValueError("Matthews metric rank must be 1, 2, or 3")

    from ..commutators import multiply
    from ..reference import reference_expectation

    virtual_indices = tuple(virtual(name) for name in "abc"[:rank])
    occupied_indices = tuple(occupied(name) for name in "ijk"[:rank])
    projector = matthews_covariant_projector(
        rank,
        virtual_indices + occupied_indices,
    )
    weights = {}
    for permutation in itertools.permutations(range(rank)):
        excitation = matthews_right_excitation(
            rank,
            virtual_indices
            + tuple(occupied_indices[position] for position in permutation),
        )
        overlap = reference_expectation(
            multiply(
                projector,
                Expression.from_terms((Term(operators=excitation),)),
            )
        )
        weight = _factorless_scalar(overlap)
        cycles = _cycles(permutation)
        previous = weights.setdefault(cycles, weight)
        if previous != weight:
            raise AssertionError("Matthews Gram weights are not class-invariant")

    return _class_matrix(rank, weights)


@lru_cache(maxsize=3)
def matthews_metric(rank: int) -> MatthewsMetric:
    """Construct and validate the exact S1/S2/S3 metric and pseudoinverse."""

    if rank not in (1, 2, 3):
        raise ValueError("Matthews metric rank must be 1, 2, or 3")
    metric = derive_matthews_gram(rank)
    expected_metric = _class_matrix(rank, _METRIC_WEIGHTS[rank])
    if metric != expected_metric:
        raise AssertionError("operator-derived Matthews Gram matrix changed")
    pseudoinverse = _class_matrix(rank, _PSEUDOINVERSE_WEIGHTS[rank])
    dimension = len(metric)
    if rank == 3:
        null = tuple(
            tuple(Fraction(1, dimension) for _ in range(dimension))
            for _ in range(dimension)
        )
    else:
        null = tuple(
            tuple(Fraction(0) for _ in range(dimension))
            for _ in range(dimension)
        )
    physical = tuple(
        tuple(
            Fraction(int(row == column)) - null[row][column]
            for column in range(dimension)
        )
        for row in range(dimension)
    )
    if _matrix_product(metric, pseudoinverse) != physical:
        raise AssertionError("Matthews metric right pseudoinverse identity failed")
    if _matrix_product(pseudoinverse, metric) != physical:
        raise AssertionError("Matthews metric left pseudoinverse identity failed")
    return MatthewsMetric(
        rank,
        metric,
        pseudoinverse,
        null,
        physical,
        _EIGENVALUES[rank],
    )


def _apply_class_function(
    expression: Expression,
    occupied_indices: tuple[Index, ...],
    weights: dict[int, Fraction],
) -> Expression:
    terms = []
    for permutation in itertools.permutations(range(len(occupied_indices))):
        substitutions = {
            index: occupied_indices[permutation[position]]
            for position, index in enumerate(occupied_indices)
        }
        permuted = permute_expression(expression, substitutions)
        terms.extend(
            term.scale(weights[_cycles(permutation)])
            for term in permuted
        )
    return Expression.from_terms(terms)


def apply_matthews_pseudoinverse(
    expression: Expression,
    occupied_indices: tuple[Index, ...],
) -> Expression:
    """Apply the exact metric pseudoinverse to a covariant rank component."""

    rank = len(occupied_indices)
    if rank not in _PSEUDOINVERSE_WEIGHTS:
        raise ValueError("Matthews metric rank must be 1, 2, or 3")
    return _apply_class_function(
        expression,
        occupied_indices,
        _PSEUDOINVERSE_WEIGHTS[rank],
    )


def matthews_null_projector(
    expression: Expression,
    occupied_indices: tuple[Index, Index, Index],
) -> Expression:
    """Project onto the triples wavefunction-null trivial S3 component."""

    return _apply_class_function(
        expression,
        occupied_indices,
        {1: Fraction(1, 6), 2: Fraction(1, 6), 3: Fraction(1, 6)},
    )


def physical_matthews_triples(
    expression: Expression,
    occupied_indices: tuple[Index, Index, Index],
) -> Expression:
    """Return the Moore--Penrose zero-null triples representative."""

    return expression - matthews_null_projector(expression, occupied_indices)


_TRIPLES_OCCUPIED_SLOTS = {
    MATTHEWS_TRIPLES: (3, 4, 5),
    MATTHEWS_LEFT_TRIPLES: (0, 1, 2),
}


def _project_triples_factor(
    factor: Tensor,
    weights: dict[int, Fraction],
) -> tuple[tuple[Fraction, Tensor], ...]:
    occupied_slots = _TRIPLES_OCCUPIED_SLOTS[factor.symbol]
    result = []
    for permutation in itertools.permutations(range(3)):
        indices = list(factor.indices)
        for target, source in enumerate(permutation):
            indices[occupied_slots[target]] = factor.indices[
                occupied_slots[source]
            ]
        result.append(
            (
                weights[_cycles(permutation)],
                factor.symbol(*indices, conjugated=factor.conjugated),
            )
        )
    return tuple(result)


def matthews_triples_gauge_component(
    expression: Expression,
    amplitude: TensorSymbol,
    *,
    component: str,
) -> Expression:
    """Return the physical or null response to one triples amplitude.

    Terms independent of ``amplitude`` are omitted.  CISDT matrix elements
    are linear in each of the independent left and right triples tensors, so
    a vanishing ``component="null"`` result proves invariance under a null
    gauge shift of that tensor.
    """

    if amplitude not in _TRIPLES_OCCUPIED_SLOTS:
        raise ValueError("amplitude must be a Matthews triples tensor")
    if component == "null":
        weights = {1: Fraction(1, 6), 2: Fraction(1, 6), 3: Fraction(1, 6)}
    elif component == "physical":
        weights = {
            1: Fraction(-1, 6),
            2: Fraction(-1, 6),
            3: Fraction(5, 6),
        }
    else:
        raise ValueError("component must be 'physical' or 'null'")

    projected_terms = []
    for term in expression:
        positions = [
            position
            for position, factor in enumerate(term.factors)
            if isinstance(factor, Tensor) and factor.symbol == amplitude
        ]
        if not positions:
            continue
        if len(positions) != 1:
            raise ValueError("expected at most one selected triples factor per term")
        position = positions[0]
        factor = term.factors[position]
        for coefficient, replacement in _project_triples_factor(factor, weights):
            factors = list(term.factors)
            factors[position] = replacement
            projected_terms.append(
                Term(
                    term.coefficient * coefficient,
                    tuple(factors),
                    term.operators,
                )
            )
    return Expression.from_terms(projected_terms)


@dataclass(frozen=True, slots=True)
class MatthewsPermutationPresentation:
    """Verified factored display of ``sigma = M^+ rho``."""

    covariant: Expression
    resolved: Expression
    occupied_indices: tuple[Index, ...]

    def expand(self) -> Expression:
        """Expand the displayed metric-permutation operator exactly."""

        if not self.occupied_indices:
            return self.covariant
        return apply_matthews_pseudoinverse(
            self.covariant,
            self.occupied_indices,
        )

    def to_latex(self) -> str:
        """Render the factored metric action and its exact permutation sum."""

        from ..printers import expression_to_latex

        rank = len(self.occupied_indices)
        if rank == 0:
            return expression_to_latex(self.covariant)
        labels = tuple(index.name for index in self.occupied_indices)
        if rank == 1:
            operator = r"\frac{1}{2}"
        elif rank == 2:
            i, j = labels
            operator = rf"\frac{{1}}{{3}}+\frac{{1}}{{6}}P({i}{j})"
        else:
            i, j, k = labels
            operator = (
                r"\frac{17}{144}"
                rf"-\frac{{P({i}{j})+P({i}{k})+P({j}{k})}}{{144}}"
                rf"-\frac{{7[P({i}{j}{k})+P({i}{k}{j})]}}{{144}}"
            )
        body = expression_to_latex(self.covariant)
        return rf"\left({operator}\right)\left[ {body} \right]"


def factor_matthews_metric_resolution(
    covariant: Expression,
    occupied_indices: tuple[Index, ...],
) -> MatthewsPermutationPresentation:
    """Build and verify a reversible Matthews metric presentation."""

    resolved = (
        covariant
        if not occupied_indices
        else apply_matthews_pseudoinverse(covariant, occupied_indices)
    )
    presentation = MatthewsPermutationPresentation(
        covariant,
        resolved,
        occupied_indices,
    )
    if presentation.expand() != resolved:
        raise AssertionError("Matthews permutation factoring failed re-expansion")
    return presentation


def _hermitian_options(factor):
    if not isinstance(factor, Tensor):
        return ((ExactScalar(1), factor),)
    mapping = {
        LEFT_REFERENCE_COEFFICIENT: REFERENCE_COEFFICIENT,
        MATTHEWS_LEFT_SINGLES: MATTHEWS_SINGLES,
        MATTHEWS_LEFT_DOUBLES: MATTHEWS_DOUBLES,
        MATTHEWS_LEFT_TRIPLES: MATTHEWS_TRIPLES,
    }
    right_symbol = mapping.get(factor.symbol)
    if right_symbol is None:
        return ((ExactScalar(1), factor),)
    rank = right_symbol.upper_count
    occupied_indices = factor.indices[:rank]
    virtual_indices = factor.indices[rank:]
    return (
        (
            ExactScalar(1),
            right_symbol(
                *(virtual_indices + occupied_indices),
                conjugated=True,
            ),
        ),
    )


def matthews_hermitian_bra_substitution(expression: Expression) -> Expression:
    """Set independent Matthews left coefficients to right adjoints explicitly."""

    terms = []
    for term in expression:
        coefficient = term.coefficient
        factors = []
        for factor in term.factors:
            option_coefficient, replacement = _hermitian_options(factor)[0]
            coefficient *= option_coefficient
            factors.append(replacement)
        terms.append(Term(coefficient, tuple(factors), term.operators))
    return Expression.from_terms(terms)
