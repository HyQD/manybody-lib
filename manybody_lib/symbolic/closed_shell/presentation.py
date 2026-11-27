"""Verified permutation factoring for canonical scalar expressions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import itertools

from .expressions import Expression, Term
from .indices import Index
from .printers import expression_to_latex


def _replace_term_indices(term: Term, substitutions: Mapping[Index, Index]) -> Term:
    return Term(
        term.coefficient,
        tuple(
            factor.replace_indices(dict(substitutions))
            for factor in term.factors
        ),
        term.operators.replace_indices(dict(substitutions)),
    )


def permute_expression(
    expression: Expression,
    substitutions: Mapping[Index, Index],
) -> Expression:
    """Apply a simultaneous free-index permutation and recanonicalize."""

    return Expression.from_terms(
        _replace_term_indices(term, substitutions)
        for term in expression
    )


@dataclass(frozen=True, slots=True)
class ColumnPermutationPresentation:
    """An expression split as ``explicit + P * representative``."""

    original: Expression
    explicit: Expression
    representative: Expression
    substitutions: tuple[tuple[Index, Index], ...]
    operator_latex: str

    @property
    def displayed_term_count(self) -> int:
        return self.explicit.term_count + self.representative.term_count

    def expand(self) -> Expression:
        substitutions = dict(self.substitutions)
        return self.explicit + self.representative + permute_expression(
            self.representative,
            substitutions,
        )

    def to_latex(self) -> str:
        pieces = []
        if self.explicit:
            pieces.append(expression_to_latex(self.explicit))
        if self.representative:
            representative = expression_to_latex(self.representative)
            pieces.append(
                rf"{self.operator_latex}\left[ {representative} \right]"
            )
        return " + ".join(pieces) if pieces else "0"


def factor_column_permutation(
    expression: Expression,
    *,
    first_upper: Index,
    second_upper: Index,
    first_lower: Index,
    second_lower: Index,
    operator_latex: str,
) -> ColumnPermutationPresentation:
    """Factor exact simultaneous upper/lower exchange pairs.

    Invariant terms remain explicit because applying the unnormalized column
    symmetrizer to them would multiply them by two.
    """

    substitutions = {
        first_upper: second_upper,
        second_upper: first_upper,
        first_lower: second_lower,
        second_lower: first_lower,
    }
    by_body = {term.body_key: term for term in expression}
    unvisited = set(by_body)
    explicit_terms = []
    representative_terms = []

    for term in expression:
        if term.body_key not in unvisited:
            continue
        swapped_expression = permute_expression(
            Expression((term,)),
            substitutions,
        )
        if swapped_expression.term_count != 1:
            explicit_terms.append(term)
            unvisited.remove(term.body_key)
            continue
        swapped = swapped_expression.terms[0]
        partner = by_body.get(swapped.body_key)
        if (
            swapped.body_key != term.body_key
            and partner is not None
            and partner.coefficient == swapped.coefficient
            and swapped.body_key in unvisited
        ):
            representative_terms.append(term)
            unvisited.remove(term.body_key)
            unvisited.remove(swapped.body_key)
        else:
            explicit_terms.append(term)
            unvisited.remove(term.body_key)

    presentation = ColumnPermutationPresentation(
        original=expression,
        explicit=Expression.from_terms(explicit_terms),
        representative=Expression.from_terms(representative_terms),
        substitutions=tuple(substitutions.items()),
        operator_latex=operator_latex,
    )
    if presentation.expand() != expression:
        raise AssertionError("column-permutation factoring failed re-expansion")
    return presentation


_S3 = tuple(itertools.permutations(range(3)))
_IDENTITY_3 = (0, 1, 2)
_SWAP_01 = (1, 0, 2)
_SWAP_02 = (2, 1, 0)
_SWAP_12 = (0, 2, 1)


def _column_substitutions(
    upper_indices: tuple[Index, Index, Index],
    lower_indices: tuple[Index, Index, Index],
    permutation: tuple[int, int, int],
) -> dict[Index, Index]:
    substitutions = {
        upper_indices[position]: upper_indices[permutation[position]]
        for position in range(3)
    }
    substitutions.update(
        {
            lower_indices[position]: lower_indices[permutation[position]]
            for position in range(3)
        }
    )
    return substitutions


@dataclass(frozen=True, slots=True)
class TriplesPermutationGroup:
    """Representatives sharing one simultaneous-column permutation sum."""

    operator_latex: str
    actions: tuple[tuple[int, int, int], ...]
    representatives: Expression


@dataclass(frozen=True, slots=True)
class TriplesColumnPermutationPresentation:
    """Verified S3-orbit presentation of a triples-rank expression."""

    original: Expression
    explicit: Expression
    groups: tuple[TriplesPermutationGroup, ...]
    upper_indices: tuple[Index, Index, Index]
    lower_indices: tuple[Index, Index, Index]

    @property
    def displayed_term_count(self) -> int:
        return self.explicit.term_count + sum(
            group.representatives.term_count
            for group in self.groups
        )

    def expand(self) -> Expression:
        terms = list(self.explicit)
        for group in self.groups:
            for action in group.actions:
                substitutions = _column_substitutions(
                    self.upper_indices,
                    self.lower_indices,
                    action,
                )
                terms.extend(
                    permute_expression(
                        group.representatives,
                        substitutions,
                    )
                )
        return Expression.from_terms(terms)

    def to_latex(self) -> str:
        pieces = []
        if self.explicit:
            pieces.append(expression_to_latex(self.explicit))
        for group in self.groups:
            if not group.representatives:
                continue
            body = expression_to_latex(group.representatives)
            pieces.append(
                rf"{group.operator_latex}\left[ {body} \right]"
            )
        return " + ".join(pieces) if pieces else "0"


def triples_column_permutation_definitions(
    upper_indices: tuple[Index, Index, Index],
    lower_indices: tuple[Index, Index, Index],
) -> str:
    """Return LaTeX definitions for the verified triples orbit operators."""

    a, b, c = (index.name for index in upper_indices)
    i, j, k = (index.name for index in lower_indices)
    return "\n".join(
        (
            r"\begin{align}",
            (
                rf"\widehat P_{{{a}{b}}}\widehat P_{{{i}{j}}}"
                rf"X^{{{a}{b}{c}}}_{{{i}{j}{k}}}"
                rf" &= X^{{{b}{a}{c}}}_{{{j}{i}{k}}}, \\"
            ),
            (
                rf"\widehat{{\mathcal P}}^{{{a}{b}}}_{{{i}{j}}}"
                rf" &= 1+\widehat P_{{{a}{b}}}\widehat P_{{{i}{j}}}, \\"
            ),
            (
                rf"\widehat{{\mathcal P}}^{{{c}/{a}{b}}}_{{{k}/{i}{j}}}"
                rf" &= 1+\widehat P_{{{a}{c}}}\widehat P_{{{i}{k}}}"
                rf"+\widehat P_{{{b}{c}}}\widehat P_{{{j}{k}}}, \\"
            ),
            (
                rf"\widehat{{\mathcal S}}^{{{a}{b}{c}}}_{{{i}{j}{k}}}"
                rf"X^{{{a}{b}{c}}}_{{{i}{j}{k}}}"
                rf" &= \sum_{{\pi\in S_3}}"
                rf"X^{{\pi({a}{b}{c})}}_{{\pi({i}{j}{k})}}."
            ),
            r"\end{align}",
        )
    )


@lru_cache(maxsize=None)
def factor_triples_column_permutations(
    expression: Expression,
    *,
    upper_indices: tuple[Index, Index, Index],
    lower_indices: tuple[Index, Index, Index],
) -> TriplesColumnPermutationPresentation:
    """Factor complete simultaneous-column S3 orbits.

    Orbit sizes six, three, and two are represented respectively by the full
    S3 symmetrizer, the three cosets of a two-column stabilizer, and a single
    column exchange.  Invariant terms remain explicit.  The chosen operators
    are unnormalized because they are presentation devices, not projectors.
    """

    by_body = {term.body_key: term for term in expression}
    unvisited = set(by_body)
    explicit_terms = []
    representatives: dict[int, list[Term]] = {2: [], 3: [], 6: []}

    def permuted(term: Term, action: tuple[int, int, int]) -> Term:
        result = permute_expression(
            Expression((term,)),
            _column_substitutions(
                upper_indices,
                lower_indices,
                action,
            ),
        )
        if result.term_count != 1:
            raise AssertionError("a column permutation changed the term count")
        return result.terms[0]

    def stabilizer(term: Term) -> set[tuple[int, int, int]]:
        return {
            action
            for action in _S3
            if permuted(term, action).body_key == term.body_key
        }

    for seed in expression:
        if seed.body_key not in unvisited:
            continue
        transformed = {
            action: permuted(seed, action)
            for action in _S3
        }
        orbit = {term.body_key for term in transformed.values()}
        complete = all(
            term.body_key in by_body
            and by_body[term.body_key].coefficient == term.coefficient
            for term in transformed.values()
        )
        if not complete:
            explicit_terms.append(seed)
            unvisited.remove(seed.body_key)
            continue

        size = len(orbit)
        candidates = [term for term in expression if term.body_key in orbit]
        if size == 3:
            candidates = [
                term
                for term in candidates
                if _SWAP_01 in stabilizer(term)
            ]
        if not candidates:
            raise AssertionError("could not choose a canonical orbit representative")
        representative = candidates[0]

        if size == 1:
            explicit_terms.append(representative)
        elif size == 2:
            actions = (_IDENTITY_3, _SWAP_01)
            if {
                permuted(representative, action).body_key
                for action in actions
            } != orbit:
                raise AssertionError(
                    "two-term triples orbit has an unexpected stabilizer"
                )
            representatives[2].append(representative)
        elif size == 3:
            actions = (_IDENTITY_3, _SWAP_02, _SWAP_12)
            if {
                permuted(representative, action).body_key
                for action in actions
            } != orbit:
                raise AssertionError(
                    "three-term triples orbit has an unexpected stabilizer"
                )
            representatives[3].append(representative)
        elif size == 6:
            representatives[6].append(representative)
        else:
            raise AssertionError(f"unexpected S3 orbit size {size}")
        unvisited.difference_update(orbit)

    a, b, c = (index.name for index in upper_indices)
    i, j, k = (index.name for index in lower_indices)
    specs = (
        (
            2,
            rf"\widehat{{\mathcal P}}^{{{a}{b}}}_{{{i}{j}}}",
            (_IDENTITY_3, _SWAP_01),
        ),
        (
            3,
            rf"\widehat{{\mathcal P}}^{{{c}/{a}{b}}}_{{{k}/{i}{j}}}",
            (_IDENTITY_3, _SWAP_02, _SWAP_12),
        ),
        (6, rf"\widehat{{\mathcal S}}^{{{a}{b}{c}}}_{{{i}{j}{k}}}", _S3),
    )
    groups = tuple(
        TriplesPermutationGroup(
            operator_latex,
            actions,
            Expression.from_terms(representatives[size]),
        )
        for size, operator_latex, actions in specs
        if representatives[size]
    )
    presentation = TriplesColumnPermutationPresentation(
        expression,
        Expression.from_terms(explicit_terms),
        groups,
        upper_indices,
        lower_indices,
    )
    if presentation.expand() != expression:
        raise AssertionError("triples permutation factoring failed re-expansion")
    return presentation
