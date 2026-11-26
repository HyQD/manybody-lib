"""Verified permutation factoring for canonical scalar expressions."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

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
