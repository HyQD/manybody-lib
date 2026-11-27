"""Operator products and commutators for the spin-free algebra."""

from __future__ import annotations

from functools import lru_cache
from typing import TypeAlias

from .expressions import Expression, Term
from .indices import Index
from .operators import (
    OneBodyOperator,
    OperatorWord,
    SpinFreeOperator,
    TwoBodyOperator,
)
from .tensors import Delta


AlgebraicOperand: TypeAlias = SpinFreeOperator | OperatorWord | Term | Expression


def _as_expression(value: AlgebraicOperand) -> Expression:
    if isinstance(value, Expression):
        return value
    if isinstance(value, Term):
        return Expression.from_terms((value,))
    if isinstance(value, OperatorWord):
        return Expression.from_terms((Term(operators=value),))
    if isinstance(value, (OneBodyOperator, TwoBodyOperator)):
        return Expression.from_terms(
            (Term(operators=OperatorWord((value,))),)
        )
    raise TypeError(f"unsupported algebraic operand {type(value).__name__}")


def _replace_term_indices(term: Term, substitutions: dict[Index, Index]) -> Term:
    if not substitutions:
        return term
    return Term(
        term.coefficient,
        tuple(
            factor.replace_indices(substitutions)
            for factor in term.factors
        ),
        term.operators.replace_indices(substitutions),
    )


def _freshen_right_dummies(left: Term, right: Term) -> Term:
    """Keep the independently scoped dummy indices of two terms distinct."""

    collisions = set(left.dummies) & set(right.dummies)
    if not collisions:
        return right

    used = {(index.space, index.name) for index in left.indices + right.indices}
    substitutions = {}
    counter = 0
    for index in sorted(collisions, key=Index.sort_key):
        while (index.space, f"{index.name}_r{counter}") in used:
            counter += 1
        replacement = index.renamed(f"{index.name}_r{counter}")
        substitutions[index] = replacement
        used.add((replacement.space, replacement.name))
        counter += 1
    return _replace_term_indices(right, substitutions)


def _multiply_terms(left: Term, right: Term) -> Term:
    right = _freshen_right_dummies(left, right)
    return Term(
        left.coefficient * right.coefficient,
        left.factors + right.factors,
        left.operators * right.operators,
    )


def multiply(left: AlgebraicOperand, right: AlgebraicOperand) -> Expression:
    """Return the distributive noncommutative product ``left * right``."""

    left_expression = _as_expression(left)
    right_expression = _as_expression(right)
    return Expression.from_terms(
        _multiply_terms(left_term, right_term)
        for left_term in left_expression
        for right_term in right_expression
    )


def _surround_words(
    expression: Expression,
    left: OperatorWord = OperatorWord(),
    right: OperatorWord = OperatorWord(),
) -> Expression:
    return Expression.from_terms(
        Term(
            term.coefficient,
            term.factors,
            left * term.operators * right,
        )
        for term in expression
    )


def atomic_commutator(
    left: SpinFreeOperator,
    right: SpinFreeOperator,
) -> Expression:
    """Evaluate an atomic commutator.

    The defining ``[E, E]`` and ``[e, E]`` rules are reduced.  There is no
    required atomic ``[e, e]`` rule, so that case remains the exact word
    difference ``e e - e e``.
    """

    if not isinstance(left, (OneBodyOperator, TwoBodyOperator)):
        raise TypeError("left must be an atomic spin-free operator")
    if not isinstance(right, (OneBodyOperator, TwoBodyOperator)):
        raise TypeError("right must be an atomic spin-free operator")

    if isinstance(left, OneBodyOperator) and isinstance(
        right, OneBodyOperator
    ):
        p, q = left.indices
        r, s = right.indices
        return Expression.from_terms(
            (
                Term(
                    factors=(Delta(q, r),),
                    operators=OperatorWord((OneBodyOperator(p, s),)),
                ),
                Term(
                    -1,
                    factors=(Delta(p, s),),
                    operators=OperatorWord((OneBodyOperator(r, q),)),
                ),
            )
        )

    if isinstance(left, TwoBodyOperator) and isinstance(
        right, OneBodyOperator
    ):
        p, q, r, s = left.indices
        t, u = right.indices
        return Expression.from_terms(
            (
                Term(
                    factors=(Delta(r, t),),
                    operators=OperatorWord(
                        (TwoBodyOperator(p, q, u, s),)
                    ),
                ),
                Term(
                    factors=(Delta(s, t),),
                    operators=OperatorWord(
                        (TwoBodyOperator(p, q, r, u),)
                    ),
                ),
                Term(
                    -1,
                    factors=(Delta(p, u),),
                    operators=OperatorWord(
                        (TwoBodyOperator(t, q, r, s),)
                    ),
                ),
                Term(
                    -1,
                    factors=(Delta(q, u),),
                    operators=OperatorWord(
                        (TwoBodyOperator(p, t, r, s),)
                    ),
                ),
            )
        )

    if isinstance(left, OneBodyOperator) and isinstance(
        right, TwoBodyOperator
    ):
        return -atomic_commutator(right, left)

    return Expression.from_terms(
        (
            Term(operators=OperatorWord((left, right))),
            Term(-1, operators=OperatorWord((right, left))),
        )
    )


@lru_cache(maxsize=None)
def _word_commutator(left: OperatorWord, right: OperatorWord) -> Expression:
    if not left or not right:
        return Expression.zero()

    if len(left) == 1 and len(right) == 1:
        return atomic_commutator(left.operators[0], right.operators[0])

    if len(right) > 1:
        first = OperatorWord((right.operators[0],))
        rest = OperatorWord(right.operators[1:])
        return _surround_words(_word_commutator(left, first), right=rest) + (
            _surround_words(_word_commutator(left, rest), left=first)
        )

    prefix = OperatorWord(left.operators[:-1])
    last = OperatorWord((left.operators[-1],))
    return _surround_words(_word_commutator(last, right), left=prefix) + (
        _surround_words(_word_commutator(prefix, right), right=last)
    )


def commutator(left: AlgebraicOperand, right: AlgebraicOperand) -> Expression:
    """Return ``[left, right]`` using atomic rules and the Leibniz laws."""

    left_expression = _as_expression(left)
    right_expression = _as_expression(right)
    terms = []
    for left_term in left_expression:
        for original_right_term in right_expression:
            right_term = _freshen_right_dummies(
                left_term,
                original_right_term,
            )
            operator_part = _word_commutator(
                left_term.operators,
                right_term.operators,
            )
            for operator_term in operator_part:
                terms.append(
                    Term(
                        left_term.coefficient
                        * right_term.coefficient
                        * operator_term.coefficient,
                        left_term.factors
                        + right_term.factors
                        + operator_term.factors,
                        operator_term.operators,
                    )
                )
    return Expression.from_terms(terms)


def commutator_cache_clear() -> None:
    """Clear the internal operator-word commutator cache."""

    _word_commutator.cache_clear()


def commutator_cache_info():
    """Return ``functools`` cache statistics for word commutators."""

    return _word_commutator.cache_info()
