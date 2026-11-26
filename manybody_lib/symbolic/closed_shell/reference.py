"""Closed-shell reference expectation values for spin-free operator words."""

from __future__ import annotations

from functools import lru_cache

from .commutators import AlgebraicOperand, commutator
from .expressions import Expression, Term
from .indices import Index, IndexRole, OrbitalSpace
from .operators import OneBodyOperator, OperatorWord, TwoBodyOperator
from .tensors import Delta, OrbitalDimension


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


def _prefix_word(expression: Expression, word: OperatorWord) -> Expression:
    return Expression.from_terms(
        Term(
            term.coefficient,
            term.factors,
            word * term.operators,
        )
        for term in expression
    )


@lru_cache(maxsize=None)
def expand_two_body_operators(word: OperatorWord) -> Expression:
    """Expand atomic ``e`` operators into their defining products of ``E``."""

    if not isinstance(word, OperatorWord):
        raise TypeError("word must be an OperatorWord")
    if not word:
        return Expression.from_terms((Term(),))

    first = word.operators[0]
    tail = expand_two_body_operators(OperatorWord(word.operators[1:]))
    if isinstance(first, OneBodyOperator):
        return _prefix_word(tail, OperatorWord((first,)))

    p, q, r, s = first.indices
    direct = _prefix_word(
        tail,
        OperatorWord(
            (OneBodyOperator(p, r), OneBodyOperator(q, s))
        ),
    )
    correction = Expression.from_terms(
        Term(
            -term.coefficient,
            (Delta(q, r),) + term.factors,
            OperatorWord((OneBodyOperator(p, s),)) * term.operators,
        )
        for term in tail
    )
    return direct + correction


def _combine_scalar_term(term: Term, expectation: Expression) -> Expression:
    return Expression.from_terms(
        Term(
            term.coefficient * expected.coefficient,
            term.factors + expected.factors,
        )
        for expected in expectation
    )


def _protect_word_dummies(
    word: OperatorWord,
) -> tuple[OperatorWord, dict[Index, Index]]:
    """Protect term-scoped dummies from word-local canonicalization."""

    dummies = sorted(
        {index for index in word.indices if index.is_dummy},
        key=Index.sort_key,
    )
    if not dummies:
        return word, {}

    used = {(index.space, index.name) for index in word.indices}
    substitutions = {}
    reverse = {}
    for position, index in enumerate(dummies):
        name = f"__bound_{position}"
        while (index.space, name) in used:
            name += "_"
        protected = Index(name, index.space, IndexRole.FREE)
        substitutions[index] = protected
        reverse[protected] = index
        used.add((protected.space, protected.name))
    return word.replace_indices(substitutions), reverse


def _expect_one_body_expression(expression: Expression) -> Expression:
    result = Expression.zero()
    for term in expression:
        result += _combine_scalar_term(
            term,
            _expect_one_body_word(term.operators),
        )
    return result


def _require_resolved_space(operator: OneBodyOperator) -> None:
    for index in operator.indices:
        if index.space not in (
            OrbitalSpace.OCCUPIED,
            OrbitalSpace.VIRTUAL,
        ):
            raise ValueError(
                "reference contraction requires occupied/virtual-resolved "
                f"operator indices; {index.name!r} is {index.space.value}"
            )


@lru_cache(maxsize=None)
def _expect_one_body_word(word: OperatorWord) -> Expression:
    """Reduce an ``E``-only word against the closed-shell determinant."""

    if not word:
        return Expression.from_terms((Term(),))
    if not all(isinstance(operator, OneBodyOperator) for operator in word.operators):
        raise TypeError("internal reference reduction expects an E-only word")

    first = word.operators[0]
    _require_resolved_space(first)
    p, q = first.indices
    tail = OperatorWord(word.operators[1:])

    if p.space is OrbitalSpace.VIRTUAL:
        return Expression.zero()

    if q.space is OrbitalSpace.OCCUPIED:
        tail_expectation = _expect_one_body_word(tail)
        return Expression.from_terms(
            Term(
                2 * term.coefficient,
                (Delta(p, q),) + term.factors,
            )
            for term in tail_expectation
        )

    if not tail:
        return Expression.zero()
    return _expect_one_body_expression(commutator(first, tail))


@lru_cache(maxsize=None)
def _reference_expectation_word(word: OperatorWord) -> Expression:
    expanded = expand_two_body_operators(word)
    return _expect_one_body_expression(expanded)


def reference_expectation(value: AlgebraicOperand) -> Expression:
    """Return ``<Phi0| value |Phi0>`` as an exactly collected expression.

    Operator indices must be resolved into occupied and virtual blocks.  A
    general spatial index has no scalar reference occupation without an
    additional occupied-space projector, so unresolved general indices are
    rejected explicitly.
    """

    expression = _as_expression(value)
    result_terms = []
    for term in expression:
        protected_word, reverse = _protect_word_dummies(term.operators)
        expectation = _reference_expectation_word(protected_word)
        for expected in expectation:
            restored_factors = tuple(
                factor.replace_indices(reverse)
                for factor in expected.factors
            )
            combined_factors = term.factors + restored_factors
            retained_indices = {
                index
                for factor in combined_factors
                for index in factor.indices
            }
            vanished_sums = tuple(
                OrbitalDimension(index.space)
                for index in term.dummies
                if index not in retained_indices
            )
            result_terms.append(
                Term(
                    term.coefficient * expected.coefficient,
                    combined_factors + vanished_sums,
                )
            )
    return Expression.from_terms(result_terms)


def reference_expectation_cache_clear() -> None:
    """Clear all terminal-expansion and reference-reduction caches."""

    expand_two_body_operators.cache_clear()
    _expect_one_body_word.cache_clear()
    _reference_expectation_word.cache_clear()


def reference_expectation_cache_info() -> dict[str, object]:
    """Return cache statistics for each reference-reduction layer."""

    return {
        "expanded_words": expand_two_body_operators.cache_info(),
        "one_body_words": _expect_one_body_word.cache_info(),
        "operator_words": _reference_expectation_word.cache_info(),
    }
