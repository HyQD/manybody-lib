"""Explicit complex conjugation of scalar symbolic expressions."""

from __future__ import annotations

from .expressions import Expression, Term
from .tensors import Delta, OrbitalDimension, Tensor


def conjugate_scalar_expression(expression: Expression) -> Expression:
    """Return the explicit complex conjugate of a scalar expression."""

    terms = []
    for term in expression:
        if term.operators:
            raise ValueError("scalar conjugation does not accept operator words")
        factors = []
        for factor in term.factors:
            if isinstance(factor, Tensor):
                factors.append(factor.conjugate())
            elif isinstance(factor, (Delta, OrbitalDimension)):
                factors.append(factor)
            else:
                raise TypeError(f"unsupported scalar factor {type(factor).__name__}")
        terms.append(Term(term.coefficient.conjugate(), tuple(factors)))
    return Expression.from_terms(terms)
