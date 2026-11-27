"""Immutable terms and exactly collected symbolic expressions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .indices import Index
from .operators import IDENTITY_WORD, OperatorWord
from .scalars import ExactScalar, ONE, RationalLike
from .tensors import ScalarFactor


def _factor_indices(factor: ScalarFactor) -> tuple[Index, ...]:
    return tuple(factor.indices)


@dataclass(frozen=True, slots=True)
class Term:
    """A coefficient, commuting scalar factors, and an ordered operator word."""

    coefficient: ExactScalar = ONE
    factors: tuple[ScalarFactor, ...] = ()
    operators: OperatorWord = IDENTITY_WORD
    dummies: tuple[Index, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coefficient",
            ExactScalar.coerce(self.coefficient),
        )
        object.__setattr__(self, "factors", tuple(self.factors))
        if not isinstance(self.operators, OperatorWord):
            raise TypeError("operators must be an OperatorWord")
        dummies = {
            index
            for factor in self.factors
            for index in _factor_indices(factor)
            if index.is_dummy
        }
        dummies.update(index for index in self.operators.indices if index.is_dummy)
        object.__setattr__(
            self,
            "dummies",
            tuple(sorted(dummies, key=Index.sort_key)),
        )

    @property
    def body_key(self) -> tuple:
        """Return the coefficient-independent key of an already canonical term."""

        return self.factors, self.operators

    @property
    def indices(self) -> tuple[Index, ...]:
        return tuple(
            index
            for factor in self.factors
            for index in _factor_indices(factor)
        ) + self.operators.indices

    def with_coefficient(
        self,
        coefficient: ExactScalar | RationalLike,
    ) -> Term:
        return Term(ExactScalar.coerce(coefficient), self.factors, self.operators)

    def scale(self, coefficient: ExactScalar | RationalLike) -> Term:
        return self.with_coefficient(
            self.coefficient * ExactScalar.coerce(coefficient)
        )


@dataclass(frozen=True, slots=True)
class Expression:
    """An immutable sum of canonical, coefficient-collected terms."""

    terms: tuple[Term, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        if any(term.coefficient.is_zero for term in self.terms):
            raise ValueError("canonical expressions cannot contain zero terms")

    @classmethod
    def from_terms(cls, terms: Iterable[Term]) -> Expression:
        from .canonicalize import canonicalize_expression

        return canonicalize_expression(tuple(terms))

    @classmethod
    def zero(cls) -> Expression:
        return cls(())

    @property
    def term_count(self) -> int:
        return len(self.terms)

    def __bool__(self) -> bool:
        return bool(self.terms)

    def __iter__(self):
        return iter(self.terms)

    def __add__(self, other: Expression | Term) -> Expression:
        if isinstance(other, Term):
            other_terms = (other,)
        elif isinstance(other, Expression):
            other_terms = other.terms
        else:
            return NotImplemented
        return Expression.from_terms(self.terms + other_terms)

    def __neg__(self) -> Expression:
        return Expression(tuple(term.scale(-1) for term in self.terms))

    def __sub__(self, other: Expression | Term) -> Expression:
        if isinstance(other, Term):
            return self + other.scale(-1)
        if isinstance(other, Expression):
            return self + -other
        return NotImplemented

    def scale(self, coefficient: ExactScalar | RationalLike) -> Expression:
        coefficient = ExactScalar.coerce(coefficient)
        if coefficient.is_zero:
            return Expression.zero()
        return Expression(
            tuple(term.scale(coefficient) for term in self.terms)
        )
