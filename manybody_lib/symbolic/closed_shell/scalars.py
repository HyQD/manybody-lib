"""Exact scalar coefficients for the closed-shell symbolic core."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Rational
from typing import TypeAlias


RationalLike: TypeAlias = Integral | Rational | Fraction


def _fraction(value: RationalLike) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, Rational):
        return Fraction(value.numerator, value.denominator)
    raise TypeError(f"expected an exact rational, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class ExactScalar:
    """A Gaussian rational ``real + i*imag``.

    Floating-point construction is deliberately rejected.  Symbolic complex
    quantities belong in tensor factors and explicit conjugation metadata;
    numerical prefactors in the algebra remain exact.
    """

    real: Fraction = Fraction(0)
    imag: Fraction = Fraction(0)

    def __init__(
        self,
        real: RationalLike = 0,
        imag: RationalLike = 0,
    ) -> None:
        object.__setattr__(self, "real", _fraction(real))
        object.__setattr__(self, "imag", _fraction(imag))

    @classmethod
    def coerce(cls, value: ExactScalar | RationalLike) -> ExactScalar:
        if isinstance(value, cls):
            return value
        return cls(value)

    @property
    def is_zero(self) -> bool:
        return self.real == 0 and self.imag == 0

    @property
    def is_one(self) -> bool:
        return self.real == 1 and self.imag == 0

    @property
    def is_real(self) -> bool:
        return self.imag == 0

    def conjugate(self) -> ExactScalar:
        return ExactScalar(self.real, -self.imag)

    def __bool__(self) -> bool:
        return not self.is_zero

    def __neg__(self) -> ExactScalar:
        return ExactScalar(-self.real, -self.imag)

    def __add__(self, other: ExactScalar | RationalLike) -> ExactScalar:
        other = ExactScalar.coerce(other)
        return ExactScalar(self.real + other.real, self.imag + other.imag)

    def __radd__(self, other: ExactScalar | RationalLike) -> ExactScalar:
        return self + other

    def __sub__(self, other: ExactScalar | RationalLike) -> ExactScalar:
        return self + -ExactScalar.coerce(other)

    def __rsub__(self, other: ExactScalar | RationalLike) -> ExactScalar:
        return ExactScalar.coerce(other) - self

    def __mul__(self, other: ExactScalar | RationalLike) -> ExactScalar:
        other = ExactScalar.coerce(other)
        return ExactScalar(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def __rmul__(self, other: ExactScalar | RationalLike) -> ExactScalar:
        return self * other


ZERO = ExactScalar(0)
ONE = ExactScalar(1)
MINUS_ONE = ExactScalar(-1)
I = ExactScalar(0, 1)
