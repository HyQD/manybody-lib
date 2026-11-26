"""Optional conversion of closed-shell symbolic objects to SymPy."""

from __future__ import annotations

from functools import singledispatch

from sympy import Add, I, IndexedBase, KroneckerDelta, Mul, Rational, S, Symbol
from sympy import conjugate as sympy_conjugate

from .expressions import Expression, Term
from .indices import Index, OrbitalSpace
from .operators import OneBodyOperator, OperatorWord, TwoBodyOperator
from .scalars import ExactScalar
from .tensors import Delta, OrbitalDimension, Tensor


@singledispatch
def to_sympy(value):
    """Convert a supported internal symbolic object to an exact SymPy object."""

    raise TypeError(f"cannot convert {type(value).__name__} to SymPy")


@to_sympy.register
def _(value: ExactScalar):
    real = Rational(value.real.numerator, value.real.denominator)
    imag = Rational(value.imag.numerator, value.imag.denominator)
    return real + I * imag


@to_sympy.register
def _(index: Index):
    assumptions = {}
    if index.space is OrbitalSpace.OCCUPIED:
        assumptions["below_fermi"] = True
    elif index.space is OrbitalSpace.VIRTUAL:
        assumptions["above_fermi"] = True
    return Symbol(index.name, **assumptions)


@to_sympy.register
def _(tensor: Tensor):
    base = IndexedBase(tensor.symbol.name)
    element = base[tuple(to_sympy(index) for index in tensor.indices)]
    if tensor.conjugated:
        return sympy_conjugate(element, evaluate=False)
    return element


@to_sympy.register
def _(delta: Delta):
    return KroneckerDelta(to_sympy(delta.left), to_sympy(delta.right))


@to_sympy.register
def _(dimension: OrbitalDimension):
    labels = {
        OrbitalSpace.OCCUPIED: "n_o",
        OrbitalSpace.VIRTUAL: "n_v",
        OrbitalSpace.GENERAL: "n",
        OrbitalSpace.SPIN: "n_s",
    }
    return Symbol(labels[dimension.space], integer=True, nonnegative=True)


def _operator_base(name: str) -> IndexedBase:
    return IndexedBase(name, commutative=False)


@to_sympy.register
def _(operator: OneBodyOperator):
    return _operator_base("E")[to_sympy(operator.p), to_sympy(operator.q)]


@to_sympy.register
def _(operator: TwoBodyOperator):
    return _operator_base("e")[
        to_sympy(operator.p),
        to_sympy(operator.q),
        to_sympy(operator.r),
        to_sympy(operator.s),
    ]


@to_sympy.register
def _(word: OperatorWord):
    if not word.operators:
        return S.One
    return Mul(*(to_sympy(operator) for operator in word.operators))


@to_sympy.register
def _(term: Term):
    factors = [to_sympy(term.coefficient)]
    factors.extend(to_sympy(factor) for factor in term.factors)
    if term.operators:
        factors.append(to_sympy(term.operators))
    return Mul(*factors)


@to_sympy.register
def _(expression: Expression):
    if not expression.terms:
        return S.Zero
    return Add(*(to_sympy(term) for term in expression.terms))
