"""Tests for initial SymPy and LaTeX conversion."""

from __future__ import annotations

from fractions import Fraction

from sympy import conjugate

from manybody_lib.symbolic.closed_shell import (
    ExactScalar,
    Expression,
    OneBodyOperator,
    OperatorWord,
    TensorSymbol,
    Term,
    expression_to_latex,
    occupied,
    virtual,
)
from manybody_lib.symbolic.closed_shell.sympy_bridge import to_sympy


def test_sympy_conversion_is_exact_and_preserves_conjugation() -> None:
    c1 = TensorSymbol("C1", rank=2, upper_count=1)
    a = virtual("a")
    i = occupied("i")
    tensor = c1(a, i, conjugated=True)
    term = Term(
        ExactScalar(Fraction(1, 2), Fraction(1, 3)),
        (tensor,),
    )

    converted = to_sympy(term)
    tensor_element = to_sympy(c1(a, i))
    assert converted.has(conjugate(tensor_element, evaluate=False))
    assert converted.as_coefficient(to_sympy(tensor)) == to_sympy(
        ExactScalar(Fraction(1, 2), Fraction(1, 3))
    )


def test_sympy_operator_word_remains_noncommutative_and_ordered() -> None:
    a = virtual("a")
    b = virtual("b")
    i = occupied("i")
    j = occupied("j")
    word = OperatorWord(
        (OneBodyOperator(a, i), OneBodyOperator(b, j))
    )
    converted = to_sympy(word)
    assert not converted.is_commutative
    assert converted.args == tuple(
        to_sympy(operator) for operator in word.operators
    )


def test_latex_uses_stacked_indices_and_explicit_conjugation() -> None:
    c1 = TensorSymbol("C", rank=2, upper_count=1)
    f = TensorSymbol("f", rank=2, upper_count=1)
    a = virtual("a")
    i = occupied("i")
    expression = Expression.from_terms(
        (
            Term(factors=(c1(a, i, conjugated=True),)),
            Term(-1, factors=(f(a, i),)),
        )
    )

    assert expression_to_latex(expression) == (
        r"\left(C^{a}_{i}\right)^{*} - f^{a}_{i}"
    )
