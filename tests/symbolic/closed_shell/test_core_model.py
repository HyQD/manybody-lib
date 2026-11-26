"""Unit tests for the Milestone 1 closed-shell symbolic data model."""

from __future__ import annotations

from fractions import Fraction

import pytest

from manybody_lib.symbolic.closed_shell import (
    ExactScalar,
    IndexRole,
    OneBodyOperator,
    OperatorWord,
    OrbitalSpace,
    TensorSymbol,
    TwoBodyOperator,
    general,
    occupied,
    spin,
    virtual,
)


def test_index_identity_contains_space_and_role() -> None:
    assert occupied("p") != virtual("p")
    assert general("p") != general("p", dummy=True)
    assert occupied("i").space is OrbitalSpace.OCCUPIED
    assert occupied("i", dummy=True).role is IndexRole.DUMMY
    assert spin("sigma").space is OrbitalSpace.SPIN


def test_exact_gaussian_rational_arithmetic() -> None:
    value = ExactScalar(Fraction(1, 2), Fraction(2, 3))
    other = ExactScalar(Fraction(3, 5), Fraction(-1, 7))

    assert value + other == ExactScalar(Fraction(11, 10), Fraction(11, 21))
    assert value * other == ExactScalar(Fraction(83, 210), Fraction(23, 70))
    assert value.conjugate() == ExactScalar(Fraction(1, 2), Fraction(-2, 3))
    assert value * 0 == ExactScalar(0)

    with pytest.raises(TypeError, match="exact rational"):
        ExactScalar(0.5)


def test_tensor_and_operator_valence_validation() -> None:
    f = TensorSymbol("f", rank=2, upper_count=1)
    with pytest.raises(ValueError, match="expects 2 indices"):
        f(occupied("i"))

    with pytest.raises(TypeError, match="operator indices"):
        OneBodyOperator(occupied("i"), "a")


def test_operator_words_preserve_noncommuting_order() -> None:
    a = virtual("a")
    b = virtual("b")
    i = occupied("i")
    j = occupied("j")
    single = OneBodyOperator(a, i)
    double = TwoBodyOperator(a, b, i, j)

    word = OperatorWord((single,)) * double
    assert word.operators == (single, double)
    assert (OperatorWord((double,)) * single).operators == (double, single)
