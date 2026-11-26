"""Deterministic presentation helpers for closed-shell expressions."""

from __future__ import annotations

from fractions import Fraction

from .expressions import Expression, Term
from .indices import OrbitalSpace
from .operators import OneBodyOperator, TwoBodyOperator
from .scalars import ExactScalar
from .tensors import Delta, OrbitalDimension, Tensor


def _fraction_latex(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return rf"\frac{{{value.numerator}}}{{{value.denominator}}}"


def _scalar_latex(value: ExactScalar) -> str:
    if value.imag == 0:
        return _fraction_latex(value.real)
    if value.real == 0:
        if value.imag == 1:
            return "i"
        if value.imag == -1:
            return "-i"
        return _fraction_latex(value.imag) + "i"
    sign = "+" if value.imag > 0 else "-"
    magnitude = abs(value.imag)
    imag = "i" if magnitude == 1 else _fraction_latex(magnitude) + "i"
    return rf"\left({_fraction_latex(value.real)} {sign} {imag}\right)"


def _tensor_latex(tensor: Tensor) -> str:
    symbol = tensor.symbol
    label = symbol.latex_name or symbol.name
    upper = "".join(index.name for index in tensor.indices[: symbol.upper_count])
    lower = "".join(index.name for index in tensor.indices[symbol.upper_count :])
    if upper:
        rendered = rf"{label}^{{{upper}}}"
    else:
        rendered = label
    if lower:
        rendered += rf"_{{{lower}}}"
    if tensor.conjugated:
        rendered = rf"\left({rendered}\right)^{{*}}"
    return rendered


def _factor_latex(factor) -> str:
    if isinstance(factor, Tensor):
        return _tensor_latex(factor)
    if isinstance(factor, Delta):
        return rf"\delta_{{{factor.left.name}{factor.right.name}}}"
    if isinstance(factor, OrbitalDimension):
        labels = {
            OrbitalSpace.OCCUPIED: r"n_{\mathrm{o}}",
            OrbitalSpace.VIRTUAL: r"n_{\mathrm{v}}",
            OrbitalSpace.GENERAL: "n",
            OrbitalSpace.SPIN: r"n_{\mathrm{s}}",
        }
        return labels[factor.space]
    raise TypeError(f"cannot print {type(factor).__name__}")


def _operator_latex(operator) -> str:
    indices = "".join(index.name for index in operator.indices)
    if isinstance(operator, OneBodyOperator):
        return rf"E_{{{indices}}}"
    if isinstance(operator, TwoBodyOperator):
        return rf"e_{{{indices}}}"
    raise TypeError(f"cannot print {type(operator).__name__}")


def _term_product_latex(term: Term, coefficient: ExactScalar) -> str:
    pieces = [_factor_latex(factor) for factor in term.factors]
    pieces.extend(_operator_latex(operator) for operator in term.operators.operators)
    if not pieces or not coefficient.is_one:
        pieces.insert(0, _scalar_latex(coefficient))
    return " ".join(pieces)


def term_to_latex(term: Term) -> str:
    """Return standalone LaTeX for one term."""

    return _term_product_latex(term, term.coefficient)


def expression_to_latex(expression: Expression) -> str:
    """Return deterministic Einstein-summation LaTeX for an expression."""

    if not expression.terms:
        return "0"
    rendered = []
    for position, term in enumerate(expression.terms):
        coefficient = term.coefficient
        if coefficient.is_real and coefficient.real < 0:
            body = _term_product_latex(term, -coefficient)
            rendered.append(("-" if position == 0 else " - ") + body)
        else:
            body = _term_product_latex(term, coefficient)
            rendered.append(("" if position == 0 else " + ") + body)
    return "".join(rendered)
