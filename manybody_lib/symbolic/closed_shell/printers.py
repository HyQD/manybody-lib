"""Deterministic presentation helpers for closed-shell expressions."""

from __future__ import annotations

from fractions import Fraction
from itertools import count
from typing import Iterable, Mapping

from .expressions import Expression, Term
from .indices import Index, OrbitalSpace
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


_PRESENTATION_INDEX_BASES = {
    OrbitalSpace.OCCUPIED: tuple("ijklmn"),
    OrbitalSpace.VIRTUAL: tuple("abcdef"),
    OrbitalSpace.GENERAL: tuple("pqrstu"),
    OrbitalSpace.SPIN: (r"\sigma", r"\tau"),
}


def _primed_index_name(base: str, order: int) -> str:
    if order == 0:
        return base
    primes = r"\prime" * order
    return rf"{base}^{{{primes}}}"


def _presentation_index_names(space: OrbitalSpace):
    bases = _PRESENTATION_INDEX_BASES[space]
    for prime_order in count():
        for base in bases:
            yield _primed_index_name(base, prime_order)


def _readable_dummy_names(
    term: Term,
    reserved_indices: Iterable[Index],
) -> dict[Index, str]:
    """Choose conventional collision-free display names for one term."""

    used = {space: set() for space in OrbitalSpace}
    for index in (*tuple(reserved_indices), *term.indices):
        if index.is_free:
            used[index.space].add(index.name)

    names = {}
    for index in term.dummies:
        candidates = _presentation_index_names(index.space)
        name = next(
            candidate
            for candidate in candidates
            if candidate not in used[index.space]
        )
        names[index] = name
        used[index.space].add(name)
    return names


def _index_latex(index: Index, names: Mapping[Index, str]) -> str:
    return names.get(index, index.name)


def _tensor_latex(tensor: Tensor, names: Mapping[Index, str]) -> str:
    symbol = tensor.symbol
    label = symbol.latex_name or symbol.name
    upper = "".join(
        _index_latex(index, names)
        for index in tensor.indices[: symbol.upper_count]
    )
    lower = "".join(
        _index_latex(index, names)
        for index in tensor.indices[symbol.upper_count :]
    )
    if upper:
        rendered = rf"{label}^{{{upper}}}"
    else:
        rendered = label
    if lower:
        rendered += rf"_{{{lower}}}"
    if tensor.conjugated:
        rendered = rf"\left({rendered}\right)^{{*}}"
    return rendered


def _factor_latex(factor, names: Mapping[Index, str]) -> str:
    if isinstance(factor, Tensor):
        return _tensor_latex(factor, names)
    if isinstance(factor, Delta):
        left = _index_latex(factor.left, names)
        right = _index_latex(factor.right, names)
        return rf"\delta_{{{left}{right}}}"
    if isinstance(factor, OrbitalDimension):
        labels = {
            OrbitalSpace.OCCUPIED: r"n_{\mathrm{o}}",
            OrbitalSpace.VIRTUAL: r"n_{\mathrm{v}}",
            OrbitalSpace.GENERAL: "n",
            OrbitalSpace.SPIN: r"n_{\mathrm{s}}",
        }
        return labels[factor.space]
    raise TypeError(f"cannot print {type(factor).__name__}")


def _operator_latex(operator, names: Mapping[Index, str]) -> str:
    indices = "".join(_index_latex(index, names) for index in operator.indices)
    if isinstance(operator, OneBodyOperator):
        return rf"E_{{{indices}}}"
    if isinstance(operator, TwoBodyOperator):
        return rf"e_{{{indices}}}"
    raise TypeError(f"cannot print {type(operator).__name__}")


def _term_product_latex(
    term: Term,
    coefficient: ExactScalar,
    names: Mapping[Index, str],
) -> str:
    pieces = [_factor_latex(factor, names) for factor in term.factors]
    pieces.extend(
        _operator_latex(operator, names)
        for operator in term.operators.operators
    )
    if not pieces or not coefficient.is_one:
        pieces.insert(0, _scalar_latex(coefficient))
    return " ".join(pieces)


def term_to_latex(
    term: Term,
    *,
    rename_dummies: bool = False,
    reserved_indices: Iterable[Index] = (),
) -> str:
    """Return standalone LaTeX for one term."""

    names = (
        _readable_dummy_names(term, reserved_indices)
        if rename_dummies
        else {}
    )
    return _term_product_latex(term, term.coefficient, names)


def expression_to_latex(
    expression: Expression,
    *,
    rename_dummies: bool = False,
    reserved_indices: Iterable[Index] = (),
) -> str:
    """Return deterministic Einstein-summation LaTeX for an expression."""

    if not expression.terms:
        return "0"
    reserved_indices = tuple(reserved_indices)
    rendered = []
    for position, term in enumerate(expression.terms):
        names = (
            _readable_dummy_names(term, reserved_indices)
            if rename_dummies
            else {}
        )
        coefficient = term.coefficient
        if coefficient.is_real and coefficient.real < 0:
            body = _term_product_latex(term, -coefficient, names)
            rendered.append(("-" if position == 0 else " - ") + body)
        else:
            body = _term_product_latex(term, coefficient, names)
            rendered.append(("" if position == 0 else " + ") + body)
    return "".join(rendered)
