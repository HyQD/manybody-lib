"""Translate frozen Drudge term records into the internal symbolic model."""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

from sympy import Indexed, IndexedBase, KroneckerDelta, Mul, Symbol, sympify

from manybody_lib.symbolic.closed_shell import (
    Delta,
    Expression,
    FOCK,
    REFERENCE_ENERGY,
    TWO_ELECTRON,
    Term,
    occupied,
    virtual,
)
from manybody_lib.symbolic.closed_shell.manifolds.hjo import (
    DOUBLES_AMPLITUDE,
    LEFT_DOUBLES_AMPLITUDE,
    LEFT_REFERENCE_COEFFICIENT,
    LEFT_SINGLES_AMPLITUDE,
    REFERENCE_COEFFICIENT,
    SINGLES_AMPLITUDE,
)


FIXTURES = Path(__file__).with_name("fixtures")
_OCCUPIED_NAMES = frozenset("ijklmn")
_VIRTUAL_NAMES = frozenset("abcdefgh")


def _fixture_index(name: str, sum_spaces: dict[str, str]):
    if name in sum_spaces:
        space = sum_spaces[name]
        if space == "O":
            return occupied(name, dummy=True)
        if space == "V":
            return virtual(name, dummy=True)
        raise ValueError(f"unknown fixture range {space!r}")
    if name[0] in _OCCUPIED_NAMES:
        return occupied(name)
    if name[0] in _VIRTUAL_NAMES:
        return virtual(name)
    raise ValueError(f"cannot infer the orbital space of {name!r}")


def fixture_expression(record: dict) -> Expression:
    """Convert one fixture equation record to a canonical expression."""

    indexed_bases = {
        name: IndexedBase(name)
        for name in ("C", "Ctilde", "f", "u")
    }
    local_symbols = {
        **indexed_bases,
        "C0": Symbol("C0"),
        "C0tilde": Symbol("C0tilde"),
        "Eref": Symbol("Eref"),
        "KroneckerDelta": KroneckerDelta,
    }
    terms = []
    for fixture_term in record["terms"]:
        sum_spaces = {
            item["index"]: item["range"]
            for item in fixture_term["sums"]
        }
        parsed = sympify(fixture_term["amplitude"], locals=local_symbols)
        coefficient, remainder = parsed.as_coeff_Mul()
        factors = []
        for factor in Mul.make_args(remainder):
            if isinstance(factor, Indexed):
                label = str(factor.base.label)
                indices = tuple(
                    _fixture_index(str(index), sum_spaces)
                    for index in factor.indices
                )
                if label == "C" and len(indices) == 2:
                    factors.append(SINGLES_AMPLITUDE(*indices))
                elif label == "C" and len(indices) == 4:
                    factors.append(DOUBLES_AMPLITUDE(*indices))
                elif label == "Ctilde" and len(indices) == 2:
                    factors.append(LEFT_SINGLES_AMPLITUDE(*indices))
                elif label == "Ctilde" and len(indices) == 4:
                    factors.append(LEFT_DOUBLES_AMPLITUDE(*indices))
                elif label == "f":
                    factors.append(FOCK(*indices))
                elif label == "u":
                    factors.append(TWO_ELECTRON(*indices))
                else:
                    raise ValueError(f"unknown fixture tensor {label!r}")
            elif isinstance(factor, KroneckerDelta):
                left, right = (
                    _fixture_index(str(index), sum_spaces)
                    for index in factor.args
                )
                factors.append(Delta(left, right))
            elif factor == local_symbols["C0"]:
                factors.append(REFERENCE_COEFFICIENT())
            elif factor == local_symbols["C0tilde"]:
                factors.append(LEFT_REFERENCE_COEFFICIENT())
            elif factor == local_symbols["Eref"]:
                factors.append(REFERENCE_ENERGY())
            else:
                raise TypeError(f"unsupported fixture factor {factor!r}")
        terms.append(
            Term(
                Fraction(int(coefficient.p), int(coefficient.q)),
                tuple(factors),
            )
        )
    return Expression.from_terms(terms)


def load_fixture_section(method: str, section: str) -> dict[str, Expression]:
    """Load and translate one named HJO fixture section."""

    data = json.loads((FIXTURES / f"hjo_{method}.json").read_text())
    return {
        label: fixture_expression(record)
        for label, record in data["expressions"][section].items()
    }


def load_right_sigma(method: str) -> dict[int, Expression]:
    """Load right sigma components keyed by excitation rank."""

    result = {}
    for label, expression in load_fixture_section(method, "right_sigma").items():
        if label == r"\sigma_{0}":
            rank = 0
        elif "^{ab}" in label:
            rank = 2
        else:
            rank = 1
        result[rank] = expression
    return result
