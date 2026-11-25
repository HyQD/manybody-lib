"""Verified compact presentation for HJO tensor equations.

Drudge derives and simplifies fully expanded tensors.  The helpers in this
module perform a separate presentation pass that groups pairs related by the
simultaneous exchange

    (upper_1, lower_1) <-> (upper_2, lower_2)

into the unnormalised column symmetrizer

    P A = A + swapped(A).

The same presentation layer recognizes the closed-shell Coulomb-exchange
combination ``L[p, q, r, s] = 2*u[p, q, r, s] - u[p, q, s, r]``.

Every compact representation is expanded and compared with the original
Drudge tensor before it is returned.  Terms that are already invariant under
the column exchange are deliberately left outside P, since applying P to such
a term would multiply it by two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from drudge.term import Term
from sympy import Indexed, IndexedBase, Mul, Rational, Symbol


@dataclass(frozen=True)
class PermutationPresentation:
    """An expanded tensor split into explicit and permutation-paired terms."""

    original: object
    explicit: object
    representative: object
    operator_latex: str

    @property
    def n_terms(self) -> int:
        """Return the term count of the original expanded tensor."""

        return self.original.n_terms


def _tensor_latex_lines(dr, tensor, *, multiline: bool) -> list[str]:
    """Format a tensor into the right-hand-side lines used by ``align``."""

    rendered = dr.format_latex(
        tensor,
        no_sum=True,
        sep_lines=multiline,
    ).strip()
    return rendered.split(r" \\ ") if multiline else [rendered]


def latex_rhs_lines(dr, expression) -> list[str]:
    """Format either an expanded tensor or a compact permutation form."""

    if not isinstance(expression, PermutationPresentation):
        return _tensor_latex_lines(
            dr,
            expression,
            multiline=expression.n_terms > 4,
        )

    displayed_terms = (
        expression.explicit.n_terms + expression.representative.n_terms
    )
    multiline = displayed_terms > 4
    lines: list[str] = []

    if expression.explicit.n_terms:
        lines.extend(
            _tensor_latex_lines(
                dr,
                expression.explicit,
                multiline=multiline,
            )
        )

    if expression.representative.n_terms:
        permutation_lines = _tensor_latex_lines(
            dr,
            expression.representative,
            multiline=multiline,
        )
        prefix = expression.operator_latex + r"\Bigl[ "
        if lines:
            prefix = "+ " + prefix
        permutation_lines[0] = prefix + permutation_lines[0]
        permutation_lines[-1] += r" \Bigr]"
        lines.extend(permutation_lines)

    return lines


def _split_numeric_coefficient(term: Term) -> tuple[object, Term]:
    """Split a term into its numerical coefficient and monic tensor part."""

    coefficient, amplitude = term.amp.as_coeff_Mul()
    monic = term.map(lambda expr: expr, amp=amplitude)
    return coefficient, monic


def _canonical_substitution(dr, term: Term, substitutions: Mapping) -> Term:
    """Apply a free-index exchange and restore Drudge canonical form."""

    result = term.subst(substitutions)
    return _canonical_term(dr, result)


def _canonical_term(dr, term: Term) -> Term:
    """Restore Drudge canonical form for a locally modified term."""

    result = term
    result = result.canon(
        symms=dr.symms.value,
        vec_colour=dr.vec_colour,
    )
    return result.reset_dumms(
        dumms=dr.dumms.value,
        excl=result.free_vars,
    )[0]


def _indexed_factor(term: Term, label: str) -> Indexed | None:
    """Return the unique top-level indexed factor with the given label."""

    matches = [
        factor
        for factor in Mul.make_args(term.amp)
        if isinstance(factor, Indexed)
        and str(factor.base.label) == label
    ]
    return matches[0] if len(matches) == 1 else None


def _replace_factor(term: Term, old: Indexed, new) -> Term:
    """Replace one top-level amplitude factor without changing the sums."""

    amplitude = term.amp.xreplace({old: new})
    return term.map(lambda expr: expr, amp=amplitude)


def _expand_exchange_integrals(
    term: Term,
    *,
    integral_base: IndexedBase,
    exchange_base: IndexedBase,
) -> list[Term]:
    """Expand all L factors in one term back into two-electron integrals."""

    substitutions = {}
    for factor in term.amp.atoms(Indexed):
        if factor.base != exchange_base or len(factor.indices) != 4:
            continue
        p, q, r, s = factor.indices
        substitutions[factor] = (
            2 * integral_base[p, q, r, s]
            - integral_base[p, q, s, r]
        )

    if not substitutions:
        return [term]
    expanded = term.map(
        lambda expr: expr,
        amp=term.amp.xreplace(substitutions),
    )
    return expanded.expand()


def factor_exchange_integrals(
    dr,
    tensor,
    *,
    integral_base: IndexedBase,
    exchange_base: IndexedBase,
):
    """Replace exact ``2*u[pqrs] - u[pqsr]`` pairs by ``L[pqrs]``.

    The common prefactor, summations, and all other indexed factors must match
    after normal Drudge canonicalization.  Unmatched terms remain unchanged.
    The compact tensor is re-expanded through the definition of ``L`` and
    required to reproduce the input tensor exactly.
    """

    coefficients: dict[Term, object] = {}
    for term in tensor.local_terms:
        coefficient, monic = _split_numeric_coefficient(term)
        coefficients[monic] = coefficients.get(monic, 0) + coefficient

    coefficients = {
        term: coefficient
        for term, coefficient in coefficients.items()
        if coefficient != 0
    }

    compact_terms: list[Term] = []
    unvisited = set(coefficients)

    for monic in sorted(coefficients, key=lambda term: term.sort_key):
        if monic not in unvisited:
            continue

        coefficient = coefficients[monic]
        integral = _indexed_factor(monic, str(integral_base.label))
        if integral is None or len(integral.indices) != 4:
            compact_terms.append(monic.scale(coefficient))
            unvisited.remove(monic)
            continue

        p, q, r, s = integral.indices
        exchanged_integral = integral_base[p, q, s, r]
        exchanged = _replace_factor(monic, integral, exchanged_integral)
        exchanged = _canonical_term(dr, exchanged)
        exchange_coefficient, exchanged_monic = (
            _split_numeric_coefficient(exchanged)
        )
        paired_coefficient = coefficients.get(exchanged_monic)

        expected_pair = (
            -Rational(1, 2) * coefficient * exchange_coefficient
        )
        if (
            exchanged_monic != monic
            and exchanged_monic in unvisited
            and paired_coefficient == expected_pair
        ):
            exchange_integral = exchange_base[p, q, r, s]
            compact = _replace_factor(monic, integral, exchange_integral)
            compact = _canonical_term(dr, compact)
            compact_terms.append(compact.scale(coefficient / 2))
            unvisited.remove(monic)
            unvisited.remove(exchanged_monic)
        else:
            compact_terms.append(monic.scale(coefficient))
            unvisited.remove(monic)

    compact_tensor = dr.create_tensor(compact_terms)
    if compact_terms:
        compact_tensor = compact_tensor.simplify()

    expanded = compact_tensor.bind(
        lambda term: _expand_exchange_integrals(
            term,
            integral_base=integral_base,
            exchange_base=exchange_base,
        )
    ).simplify()
    difference = (tensor - expanded).simplify()
    if difference.n_terms != 0:
        raise AssertionError(
            "Exchange-integral presentation failed symbolic re-expansion "
            "check"
        )

    return compact_tensor


def _swapped_tensor(dr, tensor, substitutions: Mapping):
    """Return a simplified tensor with the external exchange applied."""

    return tensor.map(lambda term: term.subst(substitutions)).simplify()


def factor_column_permutation(
    dr,
    tensor,
    *,
    first_upper: Symbol,
    second_upper: Symbol,
    first_lower: Symbol,
    second_lower: Symbol,
    operator_latex: str,
) -> PermutationPresentation:
    """Factor matched simultaneous-exchange pairs from a Drudge tensor.

    Only size-two exchange orbits with exactly matching coefficients are
    factored.  Size-one orbits, unmatched terms, and unequal pairs remain in
    the explicit part.  The returned presentation is accepted only after its
    full expansion simplifies to the original tensor.
    """

    substitutions = {
        first_upper: second_upper,
        second_upper: first_upper,
        first_lower: second_lower,
        second_lower: first_lower,
    }

    # The input tensors have already been simplified by their derivations, but
    # accumulating by monic term makes the presentation pass robust to a
    # future caller supplying multiple equal terms.
    coefficients: dict[Term, object] = {}
    for term in tensor.local_terms:
        coefficient, monic = _split_numeric_coefficient(term)
        coefficients[monic] = coefficients.get(monic, 0) + coefficient

    coefficients = {
        term: coefficient
        for term, coefficient in coefficients.items()
        if coefficient != 0
    }

    explicit_terms: list[Term] = []
    representative_terms: list[Term] = []
    unvisited = set(coefficients)

    for monic in sorted(coefficients, key=lambda term: term.sort_key):
        if monic not in unvisited:
            continue

        coefficient = coefficients[monic]
        swapped = _canonical_substitution(dr, monic, substitutions)
        swap_coefficient, swapped_monic = _split_numeric_coefficient(swapped)
        paired_coefficient = coefficients.get(swapped_monic)

        is_distinct_pair = swapped_monic != monic
        coefficients_match = (
            paired_coefficient == coefficient * swap_coefficient
        )

        if (
            is_distinct_pair
            and swapped_monic in unvisited
            and coefficients_match
        ):
            representative_terms.append(monic.scale(coefficient))
            unvisited.remove(monic)
            unvisited.remove(swapped_monic)
        else:
            explicit_terms.append(monic.scale(coefficient))
            unvisited.remove(monic)

    explicit = dr.create_tensor(explicit_terms)
    if explicit_terms:
        explicit = explicit.simplify()
    representative = dr.create_tensor(representative_terms)
    if representative_terms:
        representative = representative.simplify()
    expanded = explicit + representative
    if representative.n_terms:
        expanded += _swapped_tensor(dr, representative, substitutions)

    difference = (tensor - expanded).simplify()
    if difference.n_terms != 0:
        raise AssertionError(
            "Permutation presentation failed symbolic re-expansion check"
        )

    return PermutationPresentation(
        original=tensor,
        explicit=explicit,
        representative=representative,
        operator_latex=operator_latex,
    )
