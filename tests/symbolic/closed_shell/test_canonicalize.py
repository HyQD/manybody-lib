"""Canonicalization and exact collection tests for Milestone 1."""

from __future__ import annotations

import itertools

from manybody_lib.symbolic.closed_shell import (
    Delta,
    Expression,
    OneBodyOperator,
    OperatorWord,
    OrbitalDimension,
    OrbitalSpace,
    PermutationSymmetry,
    TensorSymbol,
    Term,
    antisymmetric_group_symmetries,
    canonicalize_term,
    factor_triples_column_permutations,
    general,
    occupied,
    permute_expression,
    simultaneous_column_symmetries,
    virtual,
)


def test_canonicalization_is_idempotent_and_dummy_name_invariant() -> None:
    f = TensorSymbol("f", rank=2, upper_count=1)
    c2 = TensorSymbol(
        "C2",
        rank=4,
        upper_count=2,
        symmetries=simultaneous_column_symmetries(2),
    )

    a0 = virtual("x", dummy=True)
    b0 = virtual("y", dummy=True)
    i0 = occupied("m", dummy=True)
    j0 = occupied("n", dummy=True)
    first = Term(
        factors=(f(a0, i0), c2(a0, b0, i0, j0)),
    )

    a1 = virtual("r", dummy=True)
    b1 = virtual("s", dummy=True)
    i1 = occupied("u", dummy=True)
    j1 = occupied("v", dummy=True)
    shuffled = Term(
        factors=(c2(b1, a1, j1, i1), f(a1, i1)),
    )

    canonical = canonicalize_term(first)
    assert canonical is not None
    assert canonical == canonicalize_term(canonical)
    assert canonical == canonicalize_term(shuffled)
    assert tuple(index.name for index in canonical.dummies) == (
        "i0",
        "i1",
        "a0",
        "a1",
    )


def test_dummy_renaming_canonicalizes_an_unresolved_graph_automorphism() -> None:
    g = TensorSymbol("g", rank=2, upper_count=1)
    first_indices = [general(name, dummy=True) for name in "abcd"]
    first = Term(
        factors=tuple(
            g(first_indices[position], first_indices[(position + 1) % 4])
            for position in range(4)
        )
    )

    second_indices = {
        old: general(new, dummy=True)
        for old, new in zip("abcd", ("w", "y", "x", "z"), strict=True)
    }
    second = Term(
        factors=tuple(
            g(second_indices[old], second_indices[new])
            for old, new in zip("abcd", "bcda", strict=True)
        )
    )

    assert canonicalize_term(first) == canonicalize_term(second)


def test_simultaneous_column_symmetry_is_not_independent_symmetry() -> None:
    c2 = TensorSymbol(
        "C2",
        rank=4,
        upper_count=2,
        symmetries=simultaneous_column_symmetries(2),
    )
    a = virtual("a")
    b = virtual("b")
    i = occupied("i")
    j = occupied("j")

    reference = canonicalize_term(Term(factors=(c2(a, b, i, j),)))
    column_swap = canonicalize_term(Term(factors=(c2(b, a, j, i),)))
    upper_only = canonicalize_term(Term(factors=(c2(b, a, i, j),)))

    assert reference == column_swap
    assert reference != upper_only


def test_antisymmetric_tensor_phase_and_repeated_index_zero() -> None:
    antisymmetry = (
        antisymmetric_group_symmetries(4, (0, 1))
        + antisymmetric_group_symmetries(4, (2, 3))
    )
    u = TensorSymbol("u", rank=4, upper_count=2, symmetries=antisymmetry)
    a = virtual("a")
    b = virtual("b")
    i = occupied("i")
    j = occupied("j")

    direct = canonicalize_term(Term(factors=(u(a, b, i, j),)))
    exchanged = canonicalize_term(Term(factors=(u(b, a, i, j),)))
    assert direct is not None and exchanged is not None
    assert direct.body_key == exchanged.body_key
    assert direct.coefficient == -exchanged.coefficient
    assert canonicalize_term(Term(factors=(u(a, a, i, j),))) is None


def test_incompatible_delta_spaces_make_a_term_zero() -> None:
    delta = Delta(occupied("i", dummy=True), virtual("a", dummy=True))
    assert canonicalize_term(Term(factors=(delta,))) is None


def test_delta_elimination_preserves_external_constraints() -> None:
    f = TensorSymbol("f", rank=2, upper_count=1)
    i = occupied("i")
    j = occupied("j")
    k = occupied("k", dummy=True)
    a = virtual("a")
    term = Term(
        factors=(Delta(i, k), Delta(k, j), f(a, k)),
    )

    canonical = canonicalize_term(term)
    assert canonical is not None
    assert Delta(i, j) in canonical.factors
    assert f(a, i) in canonical.factors
    assert not canonical.dummies


def test_dummy_only_delta_loop_produces_orbital_dimension() -> None:
    i = occupied("i", dummy=True)
    canonical = canonicalize_term(Term(factors=(Delta(i, i),)))
    assert canonical is not None
    assert canonical.factors == (OrbitalDimension(OrbitalSpace.OCCUPIED),)


def test_general_dummy_is_specialized_by_delta() -> None:
    f = TensorSymbol("f", rank=2, upper_count=1)
    p = general("p", dummy=True)
    i = occupied("i")
    a = virtual("a")
    canonical = canonicalize_term(Term(factors=(Delta(p, i), f(a, p))))
    assert canonical is not None
    assert canonical.factors == (f(a, i),)


def test_operator_dummy_indices_are_renamed_without_reordering_word() -> None:
    a = virtual("x", dummy=True)
    i = occupied("m", dummy=True)
    b = virtual("y", dummy=True)
    j = occupied("n", dummy=True)
    term = Term(
        operators=OperatorWord(
            (OneBodyOperator(a, i), OneBodyOperator(b, j))
        )
    )
    canonical = canonicalize_term(term)
    assert canonical is not None
    first, second = canonical.operators.operators
    assert first.indices == (virtual("a0", dummy=True), occupied("i0", dummy=True))
    assert second.indices == (virtual("a1", dummy=True), occupied("i1", dummy=True))


def test_expression_collects_and_cancels_exact_coefficients() -> None:
    f = TensorSymbol("f", rank=2, upper_count=1)
    a = virtual("a")
    i = occupied("i")
    term = Term(factors=(f(a, i),))

    assert Expression.from_terms((term, term.scale(2))).terms == (
        term.scale(3),
    )
    assert not Expression.from_terms((term, term.scale(-1)))


def test_triples_column_orbit_factoring_is_exact_for_all_stabilizers() -> None:
    upper = tuple(map(virtual, "abc"))
    lower = tuple(map(occupied, "ijk"))
    symbols = (
        TensorSymbol("X6", rank=6, upper_count=3),
        TensorSymbol(
            "X3",
            rank=6,
            upper_count=3,
            symmetries=(PermutationSymmetry((1, 0, 2, 4, 3, 5)),),
        ),
        TensorSymbol(
            "X2",
            rank=6,
            upper_count=3,
            symmetries=(PermutationSymmetry((1, 2, 0, 4, 5, 3)),),
        ),
    )

    for expected_size, symbol in zip((6, 3, 2), symbols, strict=True):
        seed = Expression.from_terms(
            (Term(factors=(symbol(*(upper + lower)),)),)
        )
        orbit_terms = []
        for permutation in itertools.permutations(range(3)):
            substitutions = {
                upper[position]: upper[permutation[position]]
                for position in range(3)
            }
            substitutions.update(
                {
                    lower[position]: lower[permutation[position]]
                    for position in range(3)
                }
            )
            orbit_terms.extend(permute_expression(seed, substitutions))
        orbit = Expression.from_terms(orbit_terms)
        presentation = factor_triples_column_permutations(
            orbit,
            upper_indices=upper,
            lower_indices=lower,
        )

        assert orbit.term_count == expected_size
        assert presentation.displayed_term_count == 1
        assert presentation.expand() == orbit
        assert factor_triples_column_permutations(
            orbit,
            upper_indices=upper,
            lower_indices=lower,
        ) is presentation
