"""Closed-shell reference contraction and explicit-spin validation tests."""

from __future__ import annotations

import itertools

import pytest

from manybody_lib.symbolic.closed_shell import (
    Delta,
    ExactScalar,
    Expression,
    OneBodyOperator,
    OperatorWord,
    TensorSymbol,
    Term,
    TwoBodyOperator,
    general,
    occupied,
    reference_expectation,
    reference_expectation_cache_clear,
    reference_expectation_cache_info,
    virtual,
)
from manybody_lib.symbolic.closed_shell.explicit_spin import (
    evaluate_scalar_expression,
    explicit_spin_reference_expectation,
)
from tests.symbolic.closed_shell.determinant_oracle import (
    apply_operator_word,
    bra_ket,
    closed_shell_reference,
)


def test_elementary_reference_termination_and_pair_contraction() -> None:
    i = occupied("i")
    j = occupied("j")
    a = virtual("a")
    b = virtual("b")

    assert reference_expectation(OperatorWord()) == Expression.from_terms(
        (Term(),)
    )
    assert reference_expectation(OneBodyOperator(i, j)) == (
        Expression.from_terms((Term(2, factors=(Delta(i, j),)),))
    )
    assert not reference_expectation(OneBodyOperator(a, i))
    assert not reference_expectation(OneBodyOperator(i, a))
    assert not reference_expectation(OneBodyOperator(a, b))

    pair = OperatorWord(
        (OneBodyOperator(i, a), OneBodyOperator(b, j))
    )
    assert reference_expectation(pair) == Expression.from_terms(
        (Term(2, factors=(Delta(a, b), Delta(i, j))),)
    )


def test_occupied_two_body_reference_density() -> None:
    i, j, k, l = (occupied(name) for name in "ijkl")
    operator = TwoBodyOperator(i, j, k, l)
    expected = Expression.from_terms(
        (
            Term(4, factors=(Delta(i, k), Delta(j, l))),
            Term(-2, factors=(Delta(j, k), Delta(i, l))),
        )
    )

    assert reference_expectation(operator) == expected


def test_summed_occupied_trace_produces_an_orbital_dimension() -> None:
    i = occupied("i", dummy=True)
    expression = reference_expectation(OneBodyOperator(i, i))

    assert evaluate_scalar_expression(
        expression,
        {},
        n_spatial=5,
        n_occupied=2,
    ) == ExactScalar(4)


def test_reference_contraction_preserves_amplitude_dummy_scope() -> None:
    coefficient = TensorSymbol("C", rank=2, upper_count=1)
    i = occupied("i", dummy=True)
    a = virtual("a", dummy=True)
    term = Term(
        factors=(coefficient(a, i),),
        operators=OperatorWord(
            (OneBodyOperator(i, a), OneBodyOperator(a, i))
        ),
    )

    assert reference_expectation(term) == Expression.from_terms(
        (Term(2, factors=(coefficient(a, i),)),)
    )


def test_reference_contraction_rejects_unresolved_general_indices() -> None:
    p = general("p")
    q = general("q")
    with pytest.raises(ValueError, match="occupied/virtual-resolved"):
        reference_expectation(OneBodyOperator(p, q))


def test_reference_expectation_cache_is_reused_and_deterministic() -> None:
    i = occupied("i")
    j = occupied("j")
    a = virtual("a")
    b = virtual("b")
    word = OperatorWord(
        (
            OneBodyOperator(i, a),
            OneBodyOperator(j, b),
            OneBodyOperator(b, j),
            OneBodyOperator(a, i),
        )
    )
    reference_expectation_cache_clear()

    first = reference_expectation(word)
    before = reference_expectation_cache_info()["operator_words"]
    second = reference_expectation(word)
    after = reference_expectation_cache_info()["operator_words"]

    assert first == second
    assert after.hits == before.hits + 1


def _encoded_word(word: OperatorWord, assignment) -> tuple:
    encoded = []
    for operator in word.operators:
        label = "E" if isinstance(operator, OneBodyOperator) else "e"
        encoded.append(
            (label, tuple(assignment[index] for index in operator.indices))
        )
    return tuple(encoded)


def test_explicit_spin_backend_agrees_with_independent_oracle() -> None:
    i = occupied("i")
    j = occupied("j")
    a = virtual("a")
    b = virtual("b")
    assignment = {i: 0, j: 1, a: 2, b: 3}
    words = (
        OperatorWord((OneBodyOperator(i, i),)),
        OperatorWord(
            (OneBodyOperator(i, a), OneBodyOperator(a, i))
        ),
        OperatorWord((TwoBodyOperator(i, j, i, j),)),
        OperatorWord(
            (
                OneBodyOperator(j, b),
                TwoBodyOperator(i, b, a, j),
                OneBodyOperator(a, i),
            )
        ),
    )
    reference = closed_shell_reference(n_spatial=4, n_occupied=2)

    for word in words:
        oracle_state = apply_operator_word(
            reference,
            _encoded_word(word, assignment),
        )
        oracle_value = int(bra_ket(reference, oracle_state).real)
        assert explicit_spin_reference_expectation(
            word,
            assignment,
            n_spatial=4,
            n_occupied=2,
        ) == oracle_value


def test_all_cisd_sector_words_agree_with_explicit_spin_backend() -> None:
    i = occupied("i")
    j = occupied("j")
    a = virtual("a")
    b = virtual("b")
    assignment = {i: 0, j: 1, a: 2, b: 3}
    indices = (i, j, a, b)

    bra_words = (
        (),
        (OneBodyOperator(i, a),),
        (OneBodyOperator(j, b), OneBodyOperator(i, a)),
        (OneBodyOperator(i, b), OneBodyOperator(j, a)),
    )
    ket_words = (
        (),
        (OneBodyOperator(a, i),),
        (OneBodyOperator(a, i), OneBodyOperator(b, j)),
    )
    insertions = [()]
    insertions.extend(
        (OneBodyOperator(p, q),)
        for p, q in itertools.product(indices, repeat=2)
    )
    insertions.extend(
        (TwoBodyOperator(p, q, r, s),)
        for p, q, r, s in itertools.product(indices, repeat=4)
    )

    checked = 0
    for bra, insertion, ket in itertools.product(
        bra_words,
        insertions,
        ket_words,
    ):
        word = OperatorWord(bra + insertion + ket)
        symbolic = evaluate_scalar_expression(
            reference_expectation(word),
            assignment,
            n_spatial=4,
            n_occupied=2,
        )
        explicit = explicit_spin_reference_expectation(
            word,
            assignment,
            n_spatial=4,
            n_occupied=2,
        )
        assert symbolic == ExactScalar(explicit), word
        checked += 1

    assert checked == 3276
