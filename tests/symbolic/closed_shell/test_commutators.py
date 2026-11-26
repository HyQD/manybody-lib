"""Commutator algebra tests for closed-shell symbolic Milestone 2."""

from __future__ import annotations

from manybody_lib.symbolic.closed_shell import (
    Delta,
    Expression,
    OneBodyOperator,
    OperatorWord,
    TensorSymbol,
    Term,
    TwoBodyOperator,
    atomic_commutator,
    commutator,
    general,
    multiply,
    occupied,
)


def test_atomic_one_body_commutator() -> None:
    p, q, r, s = (general(name) for name in "pqrs")
    left = OneBodyOperator(p, q)
    right = OneBodyOperator(r, s)
    expected = Expression.from_terms(
        (
            Term(
                factors=(Delta(q, r),),
                operators=OperatorWord((OneBodyOperator(p, s),)),
            ),
            Term(
                -1,
                factors=(Delta(p, s),),
                operators=OperatorWord((OneBodyOperator(r, q),)),
            ),
        )
    )

    assert atomic_commutator(left, right) == expected


def test_atomic_two_body_one_body_commutator() -> None:
    p, q, r, s, t, u = (general(name) for name in "pqrstu")
    left = TwoBodyOperator(p, q, r, s)
    right = OneBodyOperator(t, u)
    expected = Expression.from_terms(
        (
            Term(
                factors=(Delta(r, t),),
                operators=OperatorWord((TwoBodyOperator(p, q, u, s),)),
            ),
            Term(
                factors=(Delta(s, t),),
                operators=OperatorWord((TwoBodyOperator(p, q, r, u),)),
            ),
            Term(
                -1,
                factors=(Delta(p, u),),
                operators=OperatorWord((TwoBodyOperator(t, q, r, s),)),
            ),
            Term(
                -1,
                factors=(Delta(q, u),),
                operators=OperatorWord((TwoBodyOperator(p, t, r, s),)),
            ),
        )
    )

    assert atomic_commutator(left, right) == expected
    assert atomic_commutator(right, left) == -expected


def test_commutator_is_antisymmetric_for_operator_words() -> None:
    p, q, r, s, t, u = (general(name) for name in "pqrstu")
    left = OperatorWord(
        (OneBodyOperator(p, q), OneBodyOperator(r, s))
    )
    right = OperatorWord((OneBodyOperator(t, u),))

    assert commutator(left, right) == -commutator(right, left)


def test_product_commutators_obey_both_leibniz_rules() -> None:
    p, q, r, s, t, u = (general(name) for name in "pqrstu")
    a = OneBodyOperator(p, q)
    b = OneBodyOperator(r, s)
    c = OneBodyOperator(t, u)
    bc = OperatorWord((b, c))
    ab = OperatorWord((a, b))

    right_rule = multiply(commutator(a, b), c) + multiply(
        b,
        commutator(a, c),
    )
    left_rule = multiply(a, commutator(b, c)) + multiply(
        commutator(a, c),
        b,
    )

    assert commutator(a, bc) == right_rule
    assert commutator(ab, c) == left_rule


def test_selected_one_body_jacobi_identity() -> None:
    p, q, r, s, t, u = (general(name) for name in "pqrstu")
    a = OneBodyOperator(p, q)
    b = OneBodyOperator(r, s)
    c = OneBodyOperator(t, u)

    jacobi = (
        commutator(a, commutator(b, c))
        + commutator(b, commutator(c, a))
        + commutator(c, commutator(a, b))
    )

    assert jacobi == Expression.zero()


def test_commutator_is_linear_for_weighted_expressions() -> None:
    p, q, r, s, t, u = (general(name) for name in "pqrstu")
    a = OneBodyOperator(p, q)
    b = OneBodyOperator(r, s)
    c = OneBodyOperator(t, u)
    combination = Expression.from_terms(
        (
            Term(2, operators=OperatorWord((a,))),
            Term(-3, operators=OperatorWord((b,))),
        )
    )

    assert commutator(combination, c) == (
        commutator(a, c).scale(2) - commutator(b, c).scale(3)
    )


def test_multiplication_freshens_independently_scoped_dummies() -> None:
    amplitude = TensorSymbol("C", rank=1)
    i = occupied("i", dummy=True)
    term = Term(
        factors=(amplitude(i),),
        operators=OperatorWord((OneBodyOperator(i, i),)),
    )

    product = multiply(term, term)
    assert product.term_count == 1
    assert len(product.terms[0].dummies) == 2


def test_unreduced_two_body_commutator_remains_exact() -> None:
    indices = tuple(general(name) for name in "pqrstuvw")
    left = TwoBodyOperator(*indices[:4])
    right = TwoBodyOperator(*indices[4:])
    expected = Expression.from_terms(
        (
            Term(operators=OperatorWord((left, right))),
            Term(-1, operators=OperatorWord((right, left))),
        )
    )

    assert atomic_commutator(left, right) == expected
