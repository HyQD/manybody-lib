"""Matthews-manifold closed-shell CISDT tests for Milestone 5."""

from __future__ import annotations

from fractions import Fraction
import itertools
import json

from manybody_lib.symbolic.closed_shell import (
    MATTHEWS_LEFT_TRIPLES,
    MATTHEWS_TRIPLES,
    OperatorWord,
    OrbitalSpace,
    build_matthews_left_state,
    build_matthews_right_state,
    build_normal_ordered_hamiltonian,
    conjugate_scalar_expression,
    derive_matthews_rdms,
    derive_matthews_sigma,
    matthews_metric,
    matthews_hermitian_bra_substitution,
    matthews_null_projector,
    matthews_one_rdms_to_latex,
    matthews_sigma_to_latex,
    matthews_triples_gauge_component,
    matthews_triples_permutation_definitions,
    matthews_two_rdms_to_latex,
    occupied,
    virtual,
)
from manybody_lib.symbolic.closed_shell.explicit_spin import (
    evaluate_scalar_expression,
    explicit_spin_reference_expectation,
)
from manybody_lib.symbolic.closed_shell.manifolds.matthews import (
    matthews_covariant_projector,
)
from manybody_lib.symbolic.closed_shell.reference import reference_expectation
from tests.symbolic.closed_shell._fixture_parser import (
    FIXTURES,
    load_fixture_file,
)


def _fraction_matrix(values):
    return tuple(
        tuple(Fraction(value) for value in row)
        for row in values
    )


def test_operator_derived_matthews_metrics_match_frozen_exact_data() -> None:
    data = json.loads((FIXTURES / "matthews_metric.json").read_text())
    for rank in (2, 3):
        expected = data["ranks"][str(rank)]
        metric = matthews_metric(rank)
        assert metric.metric == _fraction_matrix(expected["metric"])
        assert metric.pseudoinverse == _fraction_matrix(
            expected["pseudoinverse"]
        )
        assert metric.null_projector == _fraction_matrix(
            expected["null_projector"]
        )
        assert metric.physical_projector == _fraction_matrix(
            expected["physical_projector"]
        )
        assert metric.eigenvalues == tuple(
            Fraction(value) for value in expected["eigenvalues"]
        )
        assert metric.matrix_rank == (2 if rank == 2 else 5)
        assert metric.nullity == (0 if rank == 2 else 1)


def test_matthews_states_have_the_documented_rank_prefactors() -> None:
    right = build_matthews_right_state()
    left = build_matthews_left_state()
    expected = (Fraction(1), Fraction(1), Fraction(1, 2), Fraction(1, 6))

    assert right.ranks == left.ranks == (0, 1, 2, 3)
    assert tuple(sector.prefactor.real for sector in right.sectors) == expected
    assert tuple(sector.prefactor.real for sector in left.sectors) == expected


def test_all_matthews_rdms_match_completed_drudge_fixtures() -> None:
    generated = derive_matthews_rdms()
    fixture_norm = load_fixture_file(
        "matthews_cisdt_one_rdm.json",
        "norm",
    )
    fixture_one = load_fixture_file(
        "matthews_cisdt_one_rdm.json",
        "one_rdm",
    )
    fixture_two = load_fixture_file(
        "matthews_cisdt_two_rdm.json",
        "two_rdm",
    )

    assert generated.norm == next(iter(fixture_norm.values()))
    assert {block.label for block in generated.one_body} == set(fixture_one)
    assert {block.label for block in generated.two_body} == set(fixture_two)
    for block in generated.one_body:
        assert block.expression == fixture_one[block.label]
    for block in generated.two_body:
        assert block.expression == fixture_two[block.label]

    assert len(generated.measurements) == 15
    for measurement in generated.measurements:
        assert measurement.seconds >= 0
        assert measurement.peak_bytes >= 0
        assert measurement.term_count > 0

    hermitian_norm = matthews_hermitian_bra_substitution(generated.norm)
    assert hermitian_norm == conjugate_scalar_expression(hermitian_norm)
    gamma_ai = matthews_hermitian_bra_substitution(
        generated.block(1, r"\gamma^{a}_{i}").expression
    )
    gamma_ia = matthews_hermitian_bra_substitution(
        generated.block(1, r"\gamma^{i}_{a}").expression
    )
    assert gamma_ai == conjugate_scalar_expression(gamma_ia)

    one_latex = matthews_one_rdms_to_latex(generated)
    two_latex = matthews_two_rdms_to_latex(generated)
    assert r"\langle\widetilde\Psi\vert\Psi\rangle" in one_latex
    assert r"\gamma^{a}_{i}" in one_latex
    assert r"\Gamma^{ab}_{ij}" in two_latex


def test_rdm_results_are_invariant_under_both_triples_null_gauges() -> None:
    rdms = derive_matthews_rdms()
    expressions = [rdms.norm]
    expressions.extend(block.expression for block in rdms.one_body)
    expressions.extend(block.expression for block in rdms.two_body)

    for expression, amplitude in itertools.product(
        expressions,
        (MATTHEWS_TRIPLES, MATTHEWS_LEFT_TRIPLES),
    ):
        assert not matthews_triples_gauge_component(
            expression,
            amplitude,
            component="null",
        )


def test_complete_sigma_is_metric_resolved_null_free_and_profiled() -> None:
    sigma = derive_matthews_sigma()
    assert sigma.resolved.ranks == (0, 1, 2, 3)
    term_counts = tuple(
        component.term_count
        for _, component in sigma.resolved.components
    )
    assert term_counts == (4, 16, 37, 294)
    assert len(sigma.measurements) == 4
    for measurement in sigma.measurements:
        assert measurement.seconds >= 0
        assert measurement.peak_bytes >= 0
        assert measurement.term_count > 0

    i, j, k = (occupied(name) for name in "ijk")
    assert not matthews_null_projector(sigma[3], (i, j, k))
    for rank in range(4):
        assert sigma.presentation(rank).expand() == sigma[rank]
        assert not matthews_triples_gauge_component(
            sigma[rank],
            MATTHEWS_TRIPLES,
            component="null",
        )

    factored = matthews_sigma_to_latex(sigma)
    expanded = matthews_sigma_to_latex(sigma, factor_permutations=False)
    selected = matthews_sigma_to_latex(sigma, ranks=(0, 2, 3))
    triples_presentation = sigma.column_presentation(3)
    assert triples_presentation.expand() == sigma[3]
    assert triples_presentation.displayed_term_count == 59
    assert factored.startswith(r"\begin{align}")
    assert factored.endswith(r"\end{align}")
    assert r"\widehat{\mathcal S}^{abc}_{ijk}" in factored
    assert r"\widehat{\mathcal S}^{abc}_{ijk}" not in expanded
    assert r"\sigma^{a}_{i}" not in selected
    assert r"\sigma^{ab}_{ij}" in selected
    definitions = matthews_triples_permutation_definitions()
    assert r"\widehat{\mathcal P}^{c/ab}_{k/ij}" in definitions


def _assignment(word: OperatorWord, *, repeated: bool):
    occupied_indices = sorted(
        {
            index
            for index in word.indices
            if index.space is OrbitalSpace.OCCUPIED
        },
        key=lambda index: index.sort_key(),
    )
    virtual_indices = sorted(
        {
            index
            for index in word.indices
            if index.space is OrbitalSpace.VIRTUAL
        },
        key=lambda index: index.sort_key(),
    )
    if repeated:
        n_occupied = 2
        n_virtual = 2
        occupied_values = (position % 2 for position in range(len(occupied_indices)))
        virtual_values = (
            n_occupied + position % 2
            for position in range(len(virtual_indices))
        )
    else:
        n_occupied = max(1, len(occupied_indices))
        n_virtual = max(1, len(virtual_indices))
        occupied_values = iter(range(len(occupied_indices)))
        virtual_values = (
            n_occupied + position
            for position in range(len(virtual_indices))
        )
    assignment = dict(zip(occupied_indices, occupied_values, strict=True))
    assignment.update(zip(virtual_indices, virtual_values, strict=True))
    return assignment, n_occupied + n_virtual, n_occupied


def _free_component_word(word: OperatorWord) -> OperatorWord:
    """Replace Einstein-summed labels by distinct external component labels."""

    substitutions = {}
    counts = {OrbitalSpace.OCCUPIED: 0, OrbitalSpace.VIRTUAL: 0}
    factories = {
        OrbitalSpace.OCCUPIED: occupied,
        OrbitalSpace.VIRTUAL: virtual,
    }
    for index in sorted(set(word.indices), key=lambda item: item.sort_key()):
        position = counts[index.space]
        counts[index.space] += 1
        prefix = "o" if index.space is OrbitalSpace.OCCUPIED else "v"
        substitutions[index] = factories[index.space](f"{prefix}{position}")
    return word.replace_indices(substitutions)


def test_every_triples_sigma_operator_word_matches_explicit_spin() -> None:
    """Independently check all words behind the formerly unfinished sector."""

    projector = matthews_covariant_projector(3).terms[0].operators
    hamiltonian = build_normal_ordered_hamiltonian()
    right = build_matthews_right_state()
    checked = 0
    for hamiltonian_term, ket_sector in itertools.product(
        hamiltonian,
        right.sectors[1:],
    ):
        for ket_term in ket_sector.expression:
            word = _free_component_word(
                OperatorWord(
                    projector.operators
                    + hamiltonian_term.operators.operators
                    + ket_term.operators.operators
                )
            )
            symbolic_expression = reference_expectation(word)
            for repeated in (False, True):
                assignment, n_spatial, n_occupied = _assignment(
                    word,
                    repeated=repeated,
                )
                symbolic = evaluate_scalar_expression(
                    symbolic_expression,
                    assignment,
                    n_spatial=n_spatial,
                    n_occupied=n_occupied,
                )
                explicit = explicit_spin_reference_expectation(
                    word,
                    assignment,
                    n_spatial=n_spatial,
                    n_occupied=n_occupied,
                )
                assert symbolic.real == explicit
                assert symbolic.imag == 0
            checked += 1

    assert checked == 96
