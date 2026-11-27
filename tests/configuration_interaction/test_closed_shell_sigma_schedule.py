"""Regression tests for exact reuse of closed-shell sigma contractions."""

from __future__ import annotations

import numpy as np
import pytest

from manybody_lib.configuration_interaction.closed_shell.cidt._specs import (
    SIGMA_SPECS as CIDT_SIGMA_SPECS,
)
from manybody_lib.configuration_interaction.closed_shell.cidt.sigma_vectors import (
    compute_sigma as compute_cidt_sigma,
)
from manybody_lib.configuration_interaction.closed_shell.cidt import (
    sigma_vectors_optimized as cidt_optimized,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt._contractions import (
    SIGMA_SPECS as CISDT_SIGMA_SPECS,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt._engine import (
    ContractionEvaluator,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt._schedule import (
    compile_sigma_schedule,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt.sigma_vectors import (
    compute_sigma as compute_cisdt_sigma,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt import (
    _accumulation,
    project_physical_triples,
    sigma_vectors_optimized as cisdt_optimized,
    symmetrize_triples_columns,
    triples_null_component,
)


@pytest.mark.parametrize(
    ("specs", "kernel_counts"),
    [
        (CISDT_SIGMA_SPECS, {0: 3, 1: 16, 2: 21, 3: 16}),
        (CIDT_SIGMA_SPECS, {0: 2, 2: 18, 3: 15}),
    ],
    ids=("cisdt", "cidt"),
)
def test_schedule_groups_only_equivalent_contractions(specs, kernel_counts):
    for rank, spec in specs.items():
        schedule = compile_sigma_schedule(spec)
        assert len(schedule.kernels) == kernel_counts[rank]
        assert len(
            {(kernel.operands, kernel.equation) for kernel in schedule.kernels}
        ) == len(schedule.kernels)

        terms = (
            *spec["explicit"],
            *(term for _, group in spec["groups"] for term in group),
        )
        assert sum(len(kernel.destinations) for kernel in schedule.kernels) == sum(
            bool(term[2]) for term in terms
        )
        assert len(schedule.scalars) == sum(not term[2] for term in terms)
        assert sum(schedule.group_term_counts) == len(terms)


@pytest.mark.parametrize(
    ("specs", "kernel_counts"),
    [
        (CISDT_SIGMA_SPECS, {0: 3, 1: 16, 2: 21, 3: 13}),
        (CIDT_SIGMA_SPECS, {0: 2, 2: 18, 3: 12}),
    ],
    ids=("cisdt", "cidt"),
)
def test_symmetry_schedule_reuses_expensive_triples_contractions(specs, kernel_counts):
    for rank, spec in specs.items():
        schedule = compile_sigma_schedule(spec, assume_symmetry=True)
        assert len(schedule.kernels) == kernel_counts[rank]
        reference = compile_sigma_schedule(spec)
        assert sum(len(kernel.destinations) for kernel in schedule.kernels) == sum(
            len(kernel.destinations) for kernel in reference.kernels
        )
        assert schedule.group_term_counts == reference.group_term_counts

    # Each four-term vvvv or oooo family now uses a single triples kernel.
    schedule = compile_sigma_schedule(specs[3], assume_symmetry=True)
    for block in ("vvvv", "oooo"):
        kernels = [
            kernel
            for kernel in schedule.kernels
            if kernel.operands == (("c3", ""), ("u", block))
        ]
        assert len(kernels) == 1
        assert len(kernels[0].destinations) == 4


@pytest.mark.parametrize(
    ("specs", "assume_symmetry", "kernel_counts"),
    [
        (CISDT_SIGMA_SPECS, False, {0: 2, 1: 11, 2: 18, 3: 15}),
        (CISDT_SIGMA_SPECS, True, {0: 2, 1: 11, 2: 18, 3: 12}),
        (CIDT_SIGMA_SPECS, False, {0: 1, 2: 15, 3: 14}),
        (CIDT_SIGMA_SPECS, True, {0: 1, 2: 15, 3: 11}),
    ],
    ids=("cisdt-exact", "cisdt-symmetry", "cidt-exact", "cidt-symmetry"),
)
def test_direct_exchange_schedule_combines_only_proportional_kernels(
    specs, assume_symmetry, kernel_counts
):
    l_spaces = set()
    for rank, spec in specs.items():
        schedule = compile_sigma_schedule(
            spec, assume_symmetry=assume_symmetry, combine_integrals=True
        )
        assert len(schedule.kernels) == kernel_counts[rank]
        l_spaces.update(k.operands[1][1] for k in schedule.kernels if k.use_l_integral)
        assert sum(schedule.group_term_counts) == len(schedule.scalars) + sum(
            len(kernel.destinations) for kernel in schedule.kernels
        )
    assert l_spaces == {"oovv", "ovvo", "ooov", "ovvv"}

    # The same-column mixed triples pair combines, while the different-column
    # kernels have nonproportional destination vectors and must stay separate.
    schedule = compile_sigma_schedule(
        specs[3], assume_symmetry=assume_symmetry, combine_integrals=True
    )
    mixed = [
        k for k in schedule.kernels
        if k.operands[0] == ("c3", "")
        and len(k.operands) == 2
        and k.operands[1] in (("u", "ovov"), ("u", "ovvo"))
    ]
    assert len(mixed) == 3
    assert sum(k.use_l_integral for k in mixed) == 1


def _complex_random(rng, shape):
    # Deliberately noncontiguous, with no integral or amplitude symmetries.
    padded_shape = shape[:-1] + (2 * shape[-1],)
    raw = rng.standard_normal(padded_shape) + 1j * rng.standard_normal(padded_shape)
    return raw[..., ::2]


def _separate_contractions(evaluator, spec):
    """Reference: evaluate every term, then apply the original group actions."""
    result = evaluator.evaluate(spec["explicit"], spec["spaces"])
    for axes_group, terms in spec["groups"]:
        representative = evaluator.evaluate(terms, spec["spaces"])
        for axes in axes_group:
            result += representative.transpose(axes)
    return result


def test_schedule_renames_contraction_labels_but_preserves_input_relationships():
    operands = (("f", "vv"), ("f", "vv"))
    spec = {
        "spaces": "vv",
        "explicit": (
            ((1, 1, 0, 1), (), operands, "ab,bc->ac"),
            ((2, 1, 0, 1), ("c0",), operands, "xy,yz->zx"),
            ((-1, 1, 0, 1), (), operands, "ab,cb->ac"),
        ),
        "groups": (),
    }
    schedule = compile_sigma_schedule(spec)
    assert len(schedule.kernels) == 2
    assert schedule.kernels[0].equation == "ab,bc->ac"
    assert tuple(term.axes for term in schedule.kernels[0].destinations) == (
        (0, 1),
        (1, 0),
    )

    rng = np.random.default_rng(3123)
    evaluator = ContractionEvaluator(
        {"nocc": 0, "nvir": 3, "f": _complex_random(rng, (3, 3)), "c0": 1 + 2j},
        dtype=np.dtype(np.complex128),
        occupied=slice(0, 0),
        virtual=slice(0, 3),
    )
    np.testing.assert_allclose(
        evaluator.evaluate_sigma_schedule(schedule),
        _separate_contractions(evaluator, spec),
        rtol=1e-12,
        atol=1e-12,
    )


def test_integral_pair_exchange_transforms_block_spaces_but_not_single_pairs():
    # The first two are equal under pair exchange, but the last expression
    # would require an independent exchange of the last two integral slots.
    spec = {
        "spaces": "ovov",
        "explicit": (
            ((1, 1, 0, 1), (), (("u", "ovov"),), "abcd->abcd"),
            ((2, 1, 0, 1), (), (("u", "vovo"),), "badc->abcd"),
            ((-1, 1, 0, 1), (), (("u", "ovvo"),), "abdc->abcd"),
        ),
        "groups": (),
    }
    rng = np.random.default_rng(3129)
    u = _complex_random(rng, (5,) * 4)
    u[...] = 0.5 * (u + u.transpose(1, 0, 3, 2))
    evaluator = ContractionEvaluator(
        {"nocc": 2, "nvir": 3, "u": u},
        dtype=np.dtype(np.complex128),
        occupied=slice(0, 2),
        virtual=slice(2, 5),
    )
    assert len(compile_sigma_schedule(spec).kernels) == 3
    schedule = compile_sigma_schedule(spec, assume_symmetry=True)
    assert len(schedule.kernels) == 2
    np.testing.assert_allclose(
        evaluator.evaluate_sigma_schedule(schedule),
        _separate_contractions(evaluator, spec),
        rtol=1e-12,
        atol=1e-12,
    )


def test_direct_exchange_pairs_require_proportional_destination_weights():
    amplitude = ("c1", "")
    direct = ((2, 3, 4, 5), (), (amplitude, ("u", "ovvo")), "ab,bcad->cd")
    exchange = ((-1, 3, -2, 5), (), (amplitude, ("u", "ovov")), "ab,bcda->cd")
    spec = {"spaces": "vo", "explicit": (direct, exchange), "groups": ()}
    rng = np.random.default_rng(3241)
    evaluator = ContractionEvaluator(
        {
            "nocc": 2,
            "nvir": 3,
            "u": _complex_random(rng, (5,) * 4),
            "c1": _complex_random(rng, (3, 2)),
        },
        dtype=np.dtype(np.complex128),
        occupied=slice(0, 2),
        virtual=slice(2, 5),
    )
    combined = compile_sigma_schedule(spec, combine_integrals=True)
    assert len(combined.kernels) == 1
    assert combined.kernels[0].use_l_integral
    np.testing.assert_allclose(
        evaluator.evaluate_sigma_schedule(combined),
        _separate_contractions(evaluator, spec),
        atol=1e-12,
        rtol=1e-12,
    )

    # The same two contractions also feed another group, but with coefficients
    # that are not in a 2:-1 ratio. They must not be combined.
    extra_terms = (((1, 1, 0, 1), *direct[1:]), ((1, 1, 0, 1), *exchange[1:]))
    spec["groups"] = ((((0, 1),), extra_terms),)
    separate = compile_sigma_schedule(spec, combine_integrals=True)
    assert len(separate.kernels) == 2
    assert not any(k.use_l_integral for k in separate.kernels)
    np.testing.assert_allclose(
        evaluator.evaluate_sigma_schedule(separate),
        _separate_contractions(evaluator, spec),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "specs", (CISDT_SIGMA_SPECS, CIDT_SIGMA_SPECS), ids=("cisdt", "cidt")
)
@pytest.mark.parametrize(("nocc", "nvir"), ((2, 3), (3, 2), (0, 3), (2, 0)))
@pytest.mark.parametrize("assume_symmetry", (False, True), ids=("exact", "symmetry"))
@pytest.mark.parametrize("combine_integrals", (False, True), ids=("separate", "L"))
@pytest.mark.parametrize("optimize_accumulation", (False, True), ids=("old", "fused"))
@pytest.mark.parametrize("gemm_mode", ("einsum", "full", "blocked"))
def test_schedule_matches_separate_contractions_on_complex_arrays(
    specs, nocc, nvir, assume_symmetry, combine_integrals, optimize_accumulation,
    gemm_mode, monkeypatch,
):
    rng = np.random.default_rng(7921)
    norb = nocc + nvir + 2
    data = {
        "nocc": nocc,
        "nvir": nvir,
        "e_ref": -0.7 + 0.3j,
        "c0": 0.4 - 0.8j,
        "f": _complex_random(rng, (norb, norb)),
        "u": _complex_random(rng, (norb,) * 4),
        "c1": _complex_random(rng, (nvir, nocc)),
        "c2": _complex_random(rng, (nvir,) * 2 + (nocc,) * 2),
        "c3": _complex_random(rng, (nvir,) * 3 + (nocc,) * 3),
    }
    if assume_symmetry:
        # Impose only the equalities used by the schedule, leaving f arbitrary
        # and preserving the strided layouts. In particular, u need not be
        # Hermitian, and the triples may have a nonzero null-space component.
        u = data["u"]
        u[...] = 0.5 * (u + u.transpose(1, 0, 3, 2))
        assert not np.allclose(u, u.transpose(2, 3, 0, 1).conj())
        assert not np.allclose(u, u.transpose(1, 0, 2, 3))
        c2 = data["c2"]
        c2[...] = 0.5 * (c2 + c2.transpose(1, 0, 3, 2))
        c3 = data["c3"]
        c3[...] = symmetrize_triples_columns(c3)
        if nocc and nvir:
            assert np.linalg.norm(triples_null_component(c3)) > 0.0
            assert not np.allclose(c3, c3.swapaxes(0, 1))

    original_arrays = {
        key: value.copy()
        for key, value in data.items()
        if isinstance(value, np.ndarray)
    }
    evaluator = ContractionEvaluator(
        data,
        dtype=np.dtype(np.complex128),
        occupied=slice(1, nocc + 1),
        virtual=slice(nocc + 2, norb),
    )
    einsum = np.einsum
    calls = []

    def count_einsum(equation, *operands, **kwargs):
        calls.append(equation)
        return einsum(equation, *operands, **kwargs)

    monkeypatch.setattr(np, "einsum", count_einsum)
    for spec in specs.values():
        expected = _separate_contractions(evaluator, spec)
        schedule = compile_sigma_schedule(
            spec, assume_symmetry=assume_symmetry, combine_integrals=combine_integrals
        )
        calls.clear()
        actual = evaluator.evaluate_sigma_schedule(
            schedule,
            optimize_accumulation=optimize_accumulation,
            use_gemm=gemm_mode != "einsum",
            gemm_block_size=1 if gemm_mode == "blocked" else None,
        )
        assert calls == [
            kernel.equation for kernel in schedule.kernels
            if not (
                gemm_mode != "einsum" and schedule.spaces == "vvvooo"
                and nocc and nvir and len(kernel.operands) == 2
                and kernel.operands[0] == ("c3", "")
            )
        ]
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    for key, original in original_arrays.items():
        np.testing.assert_array_equal(data[key], original)
    if combine_integrals:
        assert set(evaluator._l_cache) == {"oovv", "ovvo", "ooov", "ovvv"}
        for spaces, block in evaluator._l_cache.items():
            exchange_spaces = spaces[:2] + spaces[3] + spaces[2]
            np.testing.assert_array_equal(
                block,
                2 * evaluator._operand("u", spaces)
                - evaluator._operand("u", exchange_spaces).swapaxes(2, 3),
            )


@pytest.mark.parametrize(
    ("reference", "optimized", "include_singles"),
    [
        (compute_cisdt_sigma, cisdt_optimized.compute_sigma, True),
        (compute_cidt_sigma, cidt_optimized.compute_sigma, False),
    ],
    ids=("cisdt", "cidt"),
)
@pytest.mark.parametrize("input_kind", ("arbitrary", "pair-exchange", "physical"))
def test_optimized_sigma_vectors_match_original_public_implementation(
    reference, optimized, include_singles, input_kind, monkeypatch
):
    rng = np.random.default_rng(1735)
    nocc, nvir = 2, 3
    norb = nocc + nvir
    f = _complex_random(rng, (norb, norb))
    u = _complex_random(rng, (norb,) * 4)
    c1 = _complex_random(rng, (nvir, nocc))
    c2 = _complex_random(rng, (nvir,) * 2 + (nocc,) * 2)
    c3 = _complex_random(rng, (nvir,) * 3 + (nocc,) * 3)
    if input_kind != "arbitrary":
        u = 0.5 * (u + u.transpose(1, 0, 3, 2))
        c2 = 0.5 * (c2 + c2.transpose(1, 0, 3, 2))
        c3 = symmetrize_triples_columns(c3)
    if input_kind == "physical":
        f = 0.5 * (f + f.T.conj())
        u = 0.5 * (u + u.transpose(2, 3, 0, 1).conj())
        c3 = project_physical_triples(c3)
    amplitudes = (c1, c2, c3) if include_singles else (c2, c3)
    args = (
        -0.7 + 0.3j,
        f,
        u,
        0.4 - 0.8j,
        *amplitudes,
        slice(0, nocc),
        slice(nocc, norb),
    )

    expected = reference(*args)
    for actual, reference_component in zip(optimized(*args), expected, strict=True):
        np.testing.assert_allclose(actual, reference_component, atol=1e-12, rtol=1e-12)
    for actual, reference_component in zip(
        optimized(*args, combine_integrals=True), expected, strict=True
    ):
        np.testing.assert_allclose(actual, reference_component, atol=1e-12, rtol=1e-12)
    # Exercise the compiled point-4 path even at these tiny test sizes.
    monkeypatch.setattr(_accumulation, "_FUSED_MIN_ELEMENTS", 0)
    for actual, reference_component in zip(
        optimized(*args, combine_integrals=True, optimize_accumulation=True),
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(actual, reference_component, atol=1e-12, rtol=1e-12)
    for block_size in (None, 1):
        for actual, reference_component in zip(
            optimized(*args, combine_integrals=True, optimize_accumulation=True,
                      use_gemm=True, gemm_block_size=block_size),
            expected, strict=True,
        ):
            np.testing.assert_allclose(
                actual, reference_component, atol=1e-12, rtol=1e-12
            )
    if input_kind != "arbitrary":
        for combine_integrals in (False, True):
            for actual, reference_component in zip(
                optimized(
                    *args, assume_symmetry=True, combine_integrals=combine_integrals
                ),
                expected,
                strict=True,
            ):
                np.testing.assert_allclose(
                    actual, reference_component, atol=1e-12, rtol=1e-12
                )
        for actual, reference_component in zip(
            optimized(
                *args,
                assume_symmetry=True,
                combine_integrals=True,
                optimize_accumulation=True,
                use_gemm=True,
                gemm_block_size=1,
            ),
            expected,
            strict=True,
        ):
            np.testing.assert_allclose(
                actual, reference_component, atol=1e-12, rtol=1e-12
            )


def test_combined_integral_blocks_are_rebuilt_for_changed_orbitals():
    rng = np.random.default_rng(1751)
    nocc, nvir = 2, 3
    norb = nocc + nvir
    u = _complex_random(rng, (norb,) * 4)
    args = (
        -0.5,
        _complex_random(rng, (norb, norb)),
        u,
        0.8 - 0.1j,
        _complex_random(rng, (nvir, nocc)),
        _complex_random(rng, (nvir,) * 2 + (nocc,) * 2),
        _complex_random(rng, (nvir,) * 3 + (nocc,) * 3),
        slice(0, nocc),
        slice(nocc, norb),
    )
    cisdt_optimized.compute_sigma(*args, combine_integrals=True)
    u[...] = _complex_random(rng, u.shape)
    for actual, expected in zip(
        cisdt_optimized.compute_sigma(*args, combine_integrals=True),
        compute_cisdt_sigma(*args),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
