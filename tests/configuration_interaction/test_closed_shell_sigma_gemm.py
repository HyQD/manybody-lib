"""Explicit and tiled GEMM tests for generated triples contractions."""

from __future__ import annotations

import numpy as np
import pytest

from manybody_lib.configuration_interaction.closed_shell.cidt._specs import (
    SIGMA_SPECS as CIDT_SIGMA_SPECS,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt import (
    symmetrize_triples_columns,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt._contractions import (
    SIGMA_SPECS as CISDT_SIGMA_SPECS,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt._engine import (
    ContractionEvaluator,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt._gemm import (
    GemmPlan,
    compile_gemm_plan,
    validate_gemm_options,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt._schedule import (
    compile_sigma_schedule,
)


def _strided_random(rng, shape, complex_input=True):
    padded = shape[:-1] + (2 * shape[-1],)
    raw = rng.standard_normal(padded)
    if complex_input:
        raw = raw + 1j * rng.standard_normal(padded)
    return raw[..., ::2]


@pytest.mark.parametrize("specs", (CIDT_SIGMA_SPECS, CISDT_SIGMA_SPECS))
@pytest.mark.parametrize("assume_symmetry", (False, True))
@pytest.mark.parametrize("combine_integrals", (False, True))
@pytest.mark.parametrize("complex_input", (False, True))
def test_every_binary_triples_plan_matches_einsum(
    specs, assume_symmetry, combine_integrals, complex_input
):
    rng = np.random.default_rng(6171)
    nocc, nvir = 2, 3
    norb = nocc + nvir + 1
    shape = (nvir,) * 3 + (nocc,) * 3
    c3 = _strided_random(rng, shape, complex_input)
    if assume_symmetry:
        c3[...] = symmetrize_triples_columns(c3).real if not complex_input else (
            symmetrize_triples_columns(c3)
        )
    evaluator = ContractionEvaluator(
        {
            "nocc": nocc, "nvir": nvir, "c3": c3,
            "f": _strided_random(rng, (norb, norb), complex_input),
            "u": _strided_random(rng, (norb,) * 4, complex_input),
        },
        dtype=np.dtype(np.complex128),
        occupied=slice(0, nocc),
        virtual=slice(nocc + 1, norb),
    )
    schedule = compile_sigma_schedule(
        specs[3], assume_symmetry=assume_symmetry, combine_integrals=combine_integrals
    )
    for kernel in schedule.kernels:
        if len(kernel.operands) != 2 or kernel.operands[0] != ("c3", ""):
            continue
        _, (source, spaces) = kernel.operands
        integral = (
            evaluator._l_integral(spaces) if kernel.use_l_integral
            else evaluator._operand(source, spaces)
        )
        plan = compile_gemm_plan(kernel.equation, assume_symmetry)
        assert plan is not None
        expected = np.einsum(kernel.equation, c3, integral, optimize=True)
        for block_size in (None, 1, 2, 5):
            pieces = [
                contribution
                for _, _, contribution in plan.tiles(c3, integral, block_size)
            ]
            actual = np.concatenate(pieces, axis=plan.output_tile_axis)
            assert actual.shape == expected.shape
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_column_symmetry_exposes_virtual_pair_matrix_view():
    original = "abcdef,ghbc->ghaefd"
    unsymmetrized = compile_gemm_plan(original, False)
    symmetric = compile_gemm_plan(original, True)
    assert unsymmetrized.amplitude_axes != tuple(range(6))
    assert symmetric.amplitude_axes == tuple(range(6))
    assert symmetric.contract_first
    assert symmetric.output_tile_axis == 2
    c3 = np.zeros((4, 4, 4, 2, 2, 2), dtype=np.complex128)
    for _, _, matrix, _ in symmetric.amplitude_tiles(c3, 2):
        assert np.shares_memory(matrix, c3)


def test_gemm_plan_rejects_unsupported_contractions_and_invalid_block_sizes():
    assert compile_gemm_plan("abcdef->abcdef") is None
    assert compile_gemm_plan("abcdef,fg->abcdef") is None
    assert compile_gemm_plan("abcddf,fg->abcdgg") is None
    assert validate_gemm_options(True, np.int64(3)) == 3
    assert validate_gemm_options(False, None) is None
    with pytest.raises(ValueError):
        validate_gemm_options(False, 2)
    with pytest.raises(ValueError):
        validate_gemm_options(True, 0)
    with pytest.raises(TypeError):
        validate_gemm_options(True, True)
    with pytest.raises(TypeError):
        validate_gemm_options(True, 1.5)


def test_adjacent_mixed_gemms_share_amplitude_packing(monkeypatch):
    rng = np.random.default_rng(6173)
    nocc, nvir = 2, 3
    norb = nocc + nvir
    data = {
        "nocc": nocc, "nvir": nvir, "e_ref": -0.2 + 0.1j,
        "c1": _strided_random(rng, (nvir, nocc)),
        "c2": _strided_random(rng, (nvir,) * 2 + (nocc,) * 2),
        "c3": _strided_random(rng, (nvir,) * 3 + (nocc,) * 3),
        "f": _strided_random(rng, (norb, norb)),
        "u": _strided_random(rng, (norb,) * 4),
    }
    schedule = compile_sigma_schedule(CISDT_SIGMA_SPECS[3], combine_integrals=True)
    count = sum(
        len(kernel.operands) == 2 and kernel.operands[0] == ("c3", "")
        for kernel in schedule.kernels
    )
    calls = []
    original = GemmPlan.amplitude_tiles

    def counted(self, *args, **kwargs):
        calls.append(self.amplitude_layout)
        yield from original(self, *args, **kwargs)

    monkeypatch.setattr(GemmPlan, "amplitude_tiles", counted)
    evaluator = ContractionEvaluator(
        data, dtype=np.dtype(np.complex128),
        occupied=slice(0, nocc), virtual=slice(nocc, norb),
    )
    expected = evaluator.evaluate_sigma_schedule(schedule)
    actual = evaluator.evaluate_sigma_schedule(
        schedule, use_gemm=True, gemm_block_size=1
    )
    assert len(calls) < count
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
