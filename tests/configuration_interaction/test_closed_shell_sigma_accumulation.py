"""Memory-aware rank-six accumulation and Matthews projection regressions."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from manybody_lib.configuration_interaction.closed_shell.cisdt import (
    _accumulation,
    project_physical_triples,
    triples_null_component,
)


def _complex_random(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.mark.parametrize("source_dtype", (np.float64, np.complex128))
@pytest.mark.parametrize("fused", (False, True), ids=("numpy", "numba"))
def test_weighted_permutations_match_separate_additions(
    source_dtype, fused, monkeypatch
):
    rng = np.random.default_rng(1481)
    shape = (3, 3, 3, 2, 2, 2)
    # Include noncontiguous and readonly source views.
    raw_shape = shape[:-1] + (2 * shape[-1],)
    raw = rng.standard_normal(raw_shape)
    if source_dtype == np.complex128:
        raw = raw + 1j * rng.standard_normal(raw_shape)
    source = raw[..., ::2]
    source.flags.writeable = False
    original_source = source.copy()
    target = _complex_random(rng, shape)
    expected = target.copy()
    contributions = [
        (permutation + tuple(3 + p for p in permutation), 0.2 * i - 0.4j)
        for i, permutation in enumerate(permutations(range(3)), start=1)
    ]
    for axes, coefficient in contributions:
        expected += coefficient * source.transpose(axes)

    monkeypatch.setattr(
        _accumulation, "_FUSED_MIN_ELEMENTS", 0 if fused else target.size + 1
    )
    _accumulation.accumulate_permutations(target, source, contributions)
    np.testing.assert_allclose(target, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(source, original_source)


@pytest.mark.parametrize("allow_strided_target", (False, True))
def test_strided_destination_accumulation(monkeypatch, allow_strided_target):
    rng = np.random.default_rng(1483)
    shape = (3, 3, 3, 2, 2, 2)
    source = _complex_random(rng, shape)
    target = _complex_random(rng, shape[:-1] + (4,))[..., ::2]
    expected = target.copy()
    contributions = (
        ((0, 1, 2, 3, 4, 5), 0.7 + 0.4j),
        ((1, 0, 2, 4, 3, 5), -1),
    )
    for axes, coefficient in contributions:
        expected += coefficient * source.transpose(axes)
    # A non-C-contiguous target must use the tiled NumPy fallback even
    # when the size threshold would otherwise permit the fused kernel.
    monkeypatch.setattr(_accumulation, "_FUSED_MIN_ELEMENTS", 0)
    _accumulation.accumulate_permutations(
        target, source, contributions, allow_strided_target=allow_strided_target
    )
    np.testing.assert_allclose(target, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("nocc", (0, 1, 2, 3, 4))
@pytest.mark.parametrize("fused", (False, True), ids=("numpy", "numba"))
def test_inplace_projection_matches_original(nocc, fused, monkeypatch):
    rng = np.random.default_rng(1493)
    shape = (2, 2, 2, nocc, nocc, nocc)
    target = _complex_random(rng, shape)
    expected = project_physical_triples(target)
    monkeypatch.setattr(
        _accumulation, "_FUSED_MIN_ELEMENTS", 0 if fused else target.size + 1
    )
    _accumulation.project_triples_inplace(target)
    np.testing.assert_allclose(target, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(triples_null_component(target), 0, atol=1e-12)
