"""Dedicated spin-orbital and closed-shell CIDT tests."""

from __future__ import annotations

import numpy as np

from manybody_lib.configuration_interaction.closed_shell.cidt import (
    compute_norm as closed_cidt_norm,
)
from manybody_lib.configuration_interaction.closed_shell.cidt import (
    compute_one_rdm as closed_cidt_one_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cidt import (
    compute_sigma as closed_cidt_sigma,
)
from manybody_lib.configuration_interaction.closed_shell.cidt import (
    compute_two_rdm as closed_cidt_two_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt import (
    compute_norm as closed_cisdt_norm,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt import (
    compute_one_rdm as closed_cisdt_one_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt import (
    compute_sigma as closed_cisdt_sigma,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt import (
    compute_two_rdm as closed_cisdt_two_rdm,
)
from manybody_lib.configuration_interaction.closed_shell.cisdt import (
    project_physical_triples,
    symmetrize_triples_columns,
)
from manybody_lib.configuration_interaction.spin_orbital.cidt import (
    compute_norm as spin_cidt_norm,
)
from manybody_lib.configuration_interaction.spin_orbital.cidt import (
    compute_one_rdm as spin_cidt_one_rdm,
)
from manybody_lib.configuration_interaction.spin_orbital.cidt import (
    compute_sigma as spin_cidt_sigma,
)
from manybody_lib.configuration_interaction.spin_orbital.cidt import (
    compute_two_rdm as spin_cidt_two_rdm,
)
from manybody_lib.configuration_interaction.spin_orbital.cisdt import (
    antisymmetrize_doubles,
    antisymmetrize_triples,
)
from manybody_lib.configuration_interaction.spin_orbital.cisdt import (
    compute_norm as spin_cisdt_norm,
)
from manybody_lib.configuration_interaction.spin_orbital.cisdt import (
    compute_one_rdm as spin_cisdt_one_rdm,
)
from manybody_lib.configuration_interaction.spin_orbital.cisdt import (
    compute_sigma as spin_cisdt_sigma,
)
from manybody_lib.configuration_interaction.spin_orbital.cisdt import (
    compute_two_rdm as spin_cisdt_two_rdm,
)


def _complex_random(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_spin_orbital_cidt_is_cisdt_without_singles() -> None:
    rng = np.random.default_rng(1901)
    nocc, nvir = 3, 4
    norb = nocc + nvir
    occupied = slice(0, nocc)
    virtual = slice(nocc, norb)
    e_ref = np.complex128(-0.3 + 0.1j)
    f = _complex_random(rng, (norb, norb))
    u = antisymmetrize_doubles(_complex_random(rng, (norb,) * 4))
    c0 = np.complex128(0.2 - 0.6j)
    c1 = np.zeros((nvir, nocc), dtype=np.complex128)
    c2 = antisymmetrize_doubles(
        _complex_random(rng, (nvir, nvir, nocc, nocc))
    )
    c3 = antisymmetrize_triples(
        _complex_random(rng, (nvir, nvir, nvir, nocc, nocc, nocc))
    )

    cidt_sigma = spin_cidt_sigma(
        e_ref, f, u, c0, c2, c3, occupied, virtual
    )
    cisdt_sigma = spin_cisdt_sigma(
        e_ref, f, u, c0, c1, c2, c3, occupied, virtual
    )
    for actual, expected in zip(
        cidt_sigma,
        (cisdt_sigma[0], cisdt_sigma[2], cisdt_sigma[3]),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, atol=5e-10)
    np.testing.assert_allclose(
        spin_cidt_norm(c0, c2, c3),
        spin_cisdt_norm(c0, c1, c2, c3),
        atol=3e-10,
    )
    np.testing.assert_allclose(
        spin_cidt_one_rdm(c0, c2, c3),
        spin_cisdt_one_rdm(c0, c1, c2, c3),
        atol=3e-10,
    )
    np.testing.assert_allclose(
        spin_cidt_two_rdm(c0, c2, c3),
        spin_cisdt_two_rdm(c0, c1, c2, c3),
        atol=5e-10,
    )


def test_closed_shell_cidt_is_cisdt_without_singles() -> None:
    rng = np.random.default_rng(1913)
    nocc, nvir = 3, 4
    norb = nocc + nvir
    occupied = slice(0, nocc)
    virtual = slice(nocc, norb)
    e_ref = np.complex128(0.4 - 0.2j)
    f = _complex_random(rng, (norb, norb))
    u = _complex_random(rng, (norb,) * 4)
    c0 = np.complex128(-0.5 + 0.3j)
    c1 = np.zeros((nvir, nocc), dtype=np.complex128)
    raw2 = _complex_random(rng, (nvir, nvir, nocc, nocc))
    c2 = 0.5 * (raw2 + raw2.transpose(1, 0, 3, 2))
    raw3 = _complex_random(rng, (nvir, nvir, nvir, nocc, nocc, nocc))
    c3 = project_physical_triples(symmetrize_triples_columns(raw3))

    cidt_sigma = closed_cidt_sigma(
        e_ref, f, u, c0, c2, c3, occupied, virtual
    )
    cisdt_sigma = closed_cisdt_sigma(
        e_ref, f, u, c0, c1, c2, c3, occupied, virtual
    )
    for actual, expected in zip(
        cidt_sigma,
        (cisdt_sigma[0], cisdt_sigma[2], cisdt_sigma[3]),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, atol=8e-10)
    np.testing.assert_allclose(
        closed_cidt_norm(c0, c2, c3),
        closed_cisdt_norm(c0, c1, c2, c3),
        atol=5e-10,
    )
    np.testing.assert_allclose(
        closed_cidt_one_rdm(c0, c2, c3),
        closed_cisdt_one_rdm(c0, c1, c2, c3),
        atol=5e-10,
    )
    np.testing.assert_allclose(
        closed_cidt_two_rdm(c0, c2, c3),
        closed_cisdt_two_rdm(c0, c1, c2, c3),
        atol=8e-10,
    )


def test_closed_shell_cidt_supports_an_independent_left_state() -> None:
    rng = np.random.default_rng(1931)
    nocc, nvir = 3, 4
    c0 = np.complex128(0.7 + 0.2j)
    c1 = np.zeros((nvir, nocc), dtype=np.complex128)
    raw2 = _complex_random(rng, (nvir, nvir, nocc, nocc))
    c2 = 0.5 * (raw2 + raw2.transpose(1, 0, 3, 2))
    c3 = project_physical_triples(
        symmetrize_triples_columns(
            _complex_random(rng, (nvir, nvir, nvir, nocc, nocc, nocc))
        )
    )
    left0 = np.complex128(-0.1 + 0.4j)
    left1 = np.zeros((nocc, nvir), dtype=np.complex128)
    raw_left2 = _complex_random(rng, (nocc, nocc, nvir, nvir))
    left2 = 0.5 * (raw_left2 + raw_left2.transpose(1, 0, 3, 2))
    raw_left3 = _complex_random(rng, (nocc, nocc, nocc, nvir, nvir, nvir))
    left3 = symmetrize_triples_columns(
        raw_left3.transpose(3, 4, 5, 0, 1, 2)
    ).transpose(3, 4, 5, 0, 1, 2)

    np.testing.assert_allclose(
        closed_cidt_norm(c0, c2, c3, left0, left2, left3),
        closed_cisdt_norm(c0, c1, c2, c3, left0, left1, left2, left3),
        atol=5e-10,
    )
    np.testing.assert_allclose(
        closed_cidt_one_rdm(c0, c2, c3, left0, left2, left3),
        closed_cisdt_one_rdm(c0, c1, c2, c3, left0, left1, left2, left3),
        atol=5e-10,
    )
    np.testing.assert_allclose(
        closed_cidt_two_rdm(c0, c2, c3, left0, left2, left3),
        closed_cisdt_two_rdm(c0, c1, c2, c3, left0, left1, left2, left3),
        atol=8e-10,
    )
