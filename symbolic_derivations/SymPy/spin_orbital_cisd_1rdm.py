from sympy import Dummy, Function, conjugate, expand, symbols

from sympy.printing.latex import LatexPrinter

from sympy.physics.secondquant import (
    F,
    Fd,
    NO,
    AntiSymmetricTensor,
    evaluate_deltas,
    substitute_dummies,
    wicks,
)


# ============================================================
# Conventions
# ============================================================
#
# One-body reduced density matrix:
#
#   gamma^q_p = <Psi|a_p^dagger a_q|Psi>
#
# CISD wavefunction:
#
#   |Psi> = (c0 + C1 + C2) |Phi0>
#
#   C1 = sum_ai c_i^a a_a^dagger a_i
#
#   C2 = 1/4 sum_abij c_ij^ab
#             a_a^dagger a_b^dagger a_j a_i
#
# Bra coefficients are complex conjugated explicitly.  No normalization
# relation is imposed and the 1-RDM is not divided by the wavefunction norm.
# This preserves norm drift in, for example, real-time TDCISD.
# ============================================================


# ------------------------------------------------------------
# External block indices
# ------------------------------------------------------------

i, j = symbols("i j", below_fermi=True)
a, b = symbols("a b", above_fermi=True)


# ------------------------------------------------------------
# Independent bra/ket summation indices
# ------------------------------------------------------------

e = Dummy("e", above_fermi=True)
f = Dummy("f", above_fermi=True)
g = Dummy("g", above_fermi=True)
h = Dummy("h", above_fermi=True)

m = Dummy("m", below_fermi=True)
n = Dummy("n", below_fermi=True)
o = Dummy("o", below_fermi=True)
p = Dummy("p", below_fermi=True)


# ------------------------------------------------------------
# CISD coefficients and wavefunction factors
# ------------------------------------------------------------

c0 = symbols("c0")
c1 = Function("c")

c2_bra = AntiSymmetricTensor("c", (e, f), (m, n))
c2_ket = AntiSymmetricTensor("c", (g, h), (o, p))

C1_bra = conjugate(c1(e, m)) * Fd(m) * F(e)
C1_ket = c1(g, o) * NO(Fd(g) * F(o))

C2_bra = conjugate(c2_bra) * Fd(m) * Fd(n) * F(f) * F(e) / 4
C2_ket = c2_ket * NO(Fd(g) * Fd(h) * F(p) * F(o)) / 4

cisd_bra = conjugate(c0) + C1_bra + C2_bra
cisd_ket = c0 + C1_ket + C2_ket


# ------------------------------------------------------------
# Wick-contraction helpers
# ------------------------------------------------------------

PRETTY_INDICES = {
    "above": "cdefgh",
    "below": "klmnop",
    "general": "qrstuvwxyz",
}


def canonicalize_dummies(expr):
    """Canonicalize equivalent Einstein summations."""

    return expand(
        substitute_dummies(
            expand(expr),
            new_indices=True,
            pretty_indices=PRETTY_INDICES,
        )
    )


def contract(expr):
    """Return the fully contracted part with respect to the reference."""

    expr = wicks(
        expand(expr),
        keep_only_fully_contracted=True,
    )
    expr = evaluate_deltas(expr)
    return canonicalize_dummies(expr)


def one_body_expectation(creation_index, annihilation_index):
    """Derive <Psi|a_p^dagger a_q|Psi> for the requested indices."""

    operator = Fd(creation_index) * F(annihilation_index)
    return contract(cisd_bra * operator * cisd_ket)


# ============================================================
# CISD norm and one-body RDM blocks
# ============================================================

cisd_norm = contract(cisd_bra * cisd_ket)

BLOCK_SPECS = (
    ("ij", r"\gamma^{j}_{i}", (i, j)),
    ("ia", r"\gamma^{a}_{i}", (i, a)),
    ("ai", r"\gamma^{i}_{a}", (a, i)),
    ("ab", r"\gamma^{b}_{a}", (a, b)),
)

BLOCKS = {
    key: one_body_expectation(*indices)
    for key, _, indices in BLOCK_SPECS
}

NONZERO_BLOCKS = tuple(
    (label, BLOCKS[key])
    for key, label, _ in BLOCK_SPECS
    if BLOCKS[key] != 0
)

assert len(NONZERO_BLOCKS) == 4


# ============================================================
# LaTeX presentation
# ============================================================


class CISDOneRDMPrinter(LatexPrinter):
    """Render CISD coefficients using conventional index placement."""

    def _print(self, expr, **kwargs):
        if isinstance(expr, AntiSymmetricTensor):
            return self._print_AntiSymmetricTensor(expr)
        return super()._print(expr, **kwargs)

    def _print_Symbol(self, expr):
        if expr == c0:
            return r"c_{0}"
        return super()._print_Symbol(expr)

    def _print_Function(self, expr, exp=None):
        if expr.func == c1 and len(expr.args) == 2:
            virtual, occupied = expr.args
            amplitude = (
                rf"c_{{{self._print(occupied)}}}"
                rf"^{{{self._print(virtual)}}}"
            )
            return self._do_exponent(amplitude, exp)
        return super()._print_Function(expr, exp)

    def _print_AntiSymmetricTensor(self, expr):
        virtual = "".join(self._print(index) for index in expr.upper)
        occupied = "".join(self._print(index) for index in expr.lower)
        return rf"c_{{{occupied}}}^{{{virtual}}}"

    def _print_conjugate(self, expr, exp=None):
        conjugated = rf"\left({self._print(expr.args[0])}\right)^{{*}}"
        return self._do_exponent(conjugated, exp)

    def _print_KroneckerDelta(self, expr):
        left, right = expr.args
        return rf"\delta_{{{self._print(left)}{self._print(right)}}}"


def latex_expression(expr):
    """Return copyable LaTeX for a CISD one-body RDM expression."""

    return CISDOneRDMPrinter().doprint(expr)


def print_cisd_1rdm():
    """Print the nonzero spin-orbital CISD one-body RDM blocks."""

    print(r"\begin{align}")
    for position, (block_name, expression) in enumerate(NONZERO_BLOCKS):
        row_ending = r" \\" if position < len(NONZERO_BLOCKS) - 1 else ""
        print(block_name + " &= " + latex_expression(expression) + row_ending)
    print(r"\end{align}")


if __name__ == "__main__":
    print_cisd_1rdm()
