from sympy import (
    Dummy,
    KroneckerDelta,
    Rational,
    conjugate,
    expand,
    simplify,
    symbols,
)

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
#   gamma^q_p = <Psi| a_p^dagger a_q |Psi>
#
# CID wavefunction:
#
#   |Psi> = (c0 + C2) |Phi0>
#
#   C2 = 1/4 sum_cdkl c_kl^cd
#             a_c^dagger a_d^dagger a_l a_k
#
# The bra amplitudes are complex conjugated explicitly.  No normalization
# relation is imposed and the RDM is not divided by the wavefunction norm.
# This keeps norm drift visible during, for example, real-time TDCID.
#
# Indices:
#   i,j,k,... : occupied spin orbitals
#   a,b,c,... : virtual spin orbitals
# ============================================================


# ------------------------------------------------------------
# External block indices
# ------------------------------------------------------------

i, j = symbols("i j", below_fermi=True)
a, b = symbols("a b", above_fermi=True)


# ------------------------------------------------------------
# Independent bra/ket summation indices
# ------------------------------------------------------------

c = Dummy("c", above_fermi=True)
d = Dummy("d", above_fermi=True)
e = Dummy("e", above_fermi=True)
f = Dummy("f", above_fermi=True)

k = Dummy("k", below_fermi=True)
l = Dummy("l", below_fermi=True)
m = Dummy("m", below_fermi=True)
n = Dummy("n", below_fermi=True)


# ------------------------------------------------------------
# CID coefficients and wavefunction factors
# ------------------------------------------------------------

c0 = symbols("c0")

c2_bra_coefficient = AntiSymmetricTensor(
    "c",
    (c, d),
    (k, l),
)
c2_ket_coefficient = AntiSymmetricTensor(
    "c",
    (e, f),
    (m, n),
)

C2_ket = Rational(1, 4) * c2_ket_coefficient * NO(
    Fd(e) * Fd(f) * F(n) * F(m)
)

C2_bra = Rational(1, 4) * conjugate(c2_bra_coefficient) * (
    Fd(k) * Fd(l) * F(d) * F(c)
)

cid_ket = c0 + C2_ket
cid_bra = conjugate(c0) + C2_bra


# ------------------------------------------------------------
# Wick-contraction helpers
# ------------------------------------------------------------

PRETTY_INDICES = {
    "above": "cdefgh",
    "below": "klmno",
    "general": "pqrstuv",
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
    return contract(cid_bra * operator * cid_ket)


# ============================================================
# CID norm
# ============================================================

doubles_norm = contract(C2_bra * C2_ket)
cid_norm = c0 * conjugate(c0) + doubles_norm


# ============================================================
# One-body RDM blocks
# ============================================================
#
# The block name lists the lower (creation) index first and the upper
# (annihilation) index second, consistently with gamma^q_p.
# ============================================================


# (i,j) block: gamma^j_i
gamma_ij = one_body_expectation(i, j)


# (i,a) block: gamma^a_i
gamma_ia = one_body_expectation(i, a)


# (a,i) block: gamma^i_a
gamma_ai = one_body_expectation(a, i)


# (a,b) block: gamma^b_a
gamma_ab = one_body_expectation(a, b)


# A one-body occupied-virtual operator changes excitation rank by one.  It
# therefore cannot couple determinants in the reference-plus-doubles space.
assert gamma_ia == 0
assert gamma_ai == 0


# Hermiticity checks for the two nonzero diagonal blocks.
gamma_ji = gamma_ij.xreplace({i: j, j: i})
assert simplify(gamma_ij - conjugate(gamma_ji)) == 0

gamma_ba = gamma_ab.xreplace({a: b, b: a})
assert simplify(gamma_ab - conjugate(gamma_ba)) == 0


# ============================================================
# LaTeX presentation
# ============================================================


class CIDOneRDMPrinter(LatexPrinter):
    """Render CID coefficients using conventional orbital-index notation."""

    def _print(self, expr, **kwargs):
        if isinstance(expr, AntiSymmetricTensor):
            return self._print_AntiSymmetricTensor(expr)
        return super()._print(expr, **kwargs)

    def _print_Symbol(self, expr):
        if expr == c0:
            return r"c_{0}"
        return super()._print_Symbol(expr)

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
    """Return copyable LaTeX for a derived one-body RDM expression."""

    return CIDOneRDMPrinter().doprint(expr)


NONZERO_BLOCKS = (
    (r"\gamma^{j}_{i}", gamma_ij),
    (r"\gamma^{b}_{a}", gamma_ab),
)


def print_cid_1rdm():
    """Print the nonzero spin-orbital CID one-body RDM blocks."""

    print(r"\begin{align}")
    for position, (block_name, expression) in enumerate(NONZERO_BLOCKS):
        row_ending = r" \\" if position < len(NONZERO_BLOCKS) - 1 else ""
        print(
            block_name
            + " &= "
            + latex_expression(expression)
            + row_ending
        )
    print(r"\end{align}")


if __name__ == "__main__":
    print_cid_1rdm()
