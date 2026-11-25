from sympy import (
    Dummy,
    Function,
    KroneckerDelta,
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
# CIS wavefunction:
#
#   |Psi> = (c0 + C1) |Phi0>
#
#   C1 = sum_ai c_i^a a_a^dagger a_i
#
# The bra amplitudes are complex conjugated explicitly.  No normalization
# relation is imposed and the RDM is not divided by the wavefunction norm.
# This keeps norm drift visible during, for example, real-time TDCIS.
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
k = Dummy("k", below_fermi=True)
l = Dummy("l", below_fermi=True)


# ------------------------------------------------------------
# CIS coefficients and wavefunction factors
# ------------------------------------------------------------

c0 = symbols("c0")
c1 = Function("c")

C1_ket = c1(d, l) * NO(
    Fd(d) * F(l)
)

C1_bra = conjugate(c1(c, k)) * Fd(k) * F(c)

cis_ket = c0 + C1_ket
cis_bra = conjugate(c0) + C1_bra


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
    return contract(cis_bra * operator * cis_ket)


# ============================================================
# CIS norm
# ============================================================

singles_norm = contract(C1_bra * C1_ket)
cis_norm = c0 * conjugate(c0) + singles_norm


# ============================================================
# One-body RDM blocks
# ============================================================
#
# The block name lists the lower (creation) index first and the upper
# (annihilation) index second, consistently with gamma^q_p.
# ============================================================


# (i,j) block: gamma^j_i.  Retain the explicit CIS norm contribution.
gamma_ij = one_body_expectation(i, j)


# (i,a) block: gamma^a_i
gamma_ia = one_body_expectation(i, a)


# (a,i) block: gamma^i_a
gamma_ai = one_body_expectation(a, i)


# (a,b) block: gamma^b_a
gamma_ab = one_body_expectation(a, b)


# Hermiticity check for the two off-diagonal blocks.
assert simplify(gamma_ai - conjugate(gamma_ia)) == 0


# ============================================================
# LaTeX presentation
# ============================================================


class CISOneRDMPrinter(LatexPrinter):
    """Render CIS coefficients using conventional orbital-index notation."""

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

    def _print_conjugate(self, expr, exp=None):
        conjugated = rf"\left({self._print(expr.args[0])}\right)^{{*}}"
        return self._do_exponent(conjugated, exp)

    def _print_KroneckerDelta(self, expr):
        left, right = expr.args
        return rf"\delta_{{{self._print(left)}{self._print(right)}}}"


def latex_expression(expr):
    """Return copyable LaTeX for a derived one-body RDM expression."""

    return CISOneRDMPrinter().doprint(expr)


NONZERO_BLOCKS = (
    (r"\gamma^{j}_{i}", gamma_ij),
    (r"\gamma^{a}_{i}", gamma_ia),
    (r"\gamma^{i}_{a}", gamma_ai),
    (r"\gamma^{b}_{a}", gamma_ab),
)


def print_cis_1rdm():
    """Print the nonzero spin-orbital CIS one-body RDM blocks."""

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
    print_cis_1rdm()
