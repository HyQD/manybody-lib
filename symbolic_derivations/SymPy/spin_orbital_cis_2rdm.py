from sympy import (
    Dummy,
    Function,
    KroneckerDelta,
    Mul,
    conjugate,
    expand,
    symbols,
)

from sympy.printing.latex import LatexPrinter

from sympy.physics.secondquant import (
    F,
    Fd,
    NO,
    PermutationOperator,
    evaluate_deltas,
    simplify_index_permutations,
    substitute_dummies,
    wicks,
)


# ============================================================
# Conventions
# ============================================================
#
# Two-body reduced density matrix:
#
#   Gamma^{rs}_{pq} = <Psi|a_p^dagger a_q^dagger a_s a_r|Psi>
#
# CIS wavefunction:
#
#   |Psi> = (c0 + C1) |Phi0>
#
#   C1 = sum_ai c_i^a a_a^dagger a_i
#
# The bra amplitudes are complex conjugated explicitly.  No normalization
# relation is imposed and the 2-RDM is not divided by the wavefunction norm.
# Consequently, norm drift remains visible in time-dependent calculations.
#
# Indices:
#   i,j,k,l,... : occupied spin orbitals
#   a,b,c,d,... : virtual spin orbitals
# ============================================================


# ------------------------------------------------------------
# External block indices
# ------------------------------------------------------------

i, j, k, l = symbols("i j k l", below_fermi=True)
a, b, c, d = symbols("a b c d", above_fermi=True)


# ------------------------------------------------------------
# Independent bra/ket summation indices
# ------------------------------------------------------------

e = Dummy("e", above_fermi=True)
f = Dummy("f", above_fermi=True)
m = Dummy("m", below_fermi=True)
n = Dummy("n", below_fermi=True)


# ------------------------------------------------------------
# CIS coefficients and wavefunction factors
# ------------------------------------------------------------

c0 = symbols("c0")
c1 = Function("c")

C1_ket = c1(f, n) * NO(Fd(f) * F(n))
C1_bra = conjugate(c1(e, m)) * Fd(m) * F(e)

cis_ket = c0 + C1_ket
cis_bra = conjugate(c0) + C1_bra


# ------------------------------------------------------------
# Wick-contraction helpers
# ------------------------------------------------------------

PRETTY_INDICES = {
    "above": "efgh",
    "below": "mnop",
    "general": "qrstuv",
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


def two_body_expectation(p, q, r, s):
    """Derive <Psi|a_p^dagger a_q^dagger a_s a_r|Psi>."""

    operator = Fd(p) * Fd(q) * F(s) * F(r)
    return contract(cis_bra * operator * cis_ket)


# ============================================================
# CIS norm
# ============================================================

singles_norm = contract(C1_bra * C1_ket)
cis_norm = c0 * conjugate(c0) + singles_norm


# ============================================================
# Two-body RDM blocks
# ============================================================
#
# Each block name lists the lower creation pair followed by the upper
# annihilation pair, consistently with Gamma^{rs}_{pq}.
# ============================================================

P_ij = PermutationOperator(i, j)
P_ab = PermutationOperator(a, b)
P_kl = PermutationOperator(k, l)
P_cd = PermutationOperator(c, d)

BLOCK_SPECS = (
    ("ij_kl", r"\Gamma^{kl}_{ij}", (i, j, k, l), (P_ij, P_kl)),
    ("ij_kc", r"\Gamma^{kc}_{ij}", (i, j, k, c), (P_ij,)),
    ("ij_cd", r"\Gamma^{cd}_{ij}", (i, j, c, d), (P_ij, P_cd)),
    ("ia_kl", r"\Gamma^{kl}_{ia}", (i, a, k, l), (P_kl,)),
    ("ia_kc", r"\Gamma^{kc}_{ia}", (i, a, k, c), ()),
    ("ia_cd", r"\Gamma^{cd}_{ia}", (i, a, c, d), (P_cd,)),
    ("ab_kl", r"\Gamma^{kl}_{ab}", (a, b, k, l), (P_ab, P_kl)),
    ("ab_kc", r"\Gamma^{kc}_{ab}", (a, b, k, c), (P_ab,)),
    ("ab_cd", r"\Gamma^{cd}_{ab}", (a, b, c, d), (P_ab, P_cd)),
)

EXPANDED_BLOCKS = {
    key: two_body_expectation(*indices)
    for key, _, indices, _ in BLOCK_SPECS
}

COMPACT_BLOCKS = {
    key: simplify_index_permutations(EXPANDED_BLOCKS[key], permutations)
    if permutations
    else EXPANDED_BLOCKS[key]
    for key, _, _, permutations in BLOCK_SPECS
}

NONZERO_BLOCKS = tuple(
    (label, COMPACT_BLOCKS[key])
    for key, label, _, _ in BLOCK_SPECS
    if EXPANDED_BLOCKS[key] != 0
)

ZERO_BLOCKS = tuple(
    key
    for key, _, _, _ in BLOCK_SPECS
    if EXPANDED_BLOCKS[key] == 0
)

assert ZERO_BLOCKS == ("ij_cd", "ia_cd", "ab_kl", "ab_kc", "ab_cd")


# ============================================================
# LaTeX presentation
# ============================================================


class CISTwoRDMPrinter(LatexPrinter):
    """Render CIS coefficients and leading permutation operators cleanly."""

    def _print_Symbol(self, expr):
        if expr == c0:
            return r"c_{0}"
        return super()._print_Symbol(expr)

    def _print_Mul(self, expr):
        permutations = sorted(
            (
                factor
                for factor in expr.args
                if isinstance(factor, PermutationOperator)
            ),
            key=str,
        )
        if not permutations:
            return super()._print_Mul(expr)

        operator_product = Mul(*permutations)
        operand_latex = self._print(expr / operator_product)
        sign = ""
        if operand_latex.startswith("-"):
            sign = "- "
            operand_latex = operand_latex[1:].lstrip()

        operators_latex = " ".join(self._print(op) for op in permutations)
        return f"{sign}{operators_latex} {operand_latex}"

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
    """Return copyable LaTeX for a derived CIS two-body RDM block."""

    return CISTwoRDMPrinter().doprint(expr)


def print_cis_2rdm():
    """Print the nonzero spin-orbital CIS two-body RDM blocks."""

    print(r"\begin{align}")
    for position, (block_name, expression) in enumerate(NONZERO_BLOCKS):
        row_ending = r" \\" if position < len(NONZERO_BLOCKS) - 1 else ""
        print(block_name + " &= " + latex_expression(expression) + row_ending)
    print(r"\end{align}")


if __name__ == "__main__":
    print_cis_2rdm()
