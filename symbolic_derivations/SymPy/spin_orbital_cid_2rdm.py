from sympy import (
    Dummy,
    KroneckerDelta,
    Mul,
    Rational,
    conjugate,
    expand,
    symbols,
)

from sympy.printing.latex import LatexPrinter

from sympy.physics.secondquant import (
    F,
    Fd,
    NO,
    AntiSymmetricTensor,
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
# CID wavefunction:
#
#   |Psi> = (c0 + C2) |Phi0>
#
#   C2 = 1/4 sum_abij c_ij^ab
#             a_a^dagger a_b^dagger a_j a_i
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
g = Dummy("g", above_fermi=True)
h = Dummy("h", above_fermi=True)

m = Dummy("m", below_fermi=True)
n = Dummy("n", below_fermi=True)
o = Dummy("o", below_fermi=True)
p = Dummy("p", below_fermi=True)


# ------------------------------------------------------------
# CID coefficients and wavefunction factors
# ------------------------------------------------------------

c0 = symbols("c0")

c2_bra_coefficient = AntiSymmetricTensor("c", (e, f), (m, n))
c2_ket_coefficient = AntiSymmetricTensor("c", (g, h), (o, p))

C2_ket = Rational(1, 4) * c2_ket_coefficient * NO(
    Fd(g) * Fd(h) * F(p) * F(o)
)

C2_bra = Rational(1, 4) * conjugate(c2_bra_coefficient) * (
    Fd(m) * Fd(n) * F(f) * F(e)
)

cid_ket = c0 + C2_ket
cid_bra = conjugate(c0) + C2_bra


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


def two_body_expectation(q, r, s, t):
    """Derive <Psi|a_q^dagger a_r^dagger a_t a_s|Psi>."""

    operator = Fd(q) * Fd(r) * F(t) * F(s)
    return contract(cid_bra * operator * cid_ket)


# ============================================================
# CID norm
# ============================================================

doubles_norm = contract(C2_bra * C2_ket)
cid_norm = c0 * conjugate(c0) + doubles_norm


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

assert ZERO_BLOCKS == ("ij_kc", "ia_kl", "ia_cd", "ab_kc")


# ============================================================
# LaTeX presentation
# ============================================================


class CIDTwoRDMPrinter(LatexPrinter):
    """Render CID tensors and leading permutation operators cleanly."""

    def _print(self, expr, **kwargs):
        if isinstance(expr, AntiSymmetricTensor):
            return self._print_AntiSymmetricTensor(expr)
        return super()._print(expr, **kwargs)

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
    """Return copyable LaTeX for a derived CID two-body RDM block."""

    return CIDTwoRDMPrinter().doprint(expr)


def print_cid_2rdm():
    """Print the nonzero spin-orbital CID two-body RDM blocks."""

    print(r"\begin{align}")
    for position, (block_name, expression) in enumerate(NONZERO_BLOCKS):
        row_ending = r" \\" if position < len(NONZERO_BLOCKS) - 1 else ""
        print(block_name + " &= " + latex_expression(expression) + row_ending)
    print(r"\end{align}")


if __name__ == "__main__":
    print_cid_2rdm()
