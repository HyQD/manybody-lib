from sympy import Dummy, Function, Mul, Rational, expand, symbols

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
# Reference-normal-ordered Hamiltonian:
#
#   H = Eref
#       + sum_pq f_pq {a_p^dagger a_q}
#       + 1/4 sum_pqrs u_pqrs
#             {a_p^dagger a_q^dagger a_s a_r}
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
# The two-electron integrals and doubles amplitudes are antisymmetric within
# each index pair.  No canonical-orbital assumption is made.
# ============================================================


# ------------------------------------------------------------
# External sigma-vector indices
# ------------------------------------------------------------

a, b = symbols("a b", above_fermi=True)
i, j = symbols("i j", below_fermi=True)


# ------------------------------------------------------------
# Dummy indices
# ------------------------------------------------------------

p = Dummy("p")
q = Dummy("q")
r = Dummy("r")
s = Dummy("s")

c = Dummy("c", above_fermi=True)
d = Dummy("d", above_fermi=True)
k = Dummy("k", below_fermi=True)
l = Dummy("l", below_fermi=True)


# ------------------------------------------------------------
# Hamiltonian and CISD wavefunction
# ------------------------------------------------------------

Eref, c0 = symbols("Eref c0")
f = Function("f")
c1 = Function("c")

u = AntiSymmetricTensor("u", (p, q), (r, s))
c2 = AntiSymmetricTensor("c", (c, d), (k, l))

H1 = f(p, q) * NO(Fd(p) * F(q))
H2 = Rational(1, 4) * u * NO(Fd(p) * Fd(q) * F(s) * F(r))

C1 = c1(c, k) * NO(Fd(c) * F(k))
C2 = Rational(1, 4) * c2 * NO(Fd(c) * Fd(d) * F(l) * F(k))

cisd_ket = c0 + C1 + C2

bra_0 = 1
bra_ai = Fd(i) * F(a)
bra_abij = Fd(i) * Fd(j) * F(b) * F(a)


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


def project_sigma(bra):
    """Project the scalar, one-body, and two-body Hamiltonian pieces."""

    return {
        "E_ref": Eref * contract(bra * cisd_ket),
        "F_N": contract(bra * H1 * cisd_ket),
        "W_N": contract(bra * H2 * cisd_ket),
    }


# ============================================================
# CISD sigma vectors
# ============================================================

SIGMA0_COMPONENTS = project_sigma(bra_0)
SIGMA1_COMPONENTS = project_sigma(bra_ai)
SIGMA2_COMPONENTS = project_sigma(bra_abij)

sigma_0 = sum(SIGMA0_COMPONENTS.values())
sigma_ai = sum(SIGMA1_COMPONENTS.values())
sigma_abij_expanded = sum(SIGMA2_COMPONENTS.values())

P_ab = PermutationOperator(a, b)
P_ij = PermutationOperator(i, j)

sigma_abij = simplify_index_permutations(
    sigma_abij_expanded,
    (P_ab, P_ij),
)

EXPANDED_SIGMA = {
    "0": sigma_0,
    "ai": sigma_ai,
    "abij": sigma_abij_expanded,
}

SIGMA = {
    "0": sigma_0,
    "ai": sigma_ai,
    "abij": sigma_abij,
}


# ============================================================
# LaTeX presentation
# ============================================================


class CISDSigmaPrinter(LatexPrinter):
    """Render CISD tensors and leading permutation operators cleanly."""

    def _print(self, expr, **kwargs):
        if isinstance(expr, AntiSymmetricTensor):
            return self._print_AntiSymmetricTensor(expr)
        return super()._print(expr, **kwargs)

    def _print_Symbol(self, expr):
        if expr == Eref:
            return r"E_{\mathrm{ref}}"
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
        if expr.func == f and len(expr.args) == 2:
            left, right = expr.args
            matrix_element = rf"f_{{{self._print(left)}{self._print(right)}}}"
            return self._do_exponent(matrix_element, exp)
        return super()._print_Function(expr, exp)

    def _print_AntiSymmetricTensor(self, expr):
        if expr.symbol == c2.symbol:
            virtual = "".join(self._print(index) for index in expr.upper)
            occupied = "".join(self._print(index) for index in expr.lower)
            return rf"c_{{{occupied}}}^{{{virtual}}}"

        indices = (*expr.upper, *expr.lower)
        subscript = "".join(self._print(index) for index in indices)
        return rf"{self._print(expr.symbol)}_{{{subscript}}}"


def latex_expression(expr):
    """Return copyable LaTeX for a CISD sigma-vector expression."""

    return CISDSigmaPrinter().doprint(expr)


def print_cisd_sigma():
    """Print the spin-orbital CISD sigma vectors."""

    equations = (
        (r"\sigma_0", sigma_0),
        (r"\sigma_{i}^{a}", sigma_ai),
        (r"\sigma_{ij}^{ab}", sigma_abij),
    )

    print(r"\begin{align}")
    for position, (name, expression) in enumerate(equations):
        row_ending = r" \\" if position < len(equations) - 1 else ""
        print(name + " &= " + latex_expression(expression) + row_ending)
    print(r"\end{align}")


if __name__ == "__main__":
    print_cisd_sigma()
