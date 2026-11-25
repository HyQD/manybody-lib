from sympy import (
    Dummy,
    Function,
    Mul,
    Rational,
    S,
    expand,
    symbols,
)

from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.stringpict import prettyForm

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
# where
#
#   u_pqrs = <pq || rs>.
#
# CID wavefunction:
#
#   |Psi> = (c0 + C2) |Phi0>
#
#   C2 = 1/4 sum_cdkl c_kl^cd
#             a_c^dagger a_d^dagger a_l a_k
#
# Both u_pqrs and c_kl^cd are antisymmetric within each index pair.
# No canonical-orbital assumption is made.
# ============================================================


# ------------------------------------------------------------
# External indices of sigma_ij^ab
# ------------------------------------------------------------

a, b = symbols("a b", above_fermi=True)
i, j = symbols("i j", below_fermi=True)


# ------------------------------------------------------------
# Dummy indices
# ------------------------------------------------------------

# General Hamiltonian indices
p = Dummy("p")
q = Dummy("q")
r = Dummy("r")
s = Dummy("s")

# CID double-excitation indices
c = Dummy("c", above_fermi=True)
d = Dummy("d", above_fermi=True)
k = Dummy("k", below_fermi=True)
l = Dummy("l", below_fermi=True)


# ------------------------------------------------------------
# Scalar/tensor symbols
# ------------------------------------------------------------

Eref, c0 = symbols("Eref c0")

# f(p,q) = normal-ordered one-body matrix
f = Function("f")

# c_kl^cd = antisymmetric CID double-excitation coefficient
cid_coefficient = AntiSymmetricTensor(
    "c",
    (c, d),
    (k, l),
)


# ------------------------------------------------------------
# Hamiltonian
# ------------------------------------------------------------

H1 = f(p, q) * NO(
    Fd(p) * F(q)
)

u = AntiSymmetricTensor(
    "u",
    (p, q),
    (r, s),
)

H2 = Rational(1, 4) * u * NO(
    Fd(p) * Fd(q) * F(s) * F(r)
)

HAMILTONIAN_TERMS = (
    ("F_N", H1),
    ("W_N", H2),
)


# ------------------------------------------------------------
# Reusable determinant/excitation helpers
# ------------------------------------------------------------


def make_excitation_operator(
    amplitude,
    virtual_indices,
    occupied_indices,
    prefactor=S.One,
):
    """Build an arbitrary-rank, normal-ordered excitation operator."""

    virtual_indices = tuple(virtual_indices)
    occupied_indices = tuple(occupied_indices)

    if not virtual_indices:
        raise ValueError("An excitation operator must have positive rank")
    if len(virtual_indices) != len(occupied_indices):
        raise ValueError("Virtual and occupied index counts must match")

    operators = (
        [Fd(index) for index in virtual_indices]
        + [F(index) for index in reversed(occupied_indices)]
    )
    return prefactor * amplitude * NO(Mul(*operators))


def make_bra_projector(virtual_indices, occupied_indices):
    """Return the de-excitation string dual to an excited determinant."""

    virtual_indices = tuple(virtual_indices)
    occupied_indices = tuple(occupied_indices)

    if not virtual_indices:
        raise ValueError("A bra projector must have positive rank")
    if len(virtual_indices) != len(occupied_indices):
        raise ValueError("Virtual and occupied index counts must match")

    operators = (
        [Fd(index) for index in occupied_indices]
        + [F(index) for index in reversed(virtual_indices)]
    )
    return Mul(*operators)


C2 = make_excitation_operator(
    cid_coefficient,
    (c, d),
    (k, l),
    prefactor=Rational(1, 4),
)

bra_abij = make_bra_projector((a, b), (i, j))


# ------------------------------------------------------------
# Wick-contraction helpers
# ------------------------------------------------------------


def contract(expr):
    """Return the fully contracted part with respect to the reference."""

    expr = wicks(
        expand(expr),
        keep_only_fully_contracted=True,
    )
    expr = evaluate_deltas(expr)
    expr = substitute_dummies(
        expr,
        new_indices=True,
        pretty_indices={
            "above": "cdefgh",
            "below": "klmno",
            "general": "pqrstuv",
        },
    )
    return expand(expr)


def project_hamiltonian(bra, ket):
    """Contract every normal-ordered Hamiltonian term with ``bra``/``ket``."""

    return {
        name: contract(bra * hamiltonian_term * ket)
        for name, hamiltonian_term in HAMILTONIAN_TERMS
    }


# ============================================================
# sigma_0
# ============================================================
#
# sigma_0 = Eref c0 + <Phi0| H_N C2 |Phi0>.
# ============================================================

sigma0_HC2 = project_hamiltonian(S.One, C2)
sigma0_FC2 = sigma0_HC2["F_N"]
sigma0_WC2 = sigma0_HC2["W_N"]

sigma0 = (
    Eref * c0
    + sigma0_FC2
    + sigma0_WC2
)


# ============================================================
# sigma_ij^ab
# ============================================================
#
# <Phi_ij^ab| = <Phi0| a_i^dagger a_j^dagger a_b a_a.
# ============================================================

# Eref acting on C2 gives Eref c_ij^ab.  Deriving the overlap rather than
# inserting the amplitude by hand also verifies the 1/4 normalization of C2.
sigma2_EC2 = Eref * contract(bra_abij * C2)

# H_N acting on the reference
sigma2_H0 = project_hamiltonian(bra_abij, c0)
sigma2_F0 = sigma2_H0["F_N"]
sigma2_W0 = sigma2_H0["W_N"]

# H_N acting on C2
sigma2_HC2 = project_hamiltonian(bra_abij, C2)
sigma2_FC2 = sigma2_HC2["F_N"]
sigma2_WC2 = sigma2_HC2["W_N"]

sigma_abij_expanded = (
    sigma2_EC2
    + sigma2_F0
    + sigma2_W0
    + sigma2_FC2
    + sigma2_WC2
)


# ------------------------------------------------------------
# Compact permutation form
# ------------------------------------------------------------
#
# P(ab) X(a,b) = X(a,b) - X(b,a), and likewise for P(ij).
# Keep both the expanded Wick result and its compact permutation form so the
# simplification can be inspected without discarding any derived terms.

P_ab = PermutationOperator(a, b)
P_ij = PermutationOperator(i, j)
PERMUTATION_OPERATORS = (P_ab, P_ij)

sigma2_FC2_compact = simplify_index_permutations(
    sigma2_FC2,
    PERMUTATION_OPERATORS,
)
sigma2_WC2_compact = simplify_index_permutations(
    sigma2_WC2,
    PERMUTATION_OPERATORS,
)
sigma_abij = simplify_index_permutations(
    sigma_abij_expanded,
    PERMUTATION_OPERATORS,
)


# ============================================================
# Presentation helpers
# ============================================================


class CleanSecondQuantPrettyPrinter(PrettyPrinter):
    """Pretty-printer that hides underscores on tensor Dummy indices."""

    def _print_AntiSymmetricTensor(self, expr):
        upper = ", ".join(index.name for index in expr.upper)
        lower = ", ".join(index.name for index in expr.lower)
        return prettyForm(f"{expr.symbol}(({upper}),({lower}))")

    def _print_PermutationOperator(self, expr):
        left, right = expr.args
        return prettyForm(f"P({left.name}{right.name})")


class CIDLatexPrinter(LatexPrinter):
    """Render the CID expressions in conventional tensor notation."""

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
        permutation_operators = sorted(
            (
                factor
                for factor in expr.args
                if isinstance(factor, PermutationOperator)
            ),
            key=str,
        )
        if not permutation_operators:
            return super()._print_Mul(expr)

        operator_product = Mul(*permutation_operators)
        operand_latex = self._print(expr / operator_product)

        sign = ""
        if operand_latex.startswith("- "):
            sign = "- "
            operand_latex = operand_latex[2:]
        elif operand_latex.startswith("-"):
            sign = "- "
            operand_latex = operand_latex[1:].lstrip()

        operator_latex = " ".join(
            self._print(operator)
            for operator in permutation_operators
        )
        return f"{sign}{operator_latex} {operand_latex}"

    def _print_Function(self, expr, exp=None):
        if expr.func == f and len(expr.args) == 2:
            left, right = expr.args
            matrix_element = (
                rf"f_{{{self._print(left)}{self._print(right)}}}"
            )
            return self._do_exponent(matrix_element, exp)
        return super()._print_Function(expr, exp)

    def _print_AntiSymmetricTensor(self, expr):
        if expr.symbol == cid_coefficient.symbol:
            virtual = "".join(self._print(index) for index in expr.upper)
            occupied = "".join(self._print(index) for index in expr.lower)
            return rf"c_{{{occupied}}}^{{{virtual}}}"

        indices = (*expr.upper, *expr.lower)
        subscript = "".join(self._print(index) for index in indices)
        return rf"{self._print(expr.symbol)}_{{{subscript}}}"


def pretty_expression(expr):
    """Return a clean, presentation-only rendering of ``expr``."""

    printer = CleanSecondQuantPrettyPrinter(
        settings={"use_unicode": True, "wrap_line": False}
    )
    return printer.doprint(expr)


def latex_expression(expr):
    """Return copyable LaTeX using conventional CID index placement."""

    return CIDLatexPrinter().doprint(expr)


def print_expression(expr):
    """Pretty-print an expression using clean Einstein-summation labels."""

    print(pretty_expression(expr))


def print_cid_results():
    """Print the CID sigma-vector contributions and final expressions."""

    print("\n========================================")
    print("sigma_0 contributions")
    print("========================================\n")

    print("Eref * c0:")
    print_expression(Eref * c0)

    print("\nF_N C2:")
    print_expression(sigma0_FC2)

    print("\nW_N C2:")
    print_expression(sigma0_WC2)

    print("\nFull sigma_0:")
    print_expression(sigma0)

    print("\n\n========================================")
    print("sigma_ij^ab contributions")
    print("========================================\n")

    print("Eref * c_ij^ab:")
    print_expression(sigma2_EC2)

    print("\nc0 * F_N:")
    print_expression(sigma2_F0)

    print("\nc0 * W_N:")
    print_expression(sigma2_W0)

    print("\nF_N C2 (expanded):")
    print_expression(sigma2_FC2)

    print("\nF_N C2 (permutation form):")
    print_expression(sigma2_FC2_compact)

    print("\nW_N C2 (expanded):")
    print_expression(sigma2_WC2)

    print("\nW_N C2 (permutation form):")
    print_expression(sigma2_WC2_compact)

    print("\nFull sigma_ij^ab (expanded):")
    print_expression(sigma_abij_expanded)

    print("\nFull sigma_ij^ab (permutation form):")
    print_expression(sigma_abij)

    print("\n\n========================================")
    print("LaTeX equations")
    print("========================================\n")

    print(r"\begin{align}")
    print(r"\sigma_0 &= " + latex_expression(sigma0) + r" \\")
    print(r"\sigma_{ij}^{ab} &= " + latex_expression(sigma_abij))
    print(r"\end{align}")


if __name__ == "__main__":
    print_cid_results()
