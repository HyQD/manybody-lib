from sympy import (
    Dummy,
    Function,
    Mul,
    Rational,
    S,
    expand,
    symbols,
)

from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.stringpict import prettyForm
from sympy.printing.latex import LatexPrinter

from sympy.physics.secondquant import (
    F,
    Fd,
    NO,
    AntiSymmetricTensor,
    wicks,
    evaluate_deltas,
    substitute_dummies,
)


# ============================================================
# Conventions
# ============================================================
#
# Reference-normal-ordered Hamiltonian:
#
#   H = Eref
#       + sum_pq f_pq {a_p^\dagger a_q}
#       + 1/4 sum_pqrs u_pqrs
#             {a_p^\dagger a_q^\dagger a_s a_r}
#
# where
#
#   u_pqrs = <pq || rs>
#
# CIS wavefunction:
#
#   |Psi> = (c0 + C1) |Phi0>
#
#   C1 = sum_ai c_i^a a_a^\dagger a_i
#
# Indices:
#   i,j,... : occupied spin orbitals
#   a,b,... : virtual spin orbitals
#   p,q,... : general spin orbitals
#
# No canonical-orbital assumption is made.
# ============================================================


# ------------------------------------------------------------
# External indices of sigma_ai
# ------------------------------------------------------------

a = symbols("a", above_fermi=True)
i = symbols("i", below_fermi=True)


# ------------------------------------------------------------
# Dummy indices
# ------------------------------------------------------------

# General Hamiltonian indices
p = Dummy("p")
q = Dummy("q")
r = Dummy("r")
s = Dummy("s")

# CIS excitation indices
b = Dummy("b", above_fermi=True)
j = Dummy("j", below_fermi=True)


# ------------------------------------------------------------
# Scalar/tensor symbols
# ------------------------------------------------------------

Eref, c0 = symbols("Eref c0")

# f(p,q) = normal-ordered one-body matrix
f = Function("f")

# c(a,i) = CIS single-excitation coefficient c_i^a
c = Function("c")


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

HN = H1 + H2


# ------------------------------------------------------------
# Reusable determinant/excitation helpers
# ------------------------------------------------------------


def make_excitation_operator(
    amplitude,
    virtual_indices,
    occupied_indices,
    prefactor=S.One,
):
    """Build an arbitrary-rank, normal-ordered excitation operator.

    The occupied annihilators are reversed so that rank ``n`` has the
    conventional form

        a_a^dagger ... a_n^dagger a_j ... a_i.

    ``prefactor`` is explicit because conventions differ.  For fully
    antisymmetric amplitudes summed over unrestricted indices, the usual
    rank-n value is ``1 / (n!)**2`` (1, 1/4, and 1/36 for C1--C3).
    """

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


# The same builders can be used for later CISD/CISDT extensions.  For example,
# C2 and C3 only require antisymmetric amplitudes, rank-two/rank-three index
# tuples, and prefactors Rational(1, 4) and Rational(1, 36), respectively.
C1 = make_excitation_operator(c(b, j), (b,), (j,))


# ------------------------------------------------------------
# Wick-contraction helper
# ------------------------------------------------------------

def contract(expr):
    """
    Return the fully contracted part of expr with respect to
    the Fermi vacuum/reference determinant.
    """

    expr = expand(expr)

    # Wick-expand and discard every term that is not fully contracted.
    expr = wicks(
        expr,
        keep_only_fully_contracted=True,
    )

    # Remove Kronecker deltas by carrying out the implied substitutions.
    expr = evaluate_deltas(expr)

    # Canonicalize equivalent dummy-index summations.
    expr = substitute_dummies(
        expr,
        new_indices=True,
        pretty_indices={
            "above": "bcdefgh",
            "below": "jklmno",
            "general": "pqrstuv",
        },
    )

    return expand(expr)


HAMILTONIAN_TERMS = (
    ("F_N", H1),
    ("W_N", H2),
)


def project_hamiltonian(bra, ket):
    """Contract every normal-ordered Hamiltonian term with ``bra``/``ket``.

    Keeping this rank-independent is the main extension point for CISD and
    CISDT: construct C2/C3 and the corresponding bra projectors, then pass
    each desired bra/ket pair through this function.
    """

    return {
        name: contract(bra * hamiltonian_term * ket)
        for name, hamiltonian_term in HAMILTONIAN_TERMS
    }


# ============================================================
# sigma_0
# ============================================================
#
# sigma_0 =
#
#   Eref c0
#   + <Phi0| H_N C1 |Phi0>
#
# H_N acting on c0|Phi0> gives zero after projection because
# H_N is normal ordered.
# ============================================================

sigma0_HC1 = project_hamiltonian(S.One, C1)
sigma0_FC1 = sigma0_HC1["F_N"]
sigma0_WC1 = sigma0_HC1["W_N"]

sigma0 = (
    Eref * c0
    + sigma0_FC1
    + sigma0_WC1
)


# ============================================================
# sigma_ai
# ============================================================
#
# <Phi_i^a| = <Phi0| a_i^\dagger a_a
#
# Therefore the projector is
#
#     a_i^\dagger a_a
#
# on the left.
# ============================================================

bra_ai = make_bra_projector((a,), (i,))


# H_N acting on reference
sigma1_H0 = project_hamiltonian(bra_ai, c0)
sigma1_F0 = sigma1_H0["F_N"]
sigma1_W0 = sigma1_H0["W_N"]


# H_N acting on C1
sigma1_HC1 = project_hamiltonian(bra_ai, C1)
sigma1_FC1 = sigma1_HC1["F_N"]
sigma1_WC1 = sigma1_HC1["W_N"]


# Eref acting on C1 gives Eref c_i^a
sigma_ai = (
    Eref * c(a, i)
    + sigma1_F0
    + sigma1_W0
    + sigma1_FC1
    + sigma1_WC1
)


# ============================================================
# Presentation helpers
# ============================================================


class CleanSecondQuantPrettyPrinter(PrettyPrinter):
    """Pretty-printer that hides underscores on tensor Dummy indices."""

    def _print_AntiSymmetricTensor(self, expr):
        # AntiSymmetricTensor otherwise falls back to ``str(expr)``, which
        # exposes Dummy indices as _j and _b.  Formatting the existing tensor
        # directly is important: reconstructing it with Symbols can trigger a
        # new antisymmetric canonicalization and move a sign.
        upper = ", ".join(index.name for index in expr.upper)
        lower = ", ".join(index.name for index in expr.lower)
        return prettyForm(f"{expr.symbol}(({upper}),({lower}))")


class CISLatexPrinter(LatexPrinter):
    """Render the CIS expressions in conventional tensor notation."""

    def _print(self, expr, **kwargs):
        # AntiSymmetricTensor defines its own ``_latex`` method, which takes
        # precedence over normal printer dispatch.  Intercept only that type
        # so the stored tensor is formatted without reconstructing it.
        if isinstance(expr, AntiSymmetricTensor):
            return self._print_AntiSymmetricTensor(expr)
        return super()._print(expr, **kwargs)

    def _print_Symbol(self, expr):
        if expr == Eref:
            return r"E_{\mathrm{ref}}"
        if expr == c0:
            return r"c_{0}"
        return super()._print_Symbol(expr)

    def _print_Function(self, expr, exp=None):
        if expr.func == c and len(expr.args) == 2:
            virtual, occupied = expr.args
            amplitude = (
                rf"c_{{{self._print(occupied)}}}"
                rf"^{{{self._print(virtual)}}}"
            )
            return self._do_exponent(amplitude, exp)
        if expr.func == f and len(expr.args) == 2:
            left, right = expr.args
            matrix_element = (
                rf"f_{{{self._print(left)}{self._print(right)}}}"
            )
            return self._do_exponent(matrix_element, exp)
        return super()._print_Function(expr, exp)

    def _print_AntiSymmetricTensor(self, expr):
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
    """Return copyable LaTeX using conventional CIS index placement."""

    return CISLatexPrinter().doprint(expr)


def print_expression(expr):
    """Pretty-print an expression using clean Einstein-summation labels."""

    print(pretty_expression(expr))


def print_cis_results():
    """Print the CIS sigma-vector contributions and final expressions."""

    print("\n========================================")
    print("sigma_0 contributions")
    print("========================================\n")

    print("Eref * c0:")
    print_expression(Eref * c0)

    print("\nF_N C1:")
    print_expression(sigma0_FC1)

    print("\nW_N C1:")
    print_expression(sigma0_WC1)

    print("\nFull sigma_0:")
    print_expression(sigma0)

    print("\n\n========================================")
    print("sigma_ai contributions")
    print("========================================\n")

    print("Eref * c_i^a:")
    print_expression(Eref * c(a, i))

    print("\nc0 * F_N:")
    print_expression(sigma1_F0)

    print("\nc0 * W_N:")
    print_expression(sigma1_W0)

    print("\nF_N C1:")
    print_expression(sigma1_FC1)

    print("\nW_N C1:")
    print_expression(sigma1_WC1)

    print("\nFull sigma_ai:")
    print_expression(sigma_ai)

    print("\n\n========================================")
    print("LaTeX equations")
    print("========================================\n")

    print(r"\begin{align}")
    print(r"\sigma_0 &= " + latex_expression(sigma0) + r" \\")
    print(r"\sigma_{ai} &= " + latex_expression(sigma_ai))
    print(r"\end{align}")


if __name__ == "__main__":
    print_cis_results()
