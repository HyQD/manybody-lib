"""Derive closed-shell CISD sigma vectors and RDMs with Drudge.

The orbital basis is spatial and the reference is a closed-shell determinant.
Following the biorthogonal formulation of Helgaker, Jorgensen, and Olsen,

    E_pq = sum_sigma a^dagger_(p sigma) a_(q sigma),
    X_i^a = E_ai,
    Y_a^i = 1/2 E_ia,
    X_ij^ab = E_ai E_bj,
    Y_ab^ij = 1/3 E_jb E_ia + 1/6 E_ib E_ja.

The redundant doubles amplitudes obey the column symmetry

    C_ij^ab = C_ji^ba,    Ctilde_ab^ij = Ctilde_ba^ji.

The right and left CISD states are represented as

    |Psi> = (
        C_0 + C_i^a X_i^a + 1/2 C_ij^ab X_ij^ab
    ) |Phi_0>,

    <Psi_tilde| = <Phi_0| (
        Ctilde_0 + Ctilde_a^i Y_a^i
        + 1/2 Ctilde_ab^ij Y_ab^ij
    ).

The left coefficients are kept independent during the derivation.  For the
Hermitian adjoint of the right CISD state, substitute

    Ctilde_0 = C_0^*,
    Ctilde_a^i = 2 (C_i^a)^*,
    Ctilde_ab^ij = (4 C_ij^ab - 2 C_ji^ab)^*.

No division by <Psi_tilde|Psi> is made in the reduced density matrices.  Norm
drift therefore remains explicit.

Run in the dedicated environment with

    conda run --no-capture-output -n drudge python cisd.py

The output is a collection of complete LaTeX ``align`` environments.
"""

from __future__ import annotations

from collections.abc import Iterable

from drudge import RestrictedPartHoleDrudge
from drudge.report import ScalarLatexPrinter
from pyspark import SparkConf, SparkContext
from sympy import Indexed, IndexedBase, KroneckerDelta, Rational, Symbol

from permutation import (
    factor_column_permutation,
    factor_exchange_integrals,
    latex_rhs_lines,
)


class HJOLatexPrinter(ScalarLatexPrinter):
    """Print the HJO tensors with stacked upper and lower indices."""

    _SYMBOLS = {
        "Eref": r"E_{\mathrm{ref}}",
        "C0": r"C_{0}",
        "C0tilde": r"\widetilde{C}_{0}",
    }

    def _print_Symbol(self, expr):  # noqa: N802 - SymPy printer API
        replacement = self._SYMBOLS.get(expr.name)
        if replacement is not None:
            return replacement
        return super()._print_Symbol(expr)

    def _print_Indexed(self, expr: Indexed):  # noqa: N802
        label = str(expr.base.label)
        indices = tuple(self._print(index) for index in expr.indices)

        if label == "C" and len(indices) == 2:
            return rf"C^{{{indices[0]}}}_{{{indices[1]}}}"
        if label == "C" and len(indices) == 4:
            upper = "".join(indices[:2])
            lower = "".join(indices[2:])
            return rf"C^{{{upper}}}_{{{lower}}}"
        if label == "Ctilde" and len(indices) == 2:
            return rf"\widetilde{{C}}^{{{indices[0]}}}_{{{indices[1]}}}"
        if label == "Ctilde" and len(indices) == 4:
            upper = "".join(indices[:2])
            lower = "".join(indices[2:])
            return rf"\widetilde{{C}}^{{{upper}}}_{{{lower}}}"
        if label == "f" and len(indices) == 2:
            return rf"f^{{{indices[0]}}}_{{{indices[1]}}}"
        if label == "u" and len(indices) == 4:
            upper = "".join(indices[:2])
            lower = "".join(indices[2:])
            return rf"u^{{{upper}}}_{{{lower}}}"
        if label == "L" and len(indices) == 4:
            upper = "".join(indices[:2])
            lower = "".join(indices[2:])
            return rf"L^{{{upper}}}_{{{lower}}}"

        return super()._print_Indexed(expr)


class HJOPartHoleDrudge(RestrictedPartHoleDrudge):
    """Restricted particle-hole Drudge with project-specific LaTeX."""

    @staticmethod
    def _latex_sympy(expr):
        return HJOLatexPrinter().doprint(expr)


def make_spark_context() -> SparkContext:
    """Create a local Spark context for the CISD derivation."""

    conf = (
        SparkConf()
        .setMaster("local[4]")
        .setAppName("closed-shell-biorthogonal-cisd")
        .set("spark.ui.enabled", "false")
        .set("spark.ui.showConsoleProgress", "false")
        .set("spark.default.parallelism", "4")
        .set("spark.sql.shuffle.partitions", "4")
    )
    context = SparkContext(conf=conf)
    context.setLogLevel("ERROR")
    return context


def one_body_rdm(dr, p, q, left, right):
    """Return gamma_p^q = <Psi_tilde| E_pq |Psi>."""

    operator_right = (dr.names.e_[p, q] * right).simplify()
    return (left * operator_right).eval_fermi_vev().simplify()


def two_body_rdm(dr, p, q, r, s, left, right):
    """Return Gamma_pq^rs = <Psi_tilde| e_pqrs |Psi>."""

    e_ = dr.names.e_
    two_body_operator = (
        e_[p, r] * e_[q, s]
        - KroneckerDelta(q, r) * e_[p, s]
    )
    operator_right = (two_body_operator * right).simplify()
    return (left * operator_right).eval_fermi_vev().simplify()


def print_align(dr, equations: Iterable[tuple[str, object]]) -> None:
    """Print nonzero tensors as a complete LaTeX align environment."""

    equations = [
        (lhs, tensor)
        for lhs, tensor in equations
        if tensor.n_terms != 0
    ]
    if not equations:
        return

    print(r"\begin{align}")
    for equation_index, (lhs, tensor) in enumerate(equations):
        rhs_lines = latex_rhs_lines(dr, tensor)

        print(rf"{lhs} &= {rhs_lines[0]}", end="")
        for rhs_line in rhs_lines[1:]:
            print(r" \\")
            print(rf"  &\quad {rhs_line}", end="")

        if equation_index + 1 < len(equations):
            print(r" \\")
        else:
            print()
    print(r"\end{align}")


def derive() -> None:
    """Derive and print the closed-shell biorthogonal CISD equations."""

    context = make_spark_context()
    try:
        dr = HJOPartHoleDrudge(context, num_partitions=4)
        dr.full_simplify = False

        names = dr.names
        e_ = names.e_
        a, b, c, d = names.V_dumms[:4]
        i, j, k, l = names.O_dumms[:4]

        Eref = Symbol("Eref")
        C0 = Symbol("C0")
        C0_tilde = Symbol("C0tilde")
        C = IndexedBase("C")
        C_tilde = IndexedBase("Ctilde")
        u = IndexedBase("u")
        L = IndexedBase("L")

        # The n-body symmetry applies only to valence-four instances of these
        # bases, leaving the singles instances unchanged.
        dr.set_n_body_base(C, 2)
        dr.set_n_body_base(C_tilde, 2)
        dr.set_n_body_base(L, 2)

        X1 = e_[a, i]
        Y1 = e_[i, a] / 2
        X2 = (e_[a, i] * e_[b, j]).simplify().cache()
        Y2 = (
            Rational(1, 3) * e_[j, b] * e_[i, a]
            + Rational(1, 6) * e_[i, b] * e_[j, a]
        ).simplify().cache()

        right = dr.einst(
            C0
            + C[a, i] * X1
            + Rational(1, 2) * C[a, b, i, j] * X2
        ).simplify().cache()
        left = dr.einst(
            C0_tilde
            + C_tilde[i, a] * Y1
            + Rational(1, 2) * C_tilde[i, j, a, b] * Y2
        ).simplify().cache()

        # dr.ham is the normal-ordered spin-free Hamiltonian without E_ref.
        hamiltonian = dr.ham + Eref
        h_right = (hamiltonian * right).simplify().repartition(
            4, cache=True
        )
        left_h = (left * hamiltonian).simplify().repartition(
            4, cache=True
        )

        norm = (left * right).eval_fermi_vev().simplify()

        sigma_0 = h_right.eval_fermi_vev().simplify()
        sigma_i_a = (Y1 * h_right).eval_fermi_vev().simplify()
        sigma_ij_ab = (Y2 * h_right).eval_fermi_vev().simplify()
        sigma_tilde_0 = left_h.eval_fermi_vev().simplify()
        sigma_tilde_a_i = (left_h * X1).eval_fermi_vev().simplify()
        sigma_tilde_ab_ij = (left_h * X2).eval_fermi_vev().simplify()

        sigma_0 = factor_exchange_integrals(
            dr,
            sigma_0,
            integral_base=u,
            exchange_base=L,
        )
        sigma_i_a = factor_exchange_integrals(
            dr,
            sigma_i_a,
            integral_base=u,
            exchange_base=L,
        )
        sigma_ij_ab = factor_exchange_integrals(
            dr,
            sigma_ij_ab,
            integral_base=u,
            exchange_base=L,
        )
        sigma_tilde_a_i = factor_exchange_integrals(
            dr,
            sigma_tilde_a_i,
            integral_base=u,
            exchange_base=L,
        )
        sigma_tilde_ab_ij = factor_exchange_integrals(
            dr,
            sigma_tilde_ab_ij,
            integral_base=u,
            exchange_base=L,
        )

        sigma_ij_ab = factor_column_permutation(
            dr,
            sigma_ij_ab,
            first_upper=a,
            second_upper=b,
            first_lower=i,
            second_lower=j,
            operator_latex=r"\mathcal{P}^{ab}_{ij}",
        )
        sigma_tilde_ab_ij = factor_column_permutation(
            dr,
            sigma_tilde_ab_ij,
            first_upper=i,
            second_upper=j,
            first_lower=a,
            second_lower=b,
            operator_latex=r"\mathcal{P}^{ij}_{ab}",
        )

        one_body_blocks = (
            (r"\gamma^{j}_{i}", one_body_rdm(dr, i, j, left, right)),
            (r"\gamma^{a}_{i}", one_body_rdm(dr, i, a, left, right)),
            (r"\gamma^{i}_{a}", one_body_rdm(dr, a, i, left, right)),
            (r"\gamma^{b}_{a}", one_body_rdm(dr, a, b, left, right)),
        )

        # e_pqrs = e_qpsr, so these ten blocks represent all unique spatial
        # occupied/virtual patterns.  Singles-doubles cross terms make every
        # representative potentially nonzero for CISD.
        gamma_ij_kl = two_body_rdm(dr, i, j, k, l, left, right)
        gamma_ij_kl = factor_column_permutation(
            dr,
            gamma_ij_kl,
            first_upper=k,
            second_upper=l,
            first_lower=i,
            second_lower=j,
            operator_latex=r"\mathcal{P}^{kl}_{ij}",
        )

        two_body_blocks = (
            (
                r"\Gamma^{kl}_{ij}",
                gamma_ij_kl,
            ),
            (
                r"\Gamma^{ak}_{ij}",
                two_body_rdm(dr, i, j, a, k, left, right),
            ),
            (
                r"\Gamma^{jk}_{ai}",
                two_body_rdm(dr, a, i, j, k, left, right),
            ),
            (
                r"\Gamma^{ab}_{ij}",
                two_body_rdm(dr, i, j, a, b, left, right),
            ),
            (
                r"\Gamma^{ij}_{ab}",
                two_body_rdm(dr, a, b, i, j, left, right),
            ),
            (
                r"\Gamma^{bj}_{ai}",
                two_body_rdm(dr, a, i, b, j, left, right),
            ),
            (
                r"\Gamma^{bj}_{ia}",
                two_body_rdm(dr, i, a, b, j, left, right),
            ),
            (
                r"\Gamma^{bc}_{ai}",
                two_body_rdm(dr, a, i, b, c, left, right),
            ),
            (
                r"\Gamma^{ci}_{ab}",
                two_body_rdm(dr, a, b, c, i, left, right),
            ),
            (
                r"\Gamma^{cd}_{ab}",
                two_body_rdm(dr, a, b, c, d, left, right),
            ),
        )

        print("% Closed-shell CISD in the HJO biorthogonal basis")
        print(
            "% Column symmetry: "
            r"C^{ab}_{ij}=C^{ba}_{ji}, "
            r"\widetilde{C}^{ij}_{ab}=\widetilde{C}^{ji}_{ba}."
        )
        print(
            "% For a Hermitian bra: "
            r"\widetilde{C}_{0}=C_{0}^{*}, "
            r"\widetilde{C}^{i}_{a}=2(C^{a}_{i})^{*}, "
            r"\widetilde{C}^{ij}_{ab}="
            r"\left(4C^{ab}_{ij}-2C^{ab}_{ji}\right)^{*}."
        )
        print("% No normalization denominator has been applied.")
        print()

        print("% Norm")
        print_align(
            dr,
            ((r"\langle\widetilde{\Psi}\vert\Psi\rangle", norm),),
        )
        print()

        print("% Right sigma vector")
        print_align(
            dr,
            (
                (r"\sigma_{0}", sigma_0),
                (r"\sigma^{a}_{i}", sigma_i_a),
                (r"\sigma^{ab}_{ij}", sigma_ij_ab),
            ),
        )
        print()

        print("% Left sigma vector")
        print_align(
            dr,
            (
                (r"\widetilde{\sigma}_{0}", sigma_tilde_0),
                (r"\widetilde{\sigma}^{i}_{a}", sigma_tilde_a_i),
                (r"\widetilde{\sigma}^{ij}_{ab}", sigma_tilde_ab_ij),
            ),
        )
        print()

        print("% Nonzero one-body reduced-density-matrix blocks")
        print_align(dr, one_body_blocks)
        print()

        print("% Nonzero unique two-body reduced-density-matrix blocks")
        print_align(dr, two_body_blocks)
    finally:
        context.stop()


if __name__ == "__main__":
    derive()
