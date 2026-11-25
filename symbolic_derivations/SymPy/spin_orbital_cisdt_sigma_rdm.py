from collections import Counter
from itertools import permutations

from sympy import (
    Add,
    Dummy,
    Expr,
    Function,
    Mul,
    Rational,
    Tuple,
    conjugate,
    default_sort_key,
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
# where u_pqrs = <pq || rs>.
#
# CISDT wavefunction:
#
#   |Psi> = (c0 + C1 + C2 + C3) |Phi0>
#
#   C1 = sum_ai c_i^a a_a^dagger a_i
#
#   C2 = 1/4 sum_abij c_ij^ab
#             a_a^dagger a_b^dagger a_j a_i
#
#   C3 = 1/36 sum_abcijk c_ijk^abc
#             a_a^dagger a_b^dagger a_c^dagger a_k a_j a_i
#
# The doubles and triples amplitudes are antisymmetric separately in their
# virtual and occupied indices.  No canonical-orbital assumption is made.
#
# RDM convention:
#
#   gamma^q_p = <Psi|a_p^dagger a_q|Psi>
#
#   Gamma^{rs}_{pq}
#       = <Psi|a_p^dagger a_q^dagger a_s a_r|Psi>
#
# Bra amplitudes are conjugated explicitly.  No normalization relation is
# imposed and neither RDM is divided by <Psi|Psi>.
# ============================================================


# ------------------------------------------------------------
# External indices
# ------------------------------------------------------------

a, b, c, d = symbols("a b c d", above_fermi=True)
i, j, k, l = symbols("i j k l", below_fermi=True)


# ------------------------------------------------------------
# General Hamiltonian indices
# ------------------------------------------------------------

p = Dummy("p")
q = Dummy("q")
r = Dummy("r")
s = Dummy("s")


# ------------------------------------------------------------
# Independent bra/ket excitation indices
# ------------------------------------------------------------

e = Dummy("e", above_fermi=True)
f = Dummy("f", above_fermi=True)
g = Dummy("g", above_fermi=True)

x = Dummy("x", above_fermi=True)
y = Dummy("y", above_fermi=True)
z = Dummy("z", above_fermi=True)

m = Dummy("m", below_fermi=True)
n = Dummy("n", below_fermi=True)
o = Dummy("o", below_fermi=True)

u_index = Dummy("u", below_fermi=True)
v = Dummy("v", below_fermi=True)
w = Dummy("w", below_fermi=True)


# ------------------------------------------------------------
# Hamiltonian and CISDT wavefunction
# ------------------------------------------------------------

Eref, c0 = symbols("Eref c0")
fock = Function("f")
c1 = Function("c")

two_electron = AntiSymmetricTensor("u", (p, q), (r, s))

c2_ket = AntiSymmetricTensor("c", (e, f), (m, n))
c3_ket = AntiSymmetricTensor("c", (e, f, g), (m, n, o))

c2_bra = AntiSymmetricTensor("c", (x, y), (u_index, v))
c3_bra = AntiSymmetricTensor("c", (x, y, z), (u_index, v, w))

H1 = fock(p, q) * NO(Fd(p) * F(q))
H2 = Rational(1, 4) * two_electron * NO(
    Fd(p) * Fd(q) * F(s) * F(r)
)

C1_ket = c1(e, m) * NO(Fd(e) * F(m))

C2_ket = Rational(1, 4) * c2_ket * NO(
    Fd(e) * Fd(f) * F(n) * F(m)
)

C3_ket = Rational(1, 36) * c3_ket * NO(
    Fd(e) * Fd(f) * Fd(g) * F(o) * F(n) * F(m)
)

C1_bra = conjugate(c1(x, u_index)) * Fd(u_index) * F(x)

C2_bra = Rational(1, 4) * conjugate(c2_bra) * (
    Fd(u_index) * Fd(v) * F(y) * F(x)
)

C3_bra = Rational(1, 36) * conjugate(c3_bra) * (
    Fd(u_index) * Fd(v) * Fd(w) * F(z) * F(y) * F(x)
)

cisdt_ket = c0 + C1_ket + C2_ket + C3_ket
cisdt_bra = conjugate(c0) + C1_bra + C2_bra + C3_bra


# ------------------------------------------------------------
# Wick-contraction helpers
# ------------------------------------------------------------

PRETTY_INDICES = {
    "above": "efghxyz",
    "below": "mnopuvw",
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


# ============================================================
# Grouped triple-index permutation operators
# ============================================================
#
# The slash notation follows the CCSDT convention discussed around Eq. 10.34
# in the attached chapter.  A slash separates groups of equivalent external
# lines.  For example,
#
#   P(a/bc) = 1 - P_ab - P_ac,
#
# while a vertical bar separates independent occupied and virtual actions:
#
#   P(ijk|a/bc) = P(ijk) P(a/bc).
#
# P(ijk) is the complete signed sum over all six permutations of i,j,k, so
# P(ijk|a/bc) contains 18 signed actions before intrinsic tensor symmetries
# are used.
# ============================================================


def permutation_parity(permuted, original):
    """Return the parity of ``permuted`` relative to ``original``."""

    order = [original.index(index) for index in permuted]
    inversions = sum(
        order[left] > order[right]
        for left in range(len(order))
        for right in range(left + 1, len(order))
    )
    return -1 if inversions % 2 else 1


class GroupedPermutationOperator(Expr):
    """Signed external-index permutations with slash-separated groups.

    Each side supports the triple-index patterns needed here:

    * one group: identity;
    * one singleton and one pair: three coset representatives;
    * three singleton groups: the complete six-term antisymmetrizer.

    Occupied and virtual actions are combined by a Cartesian product.
    """

    is_commutative = True

    def __new__(cls, occupied_groups=(), virtual_groups=()):
        occupied = Tuple(
            *(Tuple(*tuple(group)) for group in occupied_groups)
        )
        virtual = Tuple(
            *(Tuple(*tuple(group)) for group in virtual_groups)
        )
        return Expr.__new__(cls, occupied, virtual)

    @property
    def occupied_groups(self):
        return tuple(tuple(group) for group in self.args[0])

    @property
    def virtual_groups(self):
        return tuple(tuple(group) for group in self.args[1])

    @staticmethod
    def _side_actions(groups):
        if not groups:
            return ((1, {}),)

        labels = tuple(index for group in groups for index in group)
        if len(groups) == 1:
            return ((1, {}),)

        if all(len(group) == 1 for group in groups):
            actions = []
            for permuted in permutations(labels):
                mapping = dict(zip(labels, permuted))
                sign = permutation_parity(permuted, labels)
                actions.append((sign, mapping))
            return tuple(actions)

        singletons = [group for group in groups if len(group) == 1]
        pairs = [group for group in groups if len(group) == 2]
        if len(labels) == 3 and len(singletons) == 1 and len(pairs) == 1:
            singled = singletons[0][0]
            actions = [(1, {})]
            for equivalent in pairs[0]:
                actions.append(
                    (-1, {singled: equivalent, equivalent: singled})
                )
            return tuple(actions)

        raise NotImplementedError(
            "Only identity, singleton/pair, and full triple groups "
            "are implemented"
        )

    @property
    def actions(self):
        occupied_actions = self._side_actions(self.occupied_groups)
        virtual_actions = self._side_actions(self.virtual_groups)

        combined = []
        for occupied_sign, occupied_map in occupied_actions:
            for virtual_sign, virtual_map in virtual_actions:
                mapping = dict(occupied_map)
                mapping.update(virtual_map)
                combined.append(
                    (occupied_sign * virtual_sign, mapping)
                )
        return tuple(combined)

    def apply(self, expr):
        """Return the explicit signed permutation expansion of ``expr``."""

        return expand(
            Add(
                *(
                    sign * expr.xreplace(mapping)
                    for sign, mapping in self.actions
                )
            )
        )


P_k_ij = GroupedPermutationOperator(
    occupied_groups=((k,), (i, j)),
)

P_c_ab = GroupedPermutationOperator(
    virtual_groups=((c,), (a, b)),
)

P_k_ij_c_ab = GroupedPermutationOperator(
    occupied_groups=((k,), (i, j)),
    virtual_groups=((c,), (a, b)),
)

# Defined for inspection and reuse.  Nonlinear CCSDT terms need this full
# occupied antisymmetrizer, whereas the linear CI triples expressions below
# do not happen to require it.
P_ijk_a_bc = GroupedPermutationOperator(
    occupied_groups=((i,), (j,), (k,)),
    virtual_groups=((a,), (b, c)),
)

assert len(P_ijk_a_bc.actions) == 18


def expression_terms(expr):
    """Return a multiset of fully expanded additive terms."""

    return Counter(Add.make_args(expand(expr)))


def simplify_grouped_permutations(expr, operators):
    """Introduce grouped operators only for exact complete term orbits."""

    operators = tuple(
        sorted(operators, key=lambda op: len(op.actions), reverse=True)
    )
    remaining = expression_terms(expr)
    compact_terms = []

    while True:
        best = None
        bases = sorted(remaining.elements(), key=default_sort_key)

        for base in bases:
            for operator in operators:
                orbit = expression_terms(operator.apply(base))
                orbit_size = sum(orbit.values())
                if orbit_size < 2 or not (orbit <= remaining):
                    continue

                score = (orbit_size, len(operator.actions))
                if best is None or score > best[0]:
                    best = (score, operator, base, orbit)

        if best is None:
            break

        _, operator, base, orbit = best
        remaining -= orbit
        compact_terms.append(operator * base)

    return Add(*compact_terms, *remaining.elements())


def expand_permutation_operators(expr):
    """Expand pairwise and grouped permutation operators in ``expr``."""

    result = 0
    for term in Add.make_args(expand(expr)):
        if isinstance(term, Mul):
            pair_operators = [
                factor
                for factor in term.args
                if isinstance(factor, PermutationOperator)
            ]
            grouped_operators = [
                factor
                for factor in term.args
                if isinstance(factor, GroupedPermutationOperator)
            ]
            operators = pair_operators + grouped_operators
            base = Mul(
                *(
                    factor
                    for factor in term.args
                    if factor not in operators
                )
            )
        else:
            operators = []
            base = term

        expanded_term = base
        for operator in operators:
            if isinstance(operator, GroupedPermutationOperator):
                expanded_term = operator.apply(expanded_term)
            else:
                expanded_term = expand(
                    expanded_term + operator.get_permuted(expanded_term)
                )
        result += expanded_term

    return expand(result)


def expressions_equal(left, right):
    """Compare expressions after dummy canonicalization."""

    difference = canonicalize_dummies(expand(left - right))
    return simplify(difference) == 0


# ============================================================
# CISDT sigma vectors
# ============================================================

bra_0 = 1
bra_ai = Fd(i) * F(a)
bra_abij = Fd(i) * Fd(j) * F(b) * F(a)
bra_abcijk = Fd(i) * Fd(j) * Fd(k) * F(c) * F(b) * F(a)


def project_sigma(bra):
    """Project the scalar, one-body, and two-body Hamiltonian pieces."""

    return {
        "E_ref": Eref * contract(bra * cisdt_ket),
        "F_N": contract(bra * H1 * cisdt_ket),
        "W_N": contract(bra * H2 * cisdt_ket),
    }


SIGMA0_COMPONENTS = project_sigma(bra_0)
SIGMA1_COMPONENTS = project_sigma(bra_ai)
SIGMA2_COMPONENTS = project_sigma(bra_abij)
SIGMA3_COMPONENTS = project_sigma(bra_abcijk)

sigma_0 = sum(SIGMA0_COMPONENTS.values())
sigma_ai = sum(SIGMA1_COMPONENTS.values())
sigma_abij_expanded = sum(SIGMA2_COMPONENTS.values())
sigma_abcijk_expanded = sum(SIGMA3_COMPONENTS.values())

P_ab = PermutationOperator(a, b)
P_ij = PermutationOperator(i, j)

sigma_abij = simplify_index_permutations(
    sigma_abij_expanded,
    (P_ab, P_ij),
)

sigma_abcijk = simplify_grouped_permutations(
    sigma_abcijk_expanded,
    (P_k_ij_c_ab, P_k_ij, P_c_ab),
)

EXPANDED_SIGMA = {
    "0": sigma_0,
    "ai": sigma_ai,
    "abij": sigma_abij_expanded,
    "abcijk": sigma_abcijk_expanded,
}

SIGMA = {
    "0": sigma_0,
    "ai": sigma_ai,
    "abij": sigma_abij,
    "abcijk": sigma_abcijk,
}

assert expressions_equal(
    expand_permutation_operators(sigma_abij),
    sigma_abij_expanded,
)
assert expressions_equal(
    expand_permutation_operators(sigma_abcijk),
    sigma_abcijk_expanded,
)


# ============================================================
# CISDT norm and one-body RDM
# ============================================================

cisdt_norm = contract(cisdt_bra * cisdt_ket)


def one_body_expectation(creation_index, annihilation_index):
    """Derive <Psi|a_p^dagger a_q|Psi>."""

    operator = Fd(creation_index) * F(annihilation_index)
    return contract(cisdt_bra * operator * cisdt_ket)


ONE_RDM_SPECS = (
    ("ij", r"\gamma^{j}_{i}", (i, j)),
    ("ia", r"\gamma^{a}_{i}", (i, a)),
    ("ai", r"\gamma^{i}_{a}", (a, i)),
    ("ab", r"\gamma^{b}_{a}", (a, b)),
)

ONE_RDM_BLOCKS = {
    key: one_body_expectation(*indices)
    for key, _, indices in ONE_RDM_SPECS
}

NONZERO_ONE_RDM_BLOCKS = tuple(
    (label, ONE_RDM_BLOCKS[key])
    for key, label, _ in ONE_RDM_SPECS
    if ONE_RDM_BLOCKS[key] != 0
)

assert len(NONZERO_ONE_RDM_BLOCKS) == 4


# ============================================================
# Two-body RDM
# ============================================================


def two_body_expectation(q_index, r_index, s_index, t_index):
    """Derive <Psi|a_q^dagger a_r^dagger a_t a_s|Psi>."""

    operator = (
        Fd(q_index) * Fd(r_index) * F(t_index) * F(s_index)
    )
    return contract(cisdt_bra * operator * cisdt_ket)


P_kl = PermutationOperator(k, l)
P_cd = PermutationOperator(c, d)

TWO_RDM_SPECS = (
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

EXPANDED_TWO_RDM_BLOCKS = {
    key: two_body_expectation(*indices)
    for key, _, indices, _ in TWO_RDM_SPECS
}

TWO_RDM_BLOCKS = {
    key: simplify_index_permutations(
        EXPANDED_TWO_RDM_BLOCKS[key],
        permutation_operators,
    )
    if permutation_operators
    else EXPANDED_TWO_RDM_BLOCKS[key]
    for key, _, _, permutation_operators in TWO_RDM_SPECS
}

NONZERO_TWO_RDM_BLOCKS = tuple(
    (label, TWO_RDM_BLOCKS[key])
    for key, label, _, _ in TWO_RDM_SPECS
    if EXPANDED_TWO_RDM_BLOCKS[key] != 0
)

assert len(NONZERO_TWO_RDM_BLOCKS) == 9

for block_key, compact_block in TWO_RDM_BLOCKS.items():
    assert expressions_equal(
        expand_permutation_operators(compact_block),
        EXPANDED_TWO_RDM_BLOCKS[block_key],
    )


# ============================================================
# LaTeX presentation
# ============================================================


class CISDTLatexPrinter(LatexPrinter):
    """Render CISDT tensors and leading permutation operators cleanly."""

    def _print(self, expr, **kwargs):
        if isinstance(expr, AntiSymmetricTensor):
            return self._print_AntiSymmetricTensor(expr)
        if isinstance(expr, PermutationOperator):
            return self._print_PermutationOperator(expr)
        return super()._print(expr, **kwargs)

    def _print_Symbol(self, expr):
        if expr == Eref:
            return r"E_{\mathrm{ref}}"
        if expr == c0:
            return r"c_{0}"
        return super()._print_Symbol(expr)

    def _print_Mul(self, expr):
        permutation_types = (
            PermutationOperator,
            GroupedPermutationOperator,
        )
        permutation_operators = sorted(
            (
                factor
                for factor in expr.args
                if isinstance(factor, permutation_types)
            ),
            key=default_sort_key,
        )
        if not permutation_operators:
            return super()._print_Mul(expr)

        operand = Mul(
            *(
                factor
                for factor in expr.args
                if factor not in permutation_operators
            )
        )
        operand_latex = self._print(operand)

        sign = ""
        if operand_latex.startswith("-"):
            sign = "- "
            operand_latex = operand_latex[1:].lstrip()

        operator_latex = " ".join(
            self._print(operator)
            for operator in permutation_operators
        )
        return f"{sign}{operator_latex} {operand_latex}"

    def _print_Function(self, expr, exp=None):
        if expr.func == c1 and len(expr.args) == 2:
            virtual, occupied = expr.args
            amplitude = (
                rf"c_{{{self._print(occupied)}}}"
                rf"^{{{self._print(virtual)}}}"
            )
            return self._do_exponent(amplitude, exp)
        if expr.func == fock and len(expr.args) == 2:
            left, right = expr.args
            matrix_element = rf"f_{{{self._print(left)}{self._print(right)}}}"
            return self._do_exponent(matrix_element, exp)
        return super()._print_Function(expr, exp)

    def _print_AntiSymmetricTensor(self, expr):
        if str(expr.symbol) == "c":
            virtual = "".join(self._print(index) for index in expr.upper)
            occupied = "".join(self._print(index) for index in expr.lower)
            return rf"c_{{{occupied}}}^{{{virtual}}}"

        indices = (*expr.upper, *expr.lower)
        subscript = "".join(self._print(index) for index in indices)
        return rf"{self._print(expr.symbol)}_{{{subscript}}}"

    def _print_conjugate(self, expr, exp=None):
        conjugated = rf"\left({self._print(expr.args[0])}\right)^{{*}}"
        return self._do_exponent(conjugated, exp)

    def _print_KroneckerDelta(self, expr):
        left, right = expr.args
        return rf"\delta_{{{self._print(left)}{self._print(right)}}}"

    def _print_PermutationOperator(self, expr):
        left, right = expr.args
        indices = self._print(left) + self._print(right)
        return rf"\widehat{{P}}\left({indices}\right)"

    def _print_GroupedPermutationOperator(self, expr):
        def groups_latex(groups):
            return "/".join(
                "".join(self._print(index) for index in group)
                for group in groups
            )

        occupied = groups_latex(expr.occupied_groups)
        virtual = groups_latex(expr.virtual_groups)
        label = r"\mid ".join(
            side for side in (occupied, virtual) if side
        )
        return rf"\widehat{{P}}\left({label}\right)"


def latex_expression(expr):
    """Return copyable LaTeX for a derived CISDT expression."""

    return CISDTLatexPrinter().doprint(expr)


def print_align(equations):
    """Print labeled expressions in a LaTeX align environment."""

    print(r"\begin{align}")
    for position, (name, expression) in enumerate(equations):
        row_ending = r" \\" if position < len(equations) - 1 else ""
        print(name + " &= " + latex_expression(expression) + row_ending)
    print(r"\end{align}")


def print_cisdt_results():
    """Print CISDT sigma vectors and all nonzero 1-RDM/2-RDM blocks."""

    print("Grouped permutation definitions")
    print(r"\begin{align}")
    print(
        r"\widehat{P}(k/ij)"
        r" &= 1-\widehat{P}_{ki}-\widehat{P}_{kj} \\"
    )
    print(
        r"\widehat{P}(c/ab)"
        r" &= 1-\widehat{P}_{ca}-\widehat{P}_{cb} \\"
    )
    print(
        r"\widehat{P}(k/ij\mid c/ab)"
        r" &= \widehat{P}(k/ij)\widehat{P}(c/ab) \\"
    )
    print(
        r"\widehat{P}(ijk\mid a/bc)"
        r" &= \widehat{P}(ijk)\widehat{P}(a/bc) \\"
    )
    print(
        r"\widehat{P}(ijk)X_{ijk}"
        r" &= X_{ijk}-X_{jik}-X_{kji}-X_{ikj}+X_{jki}+X_{kij}"
    )
    print(r"\end{align}")

    print("CISDT sigma vectors")
    print_align(
        (
            (r"\sigma_0", sigma_0),
            (r"\sigma_i^a", sigma_ai),
            (r"\sigma_{ij}^{ab}", sigma_abij),
            (r"\sigma_{ijk}^{abc}", sigma_abcijk),
        )
    )

    print("\nCISDT one-body RDM")
    print_align(NONZERO_ONE_RDM_BLOCKS)

    print("\nCISDT two-body RDM")
    print_align(NONZERO_TWO_RDM_BLOCKS)


if __name__ == "__main__":
    print_cisdt_results()
