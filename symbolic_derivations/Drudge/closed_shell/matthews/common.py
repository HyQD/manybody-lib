"""Shared machinery for closed-shell CISDT derivations.

The triples sector uses the non-orthogonal, column-symmetric orbital
(``skeleton``) manifold discussed by Matthews:

    X_3 = E_ai E_bj E_ck,
    T_3 = (1 / 3!) C_ijk^abc X_3.

The covariant projections ``<0|E_ia E_jb E_kc`` have a singular metric.
The public sigma vectors returned here are the metric-resolved quantities
``M_n^+ rho_n``.  At triples rank this is the Moore--Penrose representative
with the fully symmetric, wavefunction-null S3 component set to zero.

The implementation deliberately contracts individual excitation-rank
sectors.  It never constructs a monolithic ``bra * operator * ket`` tensor.
That is both faster and substantially less memory hungry in Drudge/Spark.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import itertools
import os
import sys
import time

from drudge import RestrictedPartHoleDrudge, Term
from drudge.report import ScalarLatexPrinter
from pyspark import SparkConf, SparkContext
from sympy import Indexed, IndexedBase, KroneckerDelta, Rational, Symbol


@dataclass(frozen=True)
class DerivationOptions:
    """Local Spark settings used by the derivation scripts."""

    workers: int = 4
    partitions: int = 4

    @classmethod
    def from_environment(cls) -> "DerivationOptions":
        workers = int(os.environ.get("DRUDGE_WORKERS", "4"))
        partitions = int(os.environ.get("DRUDGE_PARTITIONS", str(workers)))
        if workers < 1 or partitions < 1:
            raise ValueError("DRUDGE_WORKERS and DRUDGE_PARTITIONS must be positive")
        return cls(workers=workers, partitions=partitions)


class MatthewsLatexPrinter(ScalarLatexPrinter):
    """Print CI coefficients with stacked spatial-orbital indices."""

    _SYMBOLS = {
        "Eref": r"E_{\mathrm{ref}}",
        "C0": r"C_{0}",
        "C0tilde": r"\widetilde{C}_{0}",
    }
    _DUMMY_SYMBOLS = {
        **{f"a_{number}": label for number, label in enumerate("efghpqrs")},
        **{f"i_{number}": label for number, label in enumerate("mnopqrstu")},
    }

    def _print_Symbol(self, expr):  # noqa: N802 - SymPy printer API
        dummy = self._DUMMY_SYMBOLS.get(expr.name)
        if dummy is not None:
            return dummy
        replacement = self._SYMBOLS.get(expr.name)
        return replacement if replacement is not None else super()._print_Symbol(expr)

    def _print_Dummy(self, expr):  # noqa: N802 - SymPy printer API
        return self._print_Symbol(expr)

    def _print_Indexed(self, expr: Indexed):  # noqa: N802
        label = str(expr.base.label)
        indices = tuple(self._print(index) for index in expr.indices)

        if label.startswith("C") and not label.startswith("Ctilde"):
            rank = int(label.removeprefix("C"))
            upper = "".join(indices[:rank])
            lower = "".join(indices[rank:])
            return rf"C^{{{upper}}}_{{{lower}}}"
        if label.startswith("Ctilde"):
            rank = int(label.removeprefix("Ctilde"))
            upper = "".join(indices[:rank])
            lower = "".join(indices[rank:])
            return rf"\widetilde{{C}}^{{{upper}}}_{{{lower}}}"
        if label == "f" and len(indices) == 2:
            return rf"f^{{{indices[0]}}}_{{{indices[1]}}}"
        if label == "u" and len(indices) == 4:
            upper = "".join(indices[:2])
            lower = "".join(indices[2:])
            return rf"u^{{{upper}}}_{{{lower}}}"

        return super()._print_Indexed(expr)


class MatthewsPartHoleDrudge(RestrictedPartHoleDrudge):
    """Restricted particle-hole Drudge with project-specific LaTeX."""

    @staticmethod
    def _latex_sympy(expr):
        return MatthewsLatexPrinter().doprint(expr)


@dataclass
class CISDTModel:
    """Operators, sectors, and external indices for one derivation run."""

    dr: MatthewsPartHoleDrudge
    options: DerivationOptions
    e_: object
    Eref: Symbol
    right: tuple
    left: tuple
    projectors: tuple
    C: tuple
    Ctilde: tuple
    virtual: tuple
    occupied: tuple


def make_spark_context(options: DerivationOptions, app_name: str) -> SparkContext:
    """Create a quiet, explicitly local Spark context."""

    conf = (
        SparkConf()
        .setMaster(f"local[{options.workers}]")
        .setAppName(app_name)
        .set("spark.ui.enabled", "false")
        .set("spark.ui.showConsoleProgress", "false")
        .set("spark.default.parallelism", str(options.partitions))
        .set("spark.sql.shuffle.partitions", str(options.partitions))
    )
    context = SparkContext(conf=conf)
    context.setLogLevel("ERROR")
    return context


def _product(dr, factors: Sequence, partitions: int, *, cache: bool = False):
    """Multiply tensors while collapsing Spark partitions after each factor."""

    if not factors:
        return dr.einst(1)
    result = factors[0]
    for factor in factors[1:]:
        result = (result * factor).simplify()
        result = result.repartition(partitions)
    if cache:
        result = result.cache()
    return result


def _e_product(dr, e_, pairs, partitions: int, *, cache: bool = False):
    return _product(
        dr,
        tuple(e_[p, q] for p, q in pairs),
        partitions,
        cache=cache,
    )


def build_model(context: SparkContext, options: DerivationOptions) -> CISDTModel:
    """Build the Matthews-manifold CISDT bra, ket, and projectors."""

    dr = MatthewsPartHoleDrudge(context, num_partitions=options.partitions)
    dr.full_simplify = False
    names = dr.names
    e_ = names.e_

    # Four external labels, followed by disjoint right- and left-state
    # summation labels.  Keeping these disjoint reduces reset-dummy work.
    virtual = tuple(names.V_dumms[:12])
    occupied = tuple(names.O_dumms[:12])
    a, b, c, _d = virtual[:4]
    i, j, k, _l = occupied[:4]
    rv = virtual[4:7]
    ro = occupied[4:7]
    lv = virtual[8:11]
    lo = occupied[8:11]

    Eref = Symbol("Eref")
    C0 = Symbol("C0")
    C0tilde = Symbol("C0tilde")
    C1, C2, C3 = (IndexedBase(f"C{rank}") for rank in (1, 2, 3))
    Ct1, Ct2, Ct3 = (
        IndexedBase(f"Ctilde{rank}") for rank in (1, 2, 3)
    )

    # Separate tensor bases are important: set_n_body_base applies a distinct
    # column symmetry at doubles and triples rank.
    dr.set_n_body_base(C2, 2)
    dr.set_n_body_base(C3, 3)
    dr.set_n_body_base(Ct2, 2)
    dr.set_n_body_base(Ct3, 3)

    x1 = e_[rv[0], ro[0]]
    x2 = _e_product(
        dr,
        e_,
        ((rv[0], ro[0]), (rv[1], ro[1])),
        options.partitions,
        cache=True,
    )
    x3 = _e_product(
        dr,
        e_,
        ((rv[0], ro[0]), (rv[1], ro[1]), (rv[2], ro[2])),
        options.partitions,
        cache=True,
    )
    right = (
        dr.einst(C0),
        dr.einst(C1[rv[0], ro[0]] * x1).simplify().cache(),
        dr.einst(
            Rational(1, 2) * C2[*rv[:2], *ro[:2]] * x2
        ).simplify().cache(),
        dr.einst(
            Rational(1, 6) * C3[*rv, *ro] * x3
        ).simplify().cache(),
    )

    y1 = e_[lo[0], lv[0]]
    y2 = _e_product(
        dr,
        e_,
        ((lo[0], lv[0]), (lo[1], lv[1])),
        options.partitions,
        cache=True,
    )
    y3 = _e_product(
        dr,
        e_,
        ((lo[0], lv[0]), (lo[1], lv[1]), (lo[2], lv[2])),
        options.partitions,
        cache=True,
    )
    left = (
        dr.einst(C0tilde),
        dr.einst(Ct1[lo[0], lv[0]] * y1).simplify().cache(),
        dr.einst(
            Rational(1, 2) * Ct2[*lo[:2], *lv[:2]] * y2
        ).simplify().cache(),
        dr.einst(
            Rational(1, 6) * Ct3[*lo, *lv] * y3
        ).simplify().cache(),
    )

    projectors = (
        dr.einst(1),
        e_[i, a],
        _e_product(
            dr,
            e_,
            ((i, a), (j, b)),
            options.partitions,
            cache=True,
        ),
        _e_product(
            dr,
            e_,
            ((i, a), (j, b), (k, c)),
            options.partitions,
            cache=True,
        ),
    )

    return CISDTModel(
        dr=dr,
        options=options,
        e_=e_,
        Eref=Eref,
        right=right,
        left=left,
        projectors=projectors,
        C=(C0, C1, C2, C3),
        Ctilde=(C0tilde, Ct1, Ct2, Ct3),
        virtual=(a, b, c, _d),
        occupied=(i, j, k, _l),
    )


def _sum_tensors(dr, tensors: Iterable):
    result = dr.create_tensor([])
    for tensor in tensors:
        result += tensor
    return result.simplify()


def _canonical_local_tensor(dr, term: Term):
    """Rebuild a collected term with canonical orbital-range objects.

    Spark deserialisation produces ``Range`` objects that compare equal to
    the keys in ``dr.dumms`` but are not the registry objects themselves.
    Drudge's rank-three dummy reconciliation is sensitive to that distinction
    after a term has made a worker--driver--worker round trip.  Rebinding each
    sum to the matching registry key prevents occupied/virtual range mixing.
    """

    registry_ranges = tuple(dr.dumms.value)
    sums = tuple(
        (
            dummy,
            next(
                canonical
                for canonical in registry_ranges
                if canonical == orbital_range
            ),
        )
        for dummy, orbital_range in term.sums
    )
    return dr.create_tensor([Term(sums, term.amp, term.vecs)])


def _matrix_element(model: CISDTModel, left, operator, right):
    """Evaluate one deliberately small left--operator--right sector."""

    product = _product(
        model.dr,
        (left, operator, right),
        model.options.partitions,
    )
    return product.eval_fermi_vev().simplify()


def _permutation_cycles(permutation: Sequence[int]) -> int:
    seen = [False] * len(permutation)
    cycles = 0
    for start in range(len(permutation)):
        if seen[start]:
            continue
        cycles += 1
        node = start
        while not seen[node]:
            seen[node] = True
            node = permutation[node]
    return cycles


def _permute_free_indices(tensor, indices: Sequence[Symbol], permutation):
    substitution = {
        index: indices[permutation[position]]
        for position, index in enumerate(indices)
    }
    return tensor.map(lambda term: term.subst(substitution)).simplify()


def _class_function(dr, tensor, indices: Sequence[Symbol], weights_by_cycles):
    terms = []
    for permutation in itertools.permutations(range(len(indices))):
        weight = weights_by_cycles[_permutation_cycles(permutation)]
        if weight:
            terms.append(
                _permute_free_indices(tensor, indices, permutation) * weight
            )
    return _sum_tensors(dr, terms)


_METRIC_PINV = {
    1: {1: Rational(1, 2)},
    2: {2: Rational(1, 3), 1: Rational(1, 6)},
    3: {
        3: Rational(17, 144),
        2: Rational(-1, 144),
        1: Rational(-7, 144),
    },
}


def metric_pinv(dr, tensor, occupied_indices: Sequence[Symbol]):
    """Apply the exact Matthews-manifold Moore--Penrose metric inverse."""

    rank = len(occupied_indices)
    return _class_function(dr, tensor, occupied_indices, _METRIC_PINV[rank])


def null_projector(dr, tensor, occupied_indices: Sequence[Symbol]):
    """Project a triples tensor onto the wavefunction-null trivial S3 irrep."""

    if len(occupied_indices) != 3:
        raise ValueError("the nonzero null projector exists only at triples rank")
    return _class_function(
        dr,
        tensor,
        occupied_indices,
        {1: Rational(1, 6), 2: Rational(1, 6), 3: Rational(1, 6)},
    )


def physical_triples(dr, tensor, occupied_indices: Sequence[Symbol]):
    """Return ``(1 - E_triv) tensor``, the canonical Matthews representative."""

    return (tensor - null_projector(dr, tensor, occupied_indices)).simplify()


def derive_sigma_rank(model: CISDTModel, bra_rank: int):
    """Derive one metric-resolved right CISDT sigma-vector rank."""

    if bra_rank not in range(4):
        raise ValueError("bra_rank must be 0, 1, 2, or 3")
    dr = model.dr
    a, b, c, _ = model.virtual
    i, j, k, _ = model.occupied
    external = ((), (i,), (i, j), (i, j, k))
    coefficient = (
        model.C[0],
        model.C[1][a, i],
        model.C[2][a, b, i, j],
        model.C[3][a, b, c, i, j, k],
    )

    started = time.perf_counter()
    projected_h = _product(
        dr,
        (model.projectors[bra_rank], dr.ham),
        model.options.partitions,
        cache=True,
    )
    contributions = []
    for ket_rank in range(max(0, bra_rank - 2), min(3, bra_rank + 2) + 1):
        sector_started = time.perf_counter()
        if bra_rank == ket_rank == 3:
            # This is by far the largest CI block.  Multiplying E^3, all 14
            # particle-hole pieces of H_N, and C3 E^3 in one Tensor creates a
            # pathological Spark expression graph.  Each Hamiltonian piece
            # is independent, so evaluate and simplify it before summation.
            #
            piece_contributions = []
            hamiltonian_terms = tuple(dr.ham.local_terms)
            for piece_index, term in enumerate(hamiltonian_terms):
                hamiltonian_piece = _canonical_local_tensor(dr, term)
                product = _product(
                    dr,
                    (
                        model.projectors[bra_rank],
                        hamiltonian_piece,
                        model.right[ket_rank],
                    ),
                    model.options.partitions,
                )
                piece = product.eval_fermi_vev().simplify()
                if piece.n_terms:
                    piece_contributions.append(piece)
                print(
                    f"rank 3 <- ket rank 3, H piece {piece_index + 1}/14: "
                    f"{piece.n_terms} terms",
                    file=sys.stderr,
                )
            contribution = _sum_tensors(dr, piece_contributions)
            contributions.append(contribution)
            print(
                f"rank 3 <- ket rank 3: {contribution.n_terms} terms in "
                f"{time.perf_counter() - sector_started:.1f} s",
                file=sys.stderr,
            )
            continue
        product = _product(
            dr,
            (projected_h, model.right[ket_rank]),
            model.options.partitions,
        )
        print(
            f"rank {bra_rank} <- ket rank {ket_rank}: Wick evaluation",
            file=sys.stderr,
        )
        contribution = product.eval_fermi_vev().simplify()
        contributions.append(contribution)
        print(
            f"rank {bra_rank} <- ket rank {ket_rank}: "
            f"{contribution.n_terms} terms in "
            f"{time.perf_counter() - sector_started:.1f} s",
            file=sys.stderr,
        )
    raw = _sum_tensors(dr, contributions)

    if bra_rank == 0:
        resolved = raw + dr.einst(model.Eref * coefficient[bra_rank])
    else:
        resolved = metric_pinv(dr, raw, external[bra_rank])
        reference = dr.einst(model.Eref * coefficient[bra_rank])
        if bra_rank == 3:
            reference = physical_triples(dr, reference, external[bra_rank])
        resolved = (resolved + reference).simplify()
    print(
        f"rank {bra_rank}: {raw.n_terms} covariant H_N terms -> "
        f"{resolved.n_terms} sigma terms in "
        f"{time.perf_counter() - started:.1f} s",
        file=sys.stderr,
    )
    return resolved, raw


def derive_sigma_vectors(model: CISDTModel):
    """Derive all four metric-resolved right CISDT sigma-vector ranks."""

    results = [derive_sigma_rank(model, rank) for rank in range(4)]
    sigma, raw_covariant = zip(*results)

    return tuple(sigma), tuple(raw_covariant)


def derive_norm(model: CISDTModel):
    """Derive the unnormalised Matthews-manifold CISDT norm."""

    return _sum_tensors(
        model.dr,
        (
            _matrix_element(
                model,
                model.left[rank],
                model.dr.einst(1),
                model.right[rank],
            )
            for rank in range(4)
        ),
    )


def expectation_value(model: CISDTModel, operator, grade: int):
    """Derive one RDM block using only excitation-rank-compatible sectors."""

    contributions = []
    for right_rank in range(4):
        left_rank = right_rank + grade
        if not 0 <= left_rank <= 3:
            continue
        contributions.append(
            _matrix_element(
                model,
                model.left[left_rank],
                operator,
                model.right[right_rank],
            )
        )
    return _sum_tensors(model.dr, contributions)


def one_body_operator(model: CISDTModel, p, q):
    return model.e_[p, q]


def two_body_operator(model: CISDTModel, p, q, r, s):
    return (
        model.e_[p, r] * model.e_[q, s]
        - KroneckerDelta(q, r) * model.e_[p, s]
    ).simplify().repartition(model.options.partitions)


def derive_one_rdm(model: CISDTModel):
    """Return all four spin-summed one-body RDM blocks."""

    a, b, _c, _d = model.virtual
    i, j, _k, _l = model.occupied
    specs = (
        (r"\gamma^{j}_{i}", i, j, 0),
        (r"\gamma^{a}_{i}", i, a, -1),
        (r"\gamma^{i}_{a}", a, i, +1),
        (r"\gamma^{b}_{a}", a, b, 0),
    )
    result = []
    for label, p, q, grade in specs:
        started = time.perf_counter()
        tensor = expectation_value(model, one_body_operator(model, p, q), grade)
        print(
            f"{label}: {tensor.n_terms} terms in "
            f"{time.perf_counter() - started:.1f} s",
            file=sys.stderr,
        )
        result.append((label, tensor))
    return tuple(result)


def derive_two_rdm(model: CISDTModel):
    """Return representatives of all ten unique spin-summed 2-RDM blocks."""

    a, b, c, d = model.virtual
    i, j, k, l = model.occupied
    specs = (
        (r"\Gamma^{kl}_{ij}", (i, j, k, l), 0),
        (r"\Gamma^{ak}_{ij}", (i, j, a, k), -1),
        (r"\Gamma^{jk}_{ai}", (a, i, j, k), +1),
        (r"\Gamma^{ab}_{ij}", (i, j, a, b), -2),
        (r"\Gamma^{ij}_{ab}", (a, b, i, j), +2),
        (r"\Gamma^{bj}_{ai}", (a, i, b, j), 0),
        (r"\Gamma^{bj}_{ia}", (i, a, b, j), 0),
        (r"\Gamma^{bc}_{ai}", (a, i, b, c), -1),
        (r"\Gamma^{ci}_{ab}", (a, b, c, i), +1),
        (r"\Gamma^{cd}_{ab}", (a, b, c, d), 0),
    )
    result = []
    for label, indices, grade in specs:
        started = time.perf_counter()
        tensor = expectation_value(
            model,
            two_body_operator(model, *indices),
            grade,
        )
        print(
            f"{label}: {tensor.n_terms} terms in "
            f"{time.perf_counter() - started:.1f} s",
            file=sys.stderr,
        )
        if tensor.n_terms:
            result.append((label, tensor))
    return tuple(result)


def _latex_lines(dr, tensor) -> list[str]:
    rendered = dr.format_latex(
        tensor,
        no_sum=True,
        sep_lines=tensor.n_terms > 4,
    ).strip()
    return rendered.split(r" \\ ") if tensor.n_terms > 4 else [rendered]


def print_align(dr, equations: Iterable[tuple[str, object]]) -> None:
    """Print tensors as a complete, copyable LaTeX align environment."""

    equations = [(lhs, rhs) for lhs, rhs in equations if rhs.n_terms]
    print(r"\begin{align}")
    for equation_index, (lhs, tensor) in enumerate(equations):
        lines = _latex_lines(dr, tensor)
        print(rf"{lhs} &= {lines[0]}", end="")
        for line in lines[1:]:
            print(r" \\")
            print(rf"  &\quad {line}", end="")
        print(r" \\" if equation_index + 1 < len(equations) else "")
    print(r"\end{align}")


def print_conventions() -> None:
    """Print a compact LaTeX statement of the triples metric convention."""

    print("% Matthews non-orthogonal triples manifold")
    print(
        r"% T_3=(1/6) C^{abc}_{ijk}E_{ai}E_{bj}E_{ck}; "
        r"C^{abc}_{ijk}=C^{\pi(abc)}_{\pi(ijk)}."
    )
    print(
        r"% Covariant projection: "
        r"\rho^{abc}_{ijk}=\langle\Phi_0|E_{ia}E_{jb}E_{kc}H|\Psi\rangle."
    )
    print(
        r"% Reported sigma: \sigma_3=M_3^+\rho_3, "
        r"M_3^+=E_{\rm sign}/24+E_{\rm std}/6, "
        r"E_{\rm triv}\sigma_3=0."
    )
    print("% RDMs are not divided by <Psi_tilde|Psi>.")
