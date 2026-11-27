"""Deterministic canonicalization for closed-shell symbolic expressions."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterable
import itertools

from .expressions import Expression, Term
from .indices import Index, IndexRole, OrbitalSpace, merged_space
from .operators import OneBodyOperator, OperatorWord, TwoBodyOperator
from .scalars import ExactScalar
from .tensors import Delta, OrbitalDimension, ScalarFactor, Tensor


IndexKey = Callable[[Index], tuple]


def _default_index_key(index: Index) -> tuple:
    return index.sort_key()


def _symbol_key(tensor: Tensor) -> tuple:
    symbol = tensor.symbol
    symmetries = tuple(
        (symmetry.permutation, symmetry.phase)
        for symmetry in symbol.symmetries
    )
    return (
        symbol.name,
        symbol.rank,
        symbol.upper_count,
        symbol.latex_name or "",
        symmetries,
        tensor.conjugated,
    )


def _tensor_orbit(tensor: Tensor) -> tuple[tuple[int, Tensor], ...] | None:
    """Return the complete signed tensor orbit, or ``None`` when it is zero."""

    if not tensor.symbol.symmetries:
        return ((1, tensor),)
    phases = {tensor.indices: 1}
    pending = deque((tensor.indices,))
    while pending:
        indices = pending.popleft()
        phase = phases[indices]
        for symmetry in tensor.symbol.symmetries:
            transformed = symmetry.apply(indices)
            transformed_phase = phase * symmetry.phase
            known = phases.get(transformed)
            if known is not None:
                if known != transformed_phase:
                    return None
                continue
            phases[transformed] = transformed_phase
            pending.append(transformed)
    return tuple(
        (
            phase,
            Tensor(tensor.symbol, indices, tensor.conjugated),
        )
        for indices, phase in phases.items()
    )


def _operator_key(operator, index_key: IndexKey) -> tuple:
    if isinstance(operator, OneBodyOperator):
        label = "E"
    elif isinstance(operator, TwoBodyOperator):
        label = "e"
    else:
        raise TypeError(f"unknown operator {type(operator).__name__}")
    return label, tuple(index_key(index) for index in operator.indices)


def _factor_key(factor: ScalarFactor, index_key: IndexKey) -> tuple:
    if isinstance(factor, Tensor):
        return (
            "tensor",
            _symbol_key(factor),
            tuple(index_key(index) for index in factor.indices),
        )
    if isinstance(factor, Delta):
        indices = sorted((factor.left, factor.right), key=index_key)
        return "delta", tuple(index_key(index) for index in indices)
    if isinstance(factor, OrbitalDimension):
        return "dimension", factor.space.value
    raise TypeError(f"unknown scalar factor {type(factor).__name__}")


def _replace_term_indices(term: Term, substitutions: dict[Index, Index]) -> Term:
    if not substitutions:
        return term
    return Term(
        term.coefficient,
        tuple(
            factor.replace_indices(substitutions)
            for factor in term.factors
        ),
        term.operators.replace_indices(substitutions),
    )


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[Index, Index] = {}

    def add(self, index: Index) -> None:
        self.parent.setdefault(index, index)

    def find(self, index: Index) -> Index:
        parent = self.parent[index]
        if parent != index:
            parent = self.find(parent)
            self.parent[index] = parent
        return parent

    def union(self, left: Index, right: Index) -> None:
        self.add(left)
        self.add(right)
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if right_root.sort_key() < left_root.sort_key():
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root

    def components(self) -> tuple[tuple[Index, ...], ...]:
        grouped: dict[Index, list[Index]] = defaultdict(list)
        for index in self.parent:
            grouped[self.find(index)].append(index)
        return tuple(
            tuple(sorted(component, key=Index.sort_key))
            for component in grouped.values()
        )


def _process_deltas(term: Term) -> Term | None:
    deltas = tuple(
        factor for factor in term.factors if isinstance(factor, Delta)
    )
    if not deltas:
        return term

    remaining = [
        factor for factor in term.factors if not isinstance(factor, Delta)
    ]
    outside_indices = {
        index
        for factor in remaining
        for index in factor.indices
    }
    outside_indices.update(term.operators.indices)

    union_find = _UnionFind()
    for delta in deltas:
        union_find.union(delta.left, delta.right)

    substitutions: dict[Index, Index] = {}
    preserved_deltas: list[Delta] = []
    dimensions: list[OrbitalDimension] = []

    for component in union_find.components():
        space = merged_space(index.space for index in component)
        if space is None:
            return None

        free = sorted(
            (index for index in component if index.is_free),
            key=Index.sort_key,
        )
        dummy = sorted(
            (index for index in component if index.is_dummy),
            key=Index.sort_key,
        )

        if free:
            anchor_source = free[0]
            anchor = Index(anchor_source.name, space, IndexRole.FREE)
            substitutions[anchor_source] = anchor
            for index in dummy:
                substitutions[index] = anchor
            for other in free[1:]:
                preserved_deltas.append(Delta(anchor, other))
            continue

        anchor_source = dummy[0]
        anchor = Index(anchor_source.name, space, IndexRole.DUMMY)
        for index in dummy:
            substitutions[index] = anchor
        if not any(index in outside_indices for index in component):
            dimensions.append(OrbitalDimension(space))

    replaced = _replace_term_indices(
        Term(
            term.coefficient,
            tuple(remaining + dimensions),
            term.operators,
        ),
        substitutions,
    )

    unique_deltas = set()
    for delta in preserved_deltas:
        replaced_delta = delta.replace_indices(substitutions)
        if replaced_delta.left == replaced_delta.right:
            continue
        ordered = tuple(sorted(replaced_delta.indices, key=Index.sort_key))
        unique_deltas.add(Delta(*ordered))

    return Term(
        replaced.coefficient,
        replaced.factors + tuple(unique_deltas),
        replaced.operators,
    )


def _index_descriptor(index: Index, labels: dict[Index, int]) -> tuple:
    if index.is_free:
        return "free", index.sort_key()
    return "dummy", index.space.value, labels.get(index, -1)


def _dummy_occurrences(term: Term, labels: dict[Index, int]) -> dict[Index, tuple]:
    occurrences: dict[Index, list[tuple]] = defaultdict(list)

    for factor in term.factors:
        if isinstance(factor, Tensor):
            context = ("tensor", _symbol_key(factor))
        elif isinstance(factor, Delta):
            context = ("delta",)
        else:
            continue
        indices = tuple(factor.indices)
        for slot, index in enumerate(indices):
            if not index.is_dummy:
                continue
            surroundings = tuple(
                ("self", other.space.value)
                if other == index
                else _index_descriptor(other, labels)
                for other in indices
            )
            occurrences[index].append((context, slot, surroundings))

    for position, operator in enumerate(term.operators.operators):
        context = (
            "operator",
            position,
            "E" if isinstance(operator, OneBodyOperator) else "e",
        )
        for slot, index in enumerate(operator.indices):
            if not index.is_dummy:
                continue
            surroundings = tuple(
                ("self", other.space.value)
                if other == index
                else _index_descriptor(other, labels)
                for other in operator.indices
            )
            occurrences[index].append((context, slot, surroundings))

    return {
        index: tuple(sorted(values))
        for index, values in occurrences.items()
    }


def _dummy_labels(term: Term) -> tuple[dict[Index, int], dict[Index, tuple]]:
    labels = {
        dummy: list(OrbitalSpace).index(dummy.space)
        for dummy in term.dummies
    }
    signatures: dict[Index, tuple] = {}
    for _ in range(max(1, len(term.dummies) + 1)):
        signatures = _dummy_occurrences(term, labels)
        palette = {
            signature: position
            for position, signature in enumerate(sorted(set(signatures.values())))
        }
        updated = {
            dummy: palette[signatures[dummy]]
            for dummy in term.dummies
        }
        if updated == labels:
            break
        labels = updated
    return labels, signatures


_DUMMY_PREFIX = {
    OrbitalSpace.OCCUPIED: "i",
    OrbitalSpace.VIRTUAL: "a",
    OrbitalSpace.GENERAL: "p",
    OrbitalSpace.SPIN: "sigma",
}


def _dummy_renamings(term: Term) -> tuple[dict[Index, Index], ...]:
    """Return all unresolved color-class renamings to canonical dummy names."""

    if not term.dummies:
        return ({},)
    labels, signatures = _dummy_labels(term)
    groups: dict[tuple, list[Index]] = defaultdict(list)
    for index in term.dummies:
        groups[
            (
                list(OrbitalSpace).index(index.space),
                labels[index],
                signatures[index],
            )
        ].append(index)

    counters: dict[OrbitalSpace, int] = defaultdict(int)
    choices = []
    for key in sorted(groups):
        members = groups[key]
        space = members[0].space
        start = counters[space]
        targets = tuple(
            Index(
                f"{_DUMMY_PREFIX[space]}{position}",
                space,
                IndexRole.DUMMY,
            )
            for position in range(start, start + len(members))
        )
        counters[space] += len(members)
        choices.append(
            tuple(
                dict(zip(permutation, targets, strict=True))
                for permutation in itertools.permutations(members)
            )
        )

    renamings = []
    for selected in itertools.product(*choices):
        renaming = {}
        for partial in selected:
            renaming.update(partial)
        renamings.append(renaming)
    return tuple(renamings)


def _sort_and_deduplicate_factors(term: Term) -> Term:
    deltas = set()
    factors = []
    for factor in term.factors:
        if isinstance(factor, Delta):
            ordered = tuple(sorted(factor.indices, key=Index.sort_key))
            deltas.add(Delta(*ordered))
        else:
            factors.append(factor)
    factors.extend(deltas)
    factors.sort(key=lambda factor: _factor_key(factor, _default_index_key))
    return Term(term.coefficient, tuple(factors), term.operators)


def _symmetry_candidates(term: Term) -> tuple[Term, ...]:
    """Enumerate scalar-tensor slot-symmetry representatives of a term."""

    orbits = []
    for factor in term.factors:
        if isinstance(factor, Tensor):
            orbit = _tensor_orbit(factor)
            if orbit is None:
                return ()
            orbits.append(orbit)
        else:
            orbits.append(((1, factor),))

    candidates = []
    for choices in itertools.product(*orbits):
        phase = 1
        factors = []
        for factor_phase, factor in choices:
            phase *= factor_phase
            factors.append(factor)
        candidates.append(
            Term(term.coefficient * phase, tuple(factors), term.operators)
        )
    return tuple(candidates)


def _normalize_candidate(term: Term) -> Term:
    variants = (
        _sort_and_deduplicate_factors(_replace_term_indices(term, renaming))
        for renaming in _dummy_renamings(term)
    )
    return min(variants, key=_term_sort_key)


def canonicalize_term(term: Term) -> Term | None:
    """Return a deterministic equivalent term, or ``None`` when it vanishes."""

    if term.coefficient.is_zero:
        return None
    result = _process_deltas(term)
    if result is None:
        return None
    candidates = _symmetry_candidates(result)
    if not candidates:
        return None

    by_body: dict[tuple, Term] = {}
    for candidate in candidates:
        normalized = _normalize_candidate(candidate)
        previous = by_body.get(normalized.body_key)
        if previous is not None and previous.coefficient != normalized.coefficient:
            return None
        by_body[normalized.body_key] = normalized

    result = min(by_body.values(), key=_term_sort_key)
    return None if result.coefficient.is_zero else result


def _term_sort_key(term: Term) -> tuple:
    return (
        tuple(
            _factor_key(factor, _default_index_key)
            for factor in term.factors
        ),
        tuple(
            _operator_key(operator, _default_index_key)
            for operator in term.operators.operators
        ),
    )


def canonicalize_expression(terms: Iterable[Term] | Expression) -> Expression:
    """Canonicalize terms, collect exact coefficients, and sort the result."""

    if isinstance(terms, Expression):
        terms = terms.terms
    bodies: dict[tuple, tuple[Term, ExactScalar]] = {}
    for term in terms:
        canonical = canonicalize_term(term)
        if canonical is None:
            continue
        body = canonical.body_key
        if body in bodies:
            representative, coefficient = bodies[body]
            bodies[body] = (
                representative,
                coefficient + canonical.coefficient,
            )
        else:
            bodies[body] = (canonical, canonical.coefficient)

    collected = [
        representative.with_coefficient(coefficient)
        for representative, coefficient in bodies.values()
        if not coefficient.is_zero
    ]
    collected.sort(key=_term_sort_key)
    return Expression(tuple(collected))
