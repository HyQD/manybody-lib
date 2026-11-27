"""Right sigma-vector generation for closed-shell truncated CI."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ..commutators import multiply
from ..expressions import Expression
from ..hamiltonian import build_normal_ordered_hamiltonian
from ..indices import occupied, virtual
from ..manifolds.hjo import build_hjo_dual_projector, build_hjo_right_state
from ..presentation import factor_column_permutation
from ..printers import expression_to_latex
from ..reference import reference_expectation


@dataclass(frozen=True, slots=True)
class RightSigmaVector:
    """Canonical right sigma components keyed by excitation rank."""

    method: str
    components: tuple[tuple[int, Expression], ...]

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(rank for rank, _ in self.components)

    def __getitem__(self, rank: int) -> Expression:
        for component_rank, expression in self.components:
            if component_rank == rank:
                return expression
        raise KeyError(rank)


@lru_cache(maxsize=3)
def derive_right_sigma(method: str) -> RightSigmaVector:
    """Derive the CIS, CID, or CISD right sigma vector exactly."""

    state = build_hjo_right_state(method)
    hamiltonian = build_normal_ordered_hamiltonian()
    h_right = multiply(hamiltonian, state.expression)
    components = []
    for rank in state.ranks:
        projector = build_hjo_dual_projector(rank)
        component = reference_expectation(multiply(projector, h_right))
        components.append((rank, component))
    return RightSigmaVector(state.method, tuple(components))


def _component_lhs(rank: int) -> str:
    labels = {
        0: r"\sigma_{0}",
        1: r"\sigma^{a}_{i}",
        2: r"\sigma^{ab}_{ij}",
    }
    return labels[rank]


def right_sigma_to_latex(
    sigma: RightSigmaVector,
    *,
    factor_permutations: bool = True,
) -> str:
    """Render a complete ``align`` environment for a right sigma vector."""

    equations = []
    for rank, expression in sigma.components:
        if rank == 2 and factor_permutations:
            rendered = factor_column_permutation(
                expression,
                first_upper=virtual("a"),
                second_upper=virtual("b"),
                first_lower=occupied("i"),
                second_lower=occupied("j"),
                operator_latex=r"\mathcal{P}^{ab}_{ij}",
            ).to_latex()
        else:
            rendered = expression_to_latex(expression)
        equations.append((_component_lhs(rank), rendered))

    lines = [r"\begin{align}"]
    for position, (lhs, rhs) in enumerate(equations):
        suffix = r" \\" if position + 1 < len(equations) else ""
        lines.append(rf"{lhs} &= {rhs}{suffix}")
    lines.append(r"\end{align}")
    return "\n".join(lines)
