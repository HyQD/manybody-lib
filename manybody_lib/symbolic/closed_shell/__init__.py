"""Closed-shell spin-free symbolic algebra.

The dependency-free core is exported here.  SymPy conversion is available
from :mod:`manybody_lib.symbolic.closed_shell.sympy_bridge` when the optional
``symbolic`` dependencies are installed.
"""

from .canonicalize import canonicalize_expression, canonicalize_term
from .commutators import atomic_commutator, commutator, multiply
from .equations import (
    RightSigmaVector,
    derive_right_sigma,
    right_sigma_to_latex,
)
from .expressions import Expression, Term
from .hamiltonian import (
    FOCK,
    REFERENCE_ENERGY,
    TWO_ELECTRON,
    build_effective_bare_hamiltonian,
    build_normal_ordered_hamiltonian,
)
from .indices import (
    Index,
    IndexRole,
    OrbitalSpace,
    general,
    occupied,
    spin,
    virtual,
)
from .operators import OneBodyOperator, OperatorWord, TwoBodyOperator
from .manifolds import (
    CIState,
    CIStateSector,
    DOUBLES_AMPLITUDE,
    REFERENCE_COEFFICIENT,
    SINGLES_AMPLITUDE,
    build_hjo_dual_projector,
    build_hjo_right_state,
    hjo_metric,
    right_excitation,
)
from .printers import expression_to_latex, term_to_latex
from .presentation import (
    ColumnPermutationPresentation,
    factor_column_permutation,
    permute_expression,
)
from .reference import (
    expand_two_body_operators,
    reference_expectation,
    reference_expectation_cache_clear,
    reference_expectation_cache_info,
)
from .scalars import ExactScalar, I, MINUS_ONE, ONE, ZERO
from .tensors import (
    Delta,
    OrbitalDimension,
    PermutationSymmetry,
    Tensor,
    TensorSymbol,
    antisymmetric_group_symmetries,
    simultaneous_column_symmetries,
)


__all__ = [
    "CIState",
    "CIStateSector",
    "ColumnPermutationPresentation",
    "Delta",
    "DOUBLES_AMPLITUDE",
    "ExactScalar",
    "Expression",
    "FOCK",
    "I",
    "Index",
    "IndexRole",
    "MINUS_ONE",
    "ONE",
    "OneBodyOperator",
    "OperatorWord",
    "OrbitalDimension",
    "OrbitalSpace",
    "PermutationSymmetry",
    "REFERENCE_ENERGY",
    "REFERENCE_COEFFICIENT",
    "RightSigmaVector",
    "SINGLES_AMPLITUDE",
    "Tensor",
    "TensorSymbol",
    "Term",
    "TWO_ELECTRON",
    "TwoBodyOperator",
    "ZERO",
    "antisymmetric_group_symmetries",
    "atomic_commutator",
    "build_effective_bare_hamiltonian",
    "build_hjo_dual_projector",
    "build_hjo_right_state",
    "build_normal_ordered_hamiltonian",
    "canonicalize_expression",
    "canonicalize_term",
    "commutator",
    "derive_right_sigma",
    "expand_two_body_operators",
    "expression_to_latex",
    "factor_column_permutation",
    "general",
    "hjo_metric",
    "multiply",
    "occupied",
    "permute_expression",
    "reference_expectation",
    "reference_expectation_cache_clear",
    "reference_expectation_cache_info",
    "simultaneous_column_symmetries",
    "spin",
    "term_to_latex",
    "right_excitation",
    "right_sigma_to_latex",
    "virtual",
]
