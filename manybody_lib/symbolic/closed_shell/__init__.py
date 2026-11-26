"""Closed-shell spin-free symbolic algebra.

The dependency-free core is exported here.  SymPy conversion is available
from :mod:`manybody_lib.symbolic.closed_shell.sympy_bridge` when the optional
``symbolic`` dependencies are installed.
"""

from .canonicalize import canonicalize_expression, canonicalize_term
from .commutators import atomic_commutator, commutator, multiply
from .conjugation import conjugate_scalar_expression
from .equations import (
    CIRDMs,
    RDMBlock,
    RightSigmaVector,
    SymbolicQuotient,
    derive_ci_rdms,
    derive_one_rdm_block,
    derive_right_sigma,
    derive_two_rdm_block,
    normalize_by_norm,
    rdms_to_latex,
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
    CILeftState,
    CIState,
    CIStateSector,
    DOUBLES_AMPLITUDE,
    LEFT_DOUBLES_AMPLITUDE,
    LEFT_REFERENCE_COEFFICIENT,
    LEFT_SINGLES_AMPLITUDE,
    REFERENCE_COEFFICIENT,
    SINGLES_AMPLITUDE,
    build_hjo_dual_projector,
    build_hjo_left_state,
    build_hjo_right_state,
    hermitian_bra_substitution,
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
    "CILeftState",
    "CIRDMs",
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
    "LEFT_DOUBLES_AMPLITUDE",
    "LEFT_REFERENCE_COEFFICIENT",
    "LEFT_SINGLES_AMPLITUDE",
    "MINUS_ONE",
    "ONE",
    "OneBodyOperator",
    "OperatorWord",
    "OrbitalDimension",
    "OrbitalSpace",
    "PermutationSymmetry",
    "REFERENCE_ENERGY",
    "REFERENCE_COEFFICIENT",
    "RDMBlock",
    "RightSigmaVector",
    "SINGLES_AMPLITUDE",
    "SymbolicQuotient",
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
    "build_hjo_left_state",
    "build_hjo_right_state",
    "build_normal_ordered_hamiltonian",
    "canonicalize_expression",
    "canonicalize_term",
    "commutator",
    "conjugate_scalar_expression",
    "derive_ci_rdms",
    "derive_one_rdm_block",
    "derive_right_sigma",
    "derive_two_rdm_block",
    "expand_two_body_operators",
    "expression_to_latex",
    "factor_column_permutation",
    "general",
    "hjo_metric",
    "hermitian_bra_substitution",
    "multiply",
    "normalize_by_norm",
    "occupied",
    "permute_expression",
    "reference_expectation",
    "reference_expectation_cache_clear",
    "reference_expectation_cache_info",
    "rdms_to_latex",
    "simultaneous_column_symmetries",
    "spin",
    "term_to_latex",
    "right_excitation",
    "right_sigma_to_latex",
    "virtual",
]
