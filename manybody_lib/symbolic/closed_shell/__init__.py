"""Closed-shell spin-free symbolic algebra.

The dependency-free core is exported here.  SymPy conversion is available
from :mod:`manybody_lib.symbolic.closed_shell.sympy_bridge` when the optional
``symbolic`` dependencies are installed.
"""

from .canonicalize import canonicalize_expression, canonicalize_term
from .commutators import atomic_commutator, commutator, multiply
from .expressions import Expression, Term
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
from .printers import expression_to_latex, term_to_latex
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
    "Delta",
    "ExactScalar",
    "Expression",
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
    "Tensor",
    "TensorSymbol",
    "Term",
    "TwoBodyOperator",
    "ZERO",
    "antisymmetric_group_symmetries",
    "atomic_commutator",
    "canonicalize_expression",
    "canonicalize_term",
    "commutator",
    "expand_two_body_operators",
    "expression_to_latex",
    "general",
    "multiply",
    "occupied",
    "reference_expectation",
    "reference_expectation_cache_clear",
    "reference_expectation_cache_info",
    "simultaneous_column_symmetries",
    "spin",
    "term_to_latex",
    "virtual",
]
