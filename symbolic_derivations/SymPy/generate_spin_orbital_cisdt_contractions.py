"""Generate NumPy contraction tables from the spin-orbital CISDT algebra."""

from __future__ import annotations

import argparse
from pathlib import Path
import pprint
import string
import sys

from sympy import Add, KroneckerDelta, Mul, conjugate
from sympy.core.function import AppliedUndef
from sympy.physics.secondquant import AntiSymmetricTensor


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symbolic_derivations.SymPy import (  # noqa: E402
    spin_orbital_cisdt_sigma_rdm as source,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "manybody_lib"
    / "configuration_interaction"
    / "spin_orbital"
    / "cisdt"
    / "_contractions.py"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="generate spin-orbital CISDT contraction tables",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _space_code(index) -> str:
    assumptions = index.assumptions0
    if assumptions.get("below_fermi"):
        return "o"
    if assumptions.get("above_fermi"):
        return "v"
    raise ValueError(f"index {index!r} has no occupied/virtual space")


def _tensor_source(factor) -> tuple[str, tuple]:
    if isinstance(factor, AppliedUndef):
        if factor.func == source.fock:
            return "f", tuple(factor.args)
        if factor.func == source.c1:
            return "c1", tuple(factor.args)
    if isinstance(factor, AntiSymmetricTensor):
        name = str(factor.args[0])
        indices = tuple(factor.args[1]) + tuple(factor.args[2])
        if name == "u":
            return "u", indices
        if name == "c" and len(indices) == 4:
            return "c2", indices
        if name == "c" and len(indices) == 6:
            return "c3", indices
    raise TypeError(f"unsupported tensor factor: {factor!r}")


def _term_spec(term, external_indices):
    coefficient, remainder = term.as_coeff_Mul()
    factors = Mul.make_args(remainder)
    indexed_factors = []
    scalars = []
    indices = []

    for factor in factors:
        is_conjugated = factor.func == conjugate
        inner = factor.args[0] if is_conjugated else factor
        if inner == source.Eref:
            if is_conjugated:
                raise ValueError("Eref must not be explicitly conjugated")
            scalars.append("e_ref")
        elif inner == source.c0:
            scalars.append("c0_conj" if is_conjugated else "c0")
        elif isinstance(inner, KroneckerDelta):
            if is_conjugated:
                raise ValueError("Kronecker deltas must not be conjugated")
            delta_indices = tuple(inner.args)
            indexed_factors.append(
                (f"eye_{_space_code(delta_indices[0])}", "", delta_indices)
            )
            indices.extend(delta_indices)
        else:
            tensor_source, tensor_indices = _tensor_source(inner)
            if is_conjugated:
                if tensor_source in ("f", "u"):
                    raise ValueError("Hamiltonian tensors must not be conjugated")
                tensor_source += "_conj"
            spaces = ""
            if tensor_source in ("f", "u"):
                spaces = "".join(_space_code(index) for index in tensor_indices)
            indexed_factors.append((tensor_source, spaces, tensor_indices))
            indices.extend(tensor_indices)

    for index in external_indices:
        if index not in indices:
            indices.append(index)
    unique = []
    for index in indices:
        if index not in unique:
            unique.append(index)
    if len(unique) > len(string.ascii_letters):
        raise ValueError("too many indices for an einsum contraction")
    labels = dict(
        zip(unique, string.ascii_letters[: len(unique)], strict=True)
    )

    operands = tuple(
        (tensor_source, spaces)
        for tensor_source, spaces, _ in indexed_factors
    )
    operand_subscripts = tuple(
        "".join(labels[index] for index in tensor_indices)
        for _, _, tensor_indices in indexed_factors
    )
    output = "".join(labels[index] for index in external_indices)
    equation = ",".join(operand_subscripts) + "->" + output
    return (
        (int(coefficient.p), int(coefficient.q)),
        tuple(scalars),
        operands,
        equation,
    )


def _expression_specs(expression, external_indices):
    return tuple(
        _term_spec(term, external_indices)
        for term in Add.make_args(expression)
    )


def _generate_data():
    sigma_externals = {
        0: (),
        1: (source.a, source.i),
        2: (source.a, source.b, source.i, source.j),
        3: (
            source.a,
            source.b,
            source.c,
            source.i,
            source.j,
            source.k,
        ),
    }
    sigma_expressions = {
        0: source.sigma_0,
        1: source.sigma_ai,
        2: source.sigma_abij_expanded,
        3: source.sigma_abcijk_expanded,
    }
    sigma = {
        rank: {
            "spaces": "v" * rank + "o" * rank,
            "terms": _expression_specs(
                sigma_expressions[rank],
                sigma_externals[rank],
            ),
        }
        for rank in range(4)
    }
    norm = _expression_specs(source.cisdt_norm, ())
    one = {
        "".join(_space_code(index) for index in indices): {
            "spaces": "".join(_space_code(index) for index in indices),
            "terms": _expression_specs(source.ONE_RDM_BLOCKS[key], indices),
        }
        for key, _label, indices in source.ONE_RDM_SPECS
    }
    two = {
        "".join(_space_code(index) for index in indices): {
            "spaces": "".join(_space_code(index) for index in indices),
            "terms": _expression_specs(
                source.EXPANDED_TWO_RDM_BLOCKS[key],
                indices,
            ),
        }
        for key, _label, indices, _operators in source.TWO_RDM_SPECS
    }
    counts = {
        "sigma": tuple(
            len(Add.make_args(sigma_expressions[rank])) for rank in range(4)
        ),
        "norm": len(Add.make_args(source.cisdt_norm)),
        "one_rdm": tuple(
            len(Add.make_args(source.ONE_RDM_BLOCKS[key]))
            for key, _label, _indices in source.ONE_RDM_SPECS
        ),
        "two_rdm": tuple(
            len(Add.make_args(source.EXPANDED_TWO_RDM_BLOCKS[key]))
            for key, _label, _indices, _operators in source.TWO_RDM_SPECS
        ),
    }
    return sigma, norm, one, two, counts


def main() -> None:
    args = _parse_args()
    sigma, norm, one, two, counts = _generate_data()
    data = {
        "SIGMA_SPECS": sigma,
        "NORM_TERMS": norm,
        "ONE_RDM_SPECS": one,
        "TWO_RDM_SPECS": two,
        "TERM_COUNTS": counts,
    }
    lines = [
        '"""Generated spin-orbital CISDT numerical contraction tables.\n\n',
        "Do not edit by hand; regenerate with\n",
        "``symbolic_derivations/SymPy/",
        "generate_spin_orbital_cisdt_contractions.py``.\n",
        '"""\n\n',
    ]
    for name, value in data.items():
        lines.append(f"{name} = ")
        lines.append(pprint.pformat(value, width=88, sort_dicts=False))
        lines.append("\n\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(lines).rstrip() + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
