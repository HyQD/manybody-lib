"""Generate numerical contraction tables for closed-shell Matthews CISDT."""

from __future__ import annotations

import argparse
from pathlib import Path
import pprint
import string
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manybody_lib.symbolic.closed_shell import (  # noqa: E402
    Delta,
    OrbitalDimension,
    Tensor,
    derive_matthews_rdms,
    derive_matthews_sigma,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "manybody_lib"
    / "configuration_interaction"
    / "closed_shell"
    / "cisdt"
    / "_contractions.py"
)
_SCALARS = {
    "Eref": "e_ref",
    "C0": "c0",
    "C0tilde": "left_c0",
}
_ARRAYS = {
    "f": "f",
    "u": "u",
    "C1": "c1",
    "C2": "c2",
    "C3": "c3",
    "Ctilde1": "left_c1",
    "Ctilde2": "left_c2",
    "Ctilde3": "left_c3",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="generate closed-shell Matthews CISDT contraction tables",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _space_code(index) -> str:
    return "o" if index.space.value == "occupied" else "v"


def _term_spec(term, external_indices):
    indices = []
    for factor in term.factors:
        indices.extend(factor.indices)
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

    scalars = []
    operands = []
    operand_subscripts = []
    for factor in term.factors:
        if isinstance(factor, Tensor):
            if factor.conjugated:
                raise ValueError(
                    "generated independent-left equations are unconjugated"
                )
            name = factor.symbol.name
            if not factor.indices:
                scalars.append(_SCALARS[name])
                continue
            source = _ARRAYS[name]
            spaces = ""
            if source in ("f", "u"):
                spaces = "".join(_space_code(index) for index in factor.indices)
            operands.append((source, spaces))
            operand_subscripts.append(
                "".join(labels[index] for index in factor.indices)
            )
        elif isinstance(factor, Delta):
            space = _space_code(factor.left)
            operands.append((f"eye_{space}", ""))
            operand_subscripts.append(
                "".join(labels[index] for index in factor.indices)
            )
        elif isinstance(factor, OrbitalDimension):
            scalars.append(
                "nocc" if factor.space.value == "occupied" else "nvir"
            )
        else:
            raise TypeError(type(factor).__name__)

    output = "".join(labels[index] for index in external_indices)
    equation = ",".join(operand_subscripts) + "->" + output
    coefficient = (
        term.coefficient.real.numerator,
        term.coefficient.real.denominator,
        term.coefficient.imag.numerator,
        term.coefficient.imag.denominator,
    )
    return coefficient, tuple(scalars), tuple(operands), equation


def _expression_specs(expression, external_indices):
    return tuple(_term_spec(term, external_indices) for term in expression)


def _inverse_permutation(permutation):
    inverse = [0] * len(permutation)
    for position, image in enumerate(permutation):
        inverse[image] = position
    return tuple(inverse)


def _sigma_specs():
    sigma = derive_matthews_sigma()
    result = {}
    # Free indices are stable but need not occur in the first scalar factor.
    from manybody_lib.symbolic.closed_shell import occupied, virtual

    external_by_rank = {
        0: (),
        1: (virtual("a"), occupied("i")),
        2: tuple(map(virtual, "ab")) + tuple(map(occupied, "ij")),
        3: tuple(map(virtual, "abc")) + tuple(map(occupied, "ijk")),
    }
    for rank in range(4):
        external = external_by_rank[rank]
        spaces = "v" * rank + "o" * rank
        presentation = sigma.column_presentation(rank)
        if presentation is None:
            explicit = sigma[rank]
            groups = ()
        elif rank == 2:
            explicit = presentation.explicit
            groups = (
                (
                    ((0, 1, 2, 3), (1, 0, 3, 2)),
                    _expression_specs(presentation.representative, external),
                ),
            )
        else:
            explicit = presentation.explicit
            groups = tuple(
                (
                    tuple(
                        _inverse_permutation(action)
                        + tuple(
                            3 + item
                            for item in _inverse_permutation(action)
                        )
                        for action in group.actions
                    ),
                    _expression_specs(group.representatives, external),
                )
                for group in presentation.groups
            )
        result[rank] = {
            "spaces": spaces,
            "explicit": _expression_specs(explicit, external),
            "groups": groups,
        }
    return result, sigma


def _rdm_specs():
    rdms = derive_matthews_rdms()
    norm = _expression_specs(rdms.norm, ())
    one = {
        "".join(_space_code(index) for index in block.indices): {
            "spaces": "".join(_space_code(index) for index in block.indices),
            "terms": _expression_specs(block.expression, block.indices),
        }
        for block in rdms.one_body
    }
    two = {
        "".join(_space_code(index) for index in block.indices): {
            "spaces": "".join(_space_code(index) for index in block.indices),
            "terms": _expression_specs(block.expression, block.indices),
        }
        for block in rdms.two_body
    }
    return norm, one, two, rdms


def main() -> None:
    args = _parse_args()
    sigma, sigma_result = _sigma_specs()
    norm, one, two, rdm_result = _rdm_specs()
    data = {
        "SIGMA_SPECS": sigma,
        "NORM_TERMS": norm,
        "ONE_RDM_SPECS": one,
        "TWO_RDM_SPECS": two,
    }
    lines = [
        '"""Generated Matthews CISDT numerical contraction tables.\n\n',
        "Do not edit by hand; regenerate with\n",
        "``symbolic_derivations/closed_shell/",
        "generate_matthews_cisdt_contractions.py``.\n",
        '"""\n\n',
    ]
    for name, value in data.items():
        lines.append(f"{name} = ")
        lines.append(pprint.pformat(value, width=88, sort_dicts=False))
        lines.append("\n\n")
    term_counts = {
        "sigma": tuple(
            expression.term_count
            for _, expression in sigma_result.resolved.components
        ),
        "norm": rdm_result.norm.term_count,
        "one_rdm": tuple(
            block.expression.term_count for block in rdm_result.one_body
        ),
        "two_rdm": tuple(
            block.expression.term_count for block in rdm_result.two_body
        ),
    }
    lines.append("TERM_COUNTS = ")
    lines.append(pprint.pformat(term_counts, width=88, sort_dicts=False))
    lines.append("\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
