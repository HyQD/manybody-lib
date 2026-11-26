"""Generate exact Matthews metric and completed CISDT RDM fixtures."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from sympy import Matrix, Rational, eye, zeros

from _fixture_common import (
    HERE,
    REPO_ROOT,
    SCHEMA_VERSION,
    drudge_tensor_record,
    provenance,
    write_json,
)


SOURCE_DIR = (
    REPO_ROOT / "symbolic_derivations" / "Drudge" / "closed_shell"
    / "matthews"
)
COMMON = SOURCE_DIR / "common.py"
VALIDATE_METRIC = SOURCE_DIR / "validate_metric.py"


def _load_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _rational_text(value) -> str:
    value = Rational(value)
    return str(value)


def _matrix_record(matrix: Matrix) -> list[list[str]]:
    return [
        [_rational_text(matrix[row, column]) for column in range(matrix.cols)]
        for row in range(matrix.rows)
    ]


def _metric_rank(validate, rank, metric_weights, pinv_weights) -> dict:
    metric = validate.regular_class_matrix(rank, metric_weights)
    pinv = validate.regular_class_matrix(rank, pinv_weights)
    dimension = metric.rows
    null = Matrix.ones(dimension, dimension) / dimension if rank == 3 else zeros(dimension)
    physical = eye(dimension) - null
    eigenvalues = []
    for value, multiplicity, _vectors in metric.eigenvects():
        eigenvalues.extend([_rational_text(value)] * multiplicity)
    return {
        "rank": rank,
        "metric_weights_by_cycle_count": {
            str(key): _rational_text(value)
            for key, value in sorted(metric_weights.items())
        },
        "pseudoinverse_weights_by_cycle_count": {
            str(key): _rational_text(value)
            for key, value in sorted(pinv_weights.items())
        },
        "metric": _matrix_record(metric),
        "pseudoinverse": _matrix_record(pinv),
        "null_projector": _matrix_record(null),
        "physical_projector": _matrix_record(physical),
        "eigenvalues": sorted(eigenvalues, key=Rational),
    }


def generate_metric() -> Path:
    validate = _load_source(VALIDATE_METRIC, "_manybody_fixture_matthews_metric")
    ranks = {
        "2": _metric_rank(
            validate,
            2,
            {2: 4, 1: -2},
            {2: Rational(1, 3), 1: Rational(1, 6)},
        ),
        "3": _metric_rank(
            validate,
            3,
            {3: 8, 2: -4, 1: 2},
            {
                3: Rational(17, 144),
                2: Rational(-1, 144),
                1: Rational(-7, 144),
            },
        ),
    }
    destination = HERE / "matthews_metric.json"
    write_json(
        destination,
        {
            "schema_version": SCHEMA_VERSION,
            "family": "closed-shell-matthews-metric",
            "representation": "exact-regular-representation-matrices",
            "conventions": "CONVENTIONS.md",
            "provenance": provenance(
                Path(__file__),
                [VALIDATE_METRIC, COMMON],
                ["sympy"],
            ),
            "ranks": ranks,
        },
    )
    return destination


def _generate_rdm(kind: str) -> Path:
    sys.path.insert(0, str(SOURCE_DIR))
    common = _load_source(COMMON, f"_manybody_fixture_matthews_{kind}")
    options = common.DerivationOptions.from_environment()
    context = common.make_spark_context(
        options,
        f"manybody-matthews-{kind}-fixture",
    )
    try:
        model = common.build_model(context, options)
        norm = common.derive_norm(model)
        if kind == "one_rdm":
            blocks = common.derive_one_rdm(model)
            source = SOURCE_DIR / "cisdt_one_rdm.py"
        elif kind == "two_rdm":
            blocks = common.derive_two_rdm(model)
            source = SOURCE_DIR / "cisdt_two_rdm.py"
        else:
            raise ValueError(f"unknown RDM kind {kind!r}")

        expressions = {
            "norm": {
                r"\langle\widetilde\Psi|\Psi\rangle": drudge_tensor_record(
                    model.dr,
                    norm,
                )
            },
            kind: {
                label: drudge_tensor_record(model.dr, tensor)
                for label, tensor in blocks
            },
        }
    finally:
        context.stop()

    destination = HERE / f"matthews_cisdt_{kind}.json"
    write_json(
        destination,
        {
            "schema_version": SCHEMA_VERSION,
            "family": "closed-shell-matthews",
            "method": "CISDT",
            "representation": "canonical-expanded-drudge-terms",
            "normalization": "unnormalized-independent-left-right",
            "conventions": "CONVENTIONS.md",
            "provenance": provenance(
                Path(__file__),
                [source, COMMON],
                ["drudge", "sympy", "pyspark"],
            ),
            "expressions": expressions,
        },
    )
    return destination


def generate_status() -> Path:
    destination = HERE / "matthews_status.json"
    write_json(
        destination,
        {
            "schema_version": SCHEMA_VERSION,
            "family": "closed-shell-matthews-validation-status",
            "method": "CISDT",
            "conventions": "CONVENTIONS.md",
            "provenance": provenance(
                Path(__file__),
                [COMMON, SOURCE_DIR / "README.md"],
                ["drudge", "sympy", "pyspark"],
            ),
            "completed": {
                "metric": "exact S2/S3 identities validated",
                "one_rdm_blocks": [
                    r"\gamma^j_i",
                    r"\gamma^a_i",
                    r"\gamma^i_a",
                    r"\gamma^b_a",
                ],
                "two_rdm_blocks": [
                    r"\Gamma^{kl}_{ij}",
                    r"\Gamma^{ak}_{ij}",
                    r"\Gamma^{jk}_{ai}",
                    r"\Gamma^{ab}_{ij}",
                    r"\Gamma^{ij}_{ab}",
                    r"\Gamma^{bj}_{ai}",
                    r"\Gamma^{bj}_{ia}",
                    r"\Gamma^{bc}_{ai}",
                    r"\Gamma^{ci}_{ab}",
                    r"\Gamma^{cd}_{ab}",
                ],
                "sigma_ranks": [0, 1, 2],
                "rank3_off_diagonal_sectors": [
                    "singles-to-triples",
                    "doubles-to-triples",
                ],
                "rank3_diagonal_hamiltonian_pieces": "1 through 9 of 14",
            },
            "incomplete": {
                "rank3_diagonal_hamiltonian_pieces": "10 through 14 of 14",
                "reason": "run stopped at the user-requested sector boundary",
            },
        },
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", action="store_true")
    parser.add_argument("--one-rdm", action="store_true")
    parser.add_argument("--two-rdm", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = args.all or not any(
        (args.metric, args.one_rdm, args.two_rdm, args.status)
    )
    actions = []
    if selected or args.metric:
        actions.append(generate_metric)
    if selected or args.one_rdm:
        actions.append(lambda: _generate_rdm("one_rdm"))
    if selected or args.two_rdm:
        actions.append(lambda: _generate_rdm("two_rdm"))
    if selected or args.status:
        actions.append(generate_status)
    for action in actions:
        destination = action()
        print(destination.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
