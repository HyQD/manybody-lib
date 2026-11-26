"""Generate expanded spin-orbital SymPy CIS/CID/CISD fixtures."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from _fixture_common import (
    HERE,
    REPO_ROOT,
    SCHEMA_VERSION,
    provenance,
    sympy_expression_record,
    write_json,
)


SOURCE_DIR = REPO_ROOT / "symbolic_derivations" / "SymPy"
METHODS = ("cis", "cid", "cisd")


def _load_source(path: Path):
    module_name = f"_manybody_fixture_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sigma_expressions(method: str, module) -> dict:
    if method == "cis":
        return {
            r"\sigma_0": module.sigma0,
            r"\sigma_i^a": module.sigma_ai,
        }
    if method == "cid":
        return {
            r"\sigma_0": module.sigma0,
            r"\sigma_{ij}^{ab}": module.sigma_abij_expanded,
        }
    return {
        r"\sigma_0": module.EXPANDED_SIGMA["0"],
        r"\sigma_i^a": module.EXPANDED_SIGMA["ai"],
        r"\sigma_{ij}^{ab}": module.EXPANDED_SIGMA["abij"],
    }


def _records(expressions, printer) -> dict:
    return {
        name: sympy_expression_record(expression, printer)
        for name, expression in expressions
    }


def _expanded_two_rdm_blocks(module):
    """Return labeled nonzero blocks before permutation presentation."""

    return tuple(
        (label, module.EXPANDED_BLOCKS[key])
        for key, label, _indices, _permutations in module.BLOCK_SPECS
        if module.EXPANDED_BLOCKS[key] != 0
    )


def generate(method: str) -> Path:
    paths = {
        kind: SOURCE_DIR / f"spin_orbital_{method}_{kind}.py"
        for kind in ("sigma", "1rdm", "2rdm")
    }
    modules = {kind: _load_source(path) for kind, path in paths.items()}

    one_rdm = modules["1rdm"]
    two_rdm = modules["2rdm"]
    sigma = modules["sigma"]
    norm_name = f"{method}_norm"
    norm = getattr(one_rdm, norm_name)

    destination = HERE / f"spin_orbital_{method}.json"
    write_json(
        destination,
        {
            "schema_version": SCHEMA_VERSION,
            "family": "spin-orbital-sympy",
            "method": method.upper(),
            "representation": "canonical-expanded-sympy",
            "normalization": "unnormalized-hermitian-bra",
            "conventions": "CONVENTIONS.md",
            "provenance": provenance(
                Path(__file__),
                list(paths.values()),
                ["sympy"],
            ),
            "expressions": {
                "norm": {
                    r"\langle\Psi|\Psi\rangle": sympy_expression_record(
                        norm,
                        one_rdm.latex_expression,
                    )
                },
                "right_sigma": _records(
                    _sigma_expressions(method, sigma).items(),
                    sigma.latex_expression,
                ),
                "one_rdm": _records(
                    one_rdm.NONZERO_BLOCKS,
                    one_rdm.latex_expression,
                ),
                "two_rdm": _records(
                    _expanded_two_rdm_blocks(two_rdm),
                    two_rdm.latex_expression,
                ),
            },
        },
    )
    return destination


def main() -> None:
    for method in METHODS:
        destination = generate(method)
        print(destination.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
