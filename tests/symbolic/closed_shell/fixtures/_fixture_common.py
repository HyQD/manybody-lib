"""Shared deterministic serialization helpers for symbolic fixtures."""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
SCHEMA_VERSION = 1


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def provenance(generator: Path, sources: list[Path], packages: list[str]) -> dict:
    return {
        "generator": relative_path(generator),
        "sources": [
            {
                "path": relative_path(source),
                "sha256": file_sha256(source),
            }
            for source in sources
        ],
        "packages": {
            package: package_version(package)
            for package in sorted(packages)
        },
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def drudge_tensor_record(dr, tensor) -> dict:
    """Serialize a scalar Drudge tensor without relying on RDD term order."""

    terms = []
    for term in tensor.local_terms:
        if term.vecs:
            raise ValueError("fixture tensor is not fully contracted")
        terms.append(
            {
                "amplitude": str(term.amp),
                "sums": [
                    {
                        "index": str(index),
                        "range": str(orbital_range),
                    }
                    for index, orbital_range in term.sums
                ],
            }
        )
    terms.sort(
        key=lambda term: (
            tuple(
                (summation["range"], summation["index"])
                for summation in term["sums"]
            ),
            term["amplitude"],
        )
    )
    return {
        "term_count": len(terms),
        "terms": terms,
        "latex": dr.format_latex(tensor, no_sum=True, sep_lines=True).strip(),
    }


def stable_sympy_expression(expr):
    """Replace identity-bearing Dummy objects by stable typed Symbols."""

    from sympy import Dummy, Symbol

    dummies = list(expr.atoms(Dummy))

    def domain(dummy):
        if dummy.assumptions0.get("above_fermi"):
            return "virtual"
        if dummy.assumptions0.get("below_fermi"):
            return "occupied"
        return "general"

    grouped = {}
    for dummy in dummies:
        grouped.setdefault((domain(dummy), dummy.name), []).append(dummy)

    replacements = {}
    for (orbital_domain, name), members in sorted(grouped.items()):
        members.sort(key=lambda item: item.dummy_index)
        for position, dummy in enumerate(members):
            suffix = "" if len(members) == 1 else f"_{position}"
            assumptions = {}
            if orbital_domain == "virtual":
                assumptions["above_fermi"] = True
            elif orbital_domain == "occupied":
                assumptions["below_fermi"] = True
            replacements[dummy] = Symbol(
                f"dummy_{orbital_domain}_{name}{suffix}",
                **assumptions,
            )
    return expr.xreplace(replacements)


def sympy_expression_record(expr, latex_printer) -> dict:
    from sympy import Add, expand, srepr, sstr

    expanded = expand(expr)
    stable = stable_sympy_expression(expanded)
    return {
        "term_count": len(Add.make_args(expanded)),
        "expression": sstr(stable, order="lex"),
        "srepr": srepr(stable),
        "latex": latex_printer(expanded),
    }
