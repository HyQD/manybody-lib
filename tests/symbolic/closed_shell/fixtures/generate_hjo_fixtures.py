"""Generate expanded HJO CIS/CID/CISD Drudge golden fixtures.

Run in the dedicated ``drudge`` environment.  Presentation-only exchange and
permutation factoring is disabled before the source derivation is executed.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys

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
    / "biorthogonal"
)
METHODS = ("cis", "cid", "cisd")
SECTIONS = ("norm", "right_sigma", "left_sigma", "one_rdm", "two_rdm")


def _identity_postprocessor(_dr, tensor, *args, **kwargs):
    del args, kwargs
    return tensor


def _load_source(method: str):
    source = SOURCE_DIR / f"{method}.py"
    module_name = f"_manybody_fixture_hjo_{method}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, source


class Recorder:
    def __init__(self) -> None:
        self.sections: dict[str, dict] = {}
        self._position = 0

    def print_align(self, dr, equations) -> None:
        if self._position >= len(SECTIONS):
            raise RuntimeError("source emitted more equation groups than expected")
        section = SECTIONS[self._position]
        self._position += 1
        records = {}
        for lhs, tensor in equations:
            if tensor.n_terms:
                records[lhs] = drudge_tensor_record(dr, tensor)
        self.sections[section] = records

    def validate(self) -> None:
        if self._position != len(SECTIONS):
            raise RuntimeError(
                f"captured {self._position} groups; expected {len(SECTIONS)}"
            )


def generate(method: str) -> Path:
    module, source = _load_source(method)
    recorder = Recorder()

    module.factor_exchange_integrals = _identity_postprocessor
    module.factor_column_permutation = _identity_postprocessor
    module.print_align = recorder.print_align

    # The source still emits section comments; suppress them because the
    # recorder captures the underlying tensors directly.
    with redirect_stdout(io.StringIO()):
        module.derive()
    recorder.validate()

    destination = HERE / f"hjo_{method}.json"
    write_json(
        destination,
        {
            "schema_version": SCHEMA_VERSION,
            "family": "closed-shell-hjo",
            "method": method.upper(),
            "representation": "canonical-expanded-drudge-terms",
            "normalization": "unnormalized-independent-left-right",
            "conventions": "CONVENTIONS.md",
            "provenance": provenance(
                Path(__file__),
                [source, SOURCE_DIR / "permutation.py"],
                ["drudge", "sympy", "pyspark"],
            ),
            "expressions": recorder.sections,
        },
    )
    return destination


def main() -> None:
    sys.path.insert(0, str(SOURCE_DIR))
    for method in METHODS:
        destination = generate(method)
        print(destination.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
