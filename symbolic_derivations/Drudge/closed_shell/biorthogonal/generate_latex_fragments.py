"""Regenerate the LaTeX equation fragments from the Drudge scripts."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"
METHODS = ("cis", "cid", "cisd")

SECTION_MARKERS = {
    "% Norm": r"\subsubsection{Norm}",
    "% Right sigma vector": r"\subsubsection{Right sigma vector}",
    "% Left sigma vector": r"\subsubsection{Left sigma vector}",
    "% Nonzero one-body reduced-density-matrix blocks": (
        r"\subsubsection{One-body reduced density matrix}"
    ),
    "% Nonzero unique two-body reduced-density-matrix blocks": (
        r"\subsubsection{Two-body reduced density matrix}"
    ),
}


def document_fragment(output: str) -> str:
    """Convert script output into a sectioned, unnumbered LaTeX fragment."""

    output = output.replace(r"\begin{align}", r"\begin{align*}")
    output = output.replace(r"\end{align}", r"\end{align*}")
    for marker, heading in SECTION_MARKERS.items():
        output = output.replace(marker, heading)
    return output.rstrip() + "\n"


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    for method in METHODS:
        script = HERE / f"{method}.py"
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=HERE,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{script.name} failed with code {completed.returncode}:\n"
                f"{completed.stderr}"
            )

        destination = GENERATED / f"{method}_equations.tex"
        destination.write_text(
            document_fragment(completed.stdout),
            encoding="utf-8",
        )
        print(f"wrote {destination.relative_to(HERE)}")


if __name__ == "__main__":
    main()
