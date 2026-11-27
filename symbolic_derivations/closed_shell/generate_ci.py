"""Generate closed-shell HJO CIS, CID, and CISD right sigma vectors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manybody_lib.symbolic.closed_shell import (
    derive_right_sigma,
    right_sigma_to_latex,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="derive closed-shell HJO right sigma vectors",
    )
    parser.add_argument(
        "--method",
        choices=("cis", "cid", "cisd", "all"),
        default="all",
        help="truncation to derive (default: all)",
    )
    parser.add_argument(
        "--expanded",
        action="store_true",
        help="print the canonical expanded form without permutation factoring",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    methods = ("cis", "cid", "cisd") if args.method == "all" else (args.method,)
    for position, method in enumerate(methods):
        if position:
            print()
        print(f"% Closed-shell HJO {method.upper()} right sigma vector")
        print(
            right_sigma_to_latex(
                derive_right_sigma(method),
                factor_permutations=not args.expanded,
            )
        )


if __name__ == "__main__":
    main()
