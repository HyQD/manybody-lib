"""Generate closed-shell HJO CI norms, 1-RDMs, and 2-RDMs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manybody_lib.symbolic.closed_shell import derive_ci_rdms, rdms_to_latex


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="derive closed-shell HJO CI norms and RDMs",
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
        help="print canonical expanded forms without permutation factoring",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    methods = ("cis", "cid", "cisd") if args.method == "all" else (args.method,)
    for position, method in enumerate(methods):
        if position:
            print()
        print(f"% Closed-shell HJO {method.upper()} norm and RDMs")
        print(
            rdms_to_latex(
                derive_ci_rdms(method),
                factor_permutations=not args.expanded,
            )
        )


if __name__ == "__main__":
    main()
