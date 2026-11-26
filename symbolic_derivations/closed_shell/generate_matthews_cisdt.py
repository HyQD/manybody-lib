"""Generate Matthews-manifold closed-shell CISDT equations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manybody_lib.symbolic.closed_shell import (
    derive_matthews_rdms,
    derive_matthews_sigma,
    matthews_metric,
    matthews_rdms_to_latex,
    matthews_sigma_to_latex,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="derive closed-shell Matthews CISDT equations",
    )
    parser.add_argument(
        "--section",
        choices=("metric", "sigma", "rdm", "all"),
        default="all",
        help="equation family to derive (default: all)",
    )
    parser.add_argument(
        "--expanded",
        action="store_true",
        help="expand the metric permutation action in sigma components",
    )
    parser.add_argument(
        "--profile-json",
        type=Path,
        help="write per-rank/per-block runtime and peak-allocation data",
    )
    return parser.parse_args()


def _metric_latex() -> str:
    lines = [
        r"\begin{align}",
        r"\operatorname{rank} M_1 &= 1, & \operatorname{nullity} M_1 &= 0, \\",
        r"\operatorname{rank} M_2 &= 2, & \operatorname{nullity} M_2 &= 0, \\",
        r"\operatorname{rank} M_3 &= 5, & \operatorname{nullity} M_3 &= 1, \\",
        r"M_2^+ &= \frac13 + \frac16 P(ij), \\",
        (
            r"M_3^+ = \frac{17}{144}"
            r"-\frac{P(ij)+P(ik)+P(jk)}{144}"
            r"-\frac{7[P(ijk)+P(ikj)]}{144}."
        ),
        r"\end{align}",
    ]
    for rank in (1, 2, 3):
        metric = matthews_metric(rank)
        if metric.matrix_rank + metric.nullity != len(metric.metric):
            raise AssertionError("inconsistent Matthews metric dimensions")
    return "\n".join(lines)


def _measurement_record(measurement) -> dict[str, int | float | str]:
    return {
        "label": measurement.label,
        "seconds": measurement.seconds,
        "peak_bytes": measurement.peak_bytes,
        "term_count": measurement.term_count,
    }


def main() -> None:
    args = _parse_args()
    profile: dict[str, list[dict[str, int | float | str]]] = {}
    sections = (
        ("metric", "sigma", "rdm")
        if args.section == "all"
        else (args.section,)
    )
    for position, section in enumerate(sections):
        if position:
            print()
        if section == "metric":
            print("% Matthews CISDT exact Gram matrices and pseudoinverses")
            print(_metric_latex())
        elif section == "sigma":
            sigma = derive_matthews_sigma()
            print("% Matthews CISDT right sigma vector")
            print(
                matthews_sigma_to_latex(
                    sigma,
                    factor_permutations=not args.expanded,
                )
            )
            profile["sigma"] = [
                _measurement_record(item)
                for item in sigma.measurements
            ]
        else:
            rdms = derive_matthews_rdms()
            print("% Matthews CISDT norm, 1-RDM, and 2-RDM")
            print(matthews_rdms_to_latex(rdms))
            profile["rdm"] = [
                _measurement_record(item)
                for item in rdms.measurements
            ]

    if args.profile_json is not None:
        args.profile_json.write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
