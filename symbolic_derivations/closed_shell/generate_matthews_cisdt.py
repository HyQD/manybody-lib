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
    matthews_one_rdms_to_latex,
    matthews_rdms_to_latex,
    matthews_sigma_to_latex,
    matthews_triples_permutation_definitions,
    matthews_two_rdms_to_latex,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="derive closed-shell Matthews CISDT equations",
    )
    parser.add_argument(
        "--section",
        choices=("metric", "sigma", "one-rdm", "two-rdm", "rdm", "all"),
        default="all",
        help="equation family to derive (default: all)",
    )
    parser.add_argument(
        "--expanded",
        action="store_true",
        help="print canonical sigma components without permutation factoring",
    )
    parser.add_argument(
        "--profile-json",
        type=Path,
        help="write per-rank/per-block runtime and peak-allocation data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write one self-contained .tex fragment per selected section",
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
    if args.section == "all":
        sections = ("metric", "sigma", "one-rdm", "two-rdm")
    elif args.section == "rdm":
        sections = ("one-rdm", "two-rdm")
    else:
        sections = (args.section,)
    rendered = {}
    rdms = None
    for section in sections:
        if section == "metric":
            rendered[section] = "\n".join(
                (
                    "% Matthews CISDT exact Gram matrices and pseudoinverses",
                    _metric_latex(),
                )
            )
        elif section == "sigma":
            sigma = derive_matthews_sigma()
            body = matthews_sigma_to_latex(
                sigma,
                factor_permutations=not args.expanded,
            )
            pieces = ["% Matthews CISDT right sigma vector"]
            if not args.expanded:
                pieces.extend(
                    (
                        "% Simultaneous triples-column permutation operators",
                        matthews_triples_permutation_definitions(),
                    )
                )
            pieces.append(body)
            rendered[section] = "\n".join(pieces)
            profile["sigma"] = [
                _measurement_record(item)
                for item in sigma.measurements
            ]
        elif section == "one-rdm":
            if rdms is None:
                rdms = derive_matthews_rdms()
            rendered[section] = "\n".join(
                (
                    "% Matthews CISDT norm and one-body RDM",
                    matthews_one_rdms_to_latex(rdms),
                )
            )
            profile["rdm"] = [
                _measurement_record(item)
                for item in rdms.measurements
            ]
        elif section == "two-rdm":
            if rdms is None:
                rdms = derive_matthews_rdms()
            rendered[section] = "\n".join(
                (
                    "% Matthews CISDT two-body RDM",
                    matthews_two_rdms_to_latex(rdms),
                )
            )
            profile["rdm"] = [
                _measurement_record(item)
                for item in rdms.measurements
            ]
        else:
            if rdms is None:
                rdms = derive_matthews_rdms()
            rendered[section] = "\n".join(
                (
                    "% Matthews CISDT norm, 1-RDM, and 2-RDM",
                    matthews_rdms_to_latex(rdms),
                )
            )
            profile["rdm"] = [
                _measurement_record(item)
                for item in rdms.measurements
            ]

    if args.output_dir is None:
        for position, section in enumerate(sections):
            if position:
                print()
            print(rendered[section])
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        filenames = {
            "metric": "matthews_cisdt_metric.tex",
            "sigma": "matthews_cisdt_sigma.tex",
            "one-rdm": "matthews_cisdt_one_rdm.tex",
            "two-rdm": "matthews_cisdt_two_rdm.tex",
            "rdm": "matthews_cisdt_rdms.tex",
        }
        for section in sections:
            destination = args.output_dir / filenames[section]
            destination.write_text(rendered[section] + "\n")
            print(destination)

    if args.profile_json is not None:
        args.profile_json.write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
