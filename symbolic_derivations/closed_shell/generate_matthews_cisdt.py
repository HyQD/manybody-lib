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
    Expression,
    MATTHEWS_LEFT_SINGLES,
    MATTHEWS_SINGLES,
    MatthewsRDMs,
    MatthewsSigmaVector,
    RDMBlock,
    RightSigmaVector,
    Tensor,
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
        description="derive closed-shell Matthews CIDT or CISDT equations",
    )
    parser.add_argument(
        "--method",
        choices=("cisdt", "cidt"),
        default="cisdt",
        help="Matthews CI truncation to render (default: cisdt)",
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


def _without_singles(expression: Expression) -> Expression:
    singles = {MATTHEWS_SINGLES, MATTHEWS_LEFT_SINGLES}
    return Expression.from_terms(
        term
        for term in expression
        if not any(
            isinstance(factor, Tensor) and factor.symbol in singles
            for factor in term.factors
        )
    )


def _restrict_sigma_to_cidt(sigma: MatthewsSigmaVector) -> MatthewsSigmaVector:
    return MatthewsSigmaVector(
        RightSigmaVector(
            "cidt",
            tuple(
                (rank, _without_singles(expression))
                for rank, expression in sigma.resolved.components
            ),
        ),
        tuple(
            (rank, _without_singles(expression))
            for rank, expression in sigma.covariant
        ),
        (),
    )


def _restrict_rdms_to_cidt(rdms: MatthewsRDMs) -> MatthewsRDMs:
    def restrict(block: RDMBlock) -> RDMBlock:
        return RDMBlock(
            block.rank,
            block.label,
            block.indices,
            _without_singles(block.expression),
        )

    return MatthewsRDMs(
        "cidt",
        _without_singles(rdms.norm),
        tuple(restrict(block) for block in rdms.one_body),
        tuple(restrict(block) for block in rdms.two_body),
        (),
    )


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
    method_label = args.method.upper()
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
                    f"% Matthews {method_label} exact Gram matrices and pseudoinverses",
                    _metric_latex(),
                )
            )
        elif section == "sigma":
            sigma = derive_matthews_sigma()
            if args.method == "cidt":
                sigma = _restrict_sigma_to_cidt(sigma)
            body = matthews_sigma_to_latex(
                sigma,
                factor_permutations=not args.expanded,
                ranks=(0, 2, 3) if args.method == "cidt" else (0, 1, 2, 3),
            )
            pieces = [f"% Matthews {method_label} right sigma vector"]
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
                if args.method == "cidt":
                    rdms = _restrict_rdms_to_cidt(rdms)
            rendered[section] = "\n".join(
                (
                    f"% Matthews {method_label} norm and one-body RDM",
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
                if args.method == "cidt":
                    rdms = _restrict_rdms_to_cidt(rdms)
            rendered[section] = "\n".join(
                (
                    f"% Matthews {method_label} two-body RDM",
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
                if args.method == "cidt":
                    rdms = _restrict_rdms_to_cidt(rdms)
            rendered[section] = "\n".join(
                (
                    f"% Matthews {method_label} norm, 1-RDM, and 2-RDM",
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
            "metric": f"matthews_{args.method}_metric.tex",
            "sigma": f"matthews_{args.method}_sigma.tex",
            "one-rdm": f"matthews_{args.method}_one_rdm.tex",
            "two-rdm": f"matthews_{args.method}_two_rdm.tex",
            "rdm": f"matthews_{args.method}_rdms.tex",
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
