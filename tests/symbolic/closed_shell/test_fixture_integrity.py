"""Fast integrity checks for the versioned Milestone 0 fixtures."""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path


FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = FIXTURES.parents[3]
EXPECTED = {
    "hjo_cis.json",
    "hjo_cid.json",
    "hjo_cisd.json",
    "spin_orbital_cis.json",
    "spin_orbital_cid.json",
    "spin_orbital_cisd.json",
    "matthews_metric.json",
    "matthews_cisdt_one_rdm.json",
    "matthews_cisdt_two_rdm.json",
    "matthews_status.json",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_expected_fixture_set_is_present() -> None:
    present = {path.name for path in FIXTURES.glob("*.json")}
    assert EXPECTED <= present


def test_fixture_provenance_and_source_hashes() -> None:
    for name in EXPECTED:
        payload = _load(FIXTURES / name)
        assert payload["schema_version"] == 1
        assert payload["conventions"] == "CONVENTIONS.md"
        provenance = payload["provenance"]
        assert provenance["generator"].startswith(
            "tests/symbolic/closed_shell/fixtures/"
        )
        for source in provenance["sources"]:
            path = REPO_ROOT / source["path"]
            assert path.is_file(), source["path"]
            assert sha256(path.read_bytes()).hexdigest() == source["sha256"]


def test_expression_fixtures_are_nonempty_and_deterministically_sorted() -> None:
    for name in EXPECTED:
        payload = _load(FIXTURES / name)
        expressions = payload.get("expressions")
        if expressions is None:
            continue
        assert expressions
        for section in expressions.values():
            assert section
            for record in section.values():
                if "terms" not in record:
                    assert record["expression"]
                    assert record["srepr"]
                    continue
                assert record["term_count"] == len(record["terms"])
                assert record["term_count"] > 0
                keys = [
                    (
                        tuple(
                            (summation["range"], summation["index"])
                            for summation in term["sums"]
                        ),
                        term["amplitude"],
                    )
                    for term in record["terms"]
                ]
                assert keys == sorted(keys)


def _fraction_matrix(values):
    return [[Fraction(value) for value in row] for row in values]


def _matmul(left, right):
    return [
        [
            sum(
                left[row][inner] * right[inner][column]
                for inner in range(len(right))
            )
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def test_matthews_metric_fixture_identities() -> None:
    ranks = _load(FIXTURES / "matthews_metric.json")["ranks"]
    assert ranks["2"]["eigenvalues"] == ["2", "6"]
    assert ranks["3"]["eigenvalues"] == ["0", "6", "6", "6", "6", "24"]

    for record in ranks.values():
        metric = _fraction_matrix(record["metric"])
        pseudoinverse = _fraction_matrix(record["pseudoinverse"])
        physical = _fraction_matrix(record["physical_projector"])
        null = _fraction_matrix(record["null_projector"])
        zero = [[Fraction(0) for _ in row] for row in metric]
        assert _matmul(metric, pseudoinverse) == physical
        assert _matmul(pseudoinverse, metric) == physical
        assert _matmul(metric, null) == zero
        assert _matmul(pseudoinverse, null) == zero


def test_matthews_rdm_fixture_inventory_and_term_counts() -> None:
    one = _load(FIXTURES / "matthews_cisdt_one_rdm.json")["expressions"]
    two = _load(FIXTURES / "matthews_cisdt_two_rdm.json")["expressions"]

    assert one["norm"][r"\langle\widetilde\Psi|\Psi\rangle"]["term_count"] == 7
    assert {
        label: record["term_count"]
        for label, record in one["one_rdm"].items()
    } == {
        r"\gamma^{j}_{i}": 14,
        r"\gamma^{a}_{i}": 7,
        r"\gamma^{i}_{a}": 7,
        r"\gamma^{b}_{a}": 7,
    }
    assert {
        label: record["term_count"]
        for label, record in two["two_rdm"].items()
    } == {
        r"\Gamma^{kl}_{ij}": 50,
        r"\Gamma^{ak}_{ij}": 22,
        r"\Gamma^{jk}_{ai}": 22,
        r"\Gamma^{ab}_{ij}": 8,
        r"\Gamma^{ij}_{ab}": 8,
        r"\Gamma^{bj}_{ai}": 22,
        r"\Gamma^{bj}_{ia}": 22,
        r"\Gamma^{bc}_{ai}": 8,
        r"\Gamma^{ci}_{ab}": 8,
        r"\Gamma^{cd}_{ab}": 8,
    }
