"""CISDT contraction tables restricted to the CIDT excitation manifold."""

from __future__ import annotations

from ..cisdt._contractions import (
    NORM_TERMS as CISDT_NORM_TERMS,
    ONE_RDM_SPECS as CISDT_ONE_RDM_SPECS,
    SIGMA_SPECS as CISDT_SIGMA_SPECS,
    TWO_RDM_SPECS as CISDT_TWO_RDM_SPECS,
)


_SINGLES = {"c1", "c1_conj"}


def _without_singles(terms):
    return tuple(
        term
        for term in terms
        if not any(source in _SINGLES for source, _spaces in term[2])
    )


SIGMA_SPECS = {
    rank: {
        "spaces": CISDT_SIGMA_SPECS[rank]["spaces"],
        "terms": _without_singles(CISDT_SIGMA_SPECS[rank]["terms"]),
    }
    for rank in (0, 2, 3)
}
NORM_TERMS = _without_singles(CISDT_NORM_TERMS)
ONE_RDM_SPECS = {
    spaces: {
        "spaces": spec["spaces"],
        "terms": _without_singles(spec["terms"]),
    }
    for spaces, spec in CISDT_ONE_RDM_SPECS.items()
}
TWO_RDM_SPECS = {
    spaces: {
        "spaces": spec["spaces"],
        "terms": _without_singles(spec["terms"]),
    }
    for spaces, spec in CISDT_TWO_RDM_SPECS.items()
}
