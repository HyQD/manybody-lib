"""Typed orbital indices for closed-shell symbolic algebra."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import total_ordering


class OrbitalSpace(Enum):
    """Index domains understood by the closed-shell engine."""

    OCCUPIED = "occupied"
    VIRTUAL = "virtual"
    GENERAL = "general"
    SPIN = "spin"


class IndexRole(Enum):
    """Whether an index is external or Einstein summed."""

    FREE = "free"
    DUMMY = "dummy"


_SPACE_ORDER = {
    OrbitalSpace.OCCUPIED: 0,
    OrbitalSpace.VIRTUAL: 1,
    OrbitalSpace.GENERAL: 2,
    OrbitalSpace.SPIN: 3,
}
_ROLE_ORDER = {IndexRole.FREE: 0, IndexRole.DUMMY: 1}


@total_ordering
@dataclass(frozen=True, slots=True)
class Index:
    """An immutable index whose domain and role participate in equality."""

    name: str
    space: OrbitalSpace
    role: IndexRole = IndexRole.FREE

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("an index needs a nonempty display name")
        if not isinstance(self.space, OrbitalSpace):
            raise TypeError("space must be an OrbitalSpace")
        if not isinstance(self.role, IndexRole):
            raise TypeError("role must be an IndexRole")

    @property
    def is_dummy(self) -> bool:
        return self.role is IndexRole.DUMMY

    @property
    def is_free(self) -> bool:
        return self.role is IndexRole.FREE

    def sort_key(self) -> tuple[int, int, str]:
        return _SPACE_ORDER[self.space], _ROLE_ORDER[self.role], self.name

    def renamed(self, name: str) -> Index:
        return Index(name, self.space, self.role)

    def with_space(self, space: OrbitalSpace) -> Index:
        return Index(self.name, space, self.role)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return NotImplemented
        return self.sort_key() < other.sort_key()


def occupied(name: str, *, dummy: bool = False) -> Index:
    return Index(
        name,
        OrbitalSpace.OCCUPIED,
        IndexRole.DUMMY if dummy else IndexRole.FREE,
    )


def virtual(name: str, *, dummy: bool = False) -> Index:
    return Index(
        name,
        OrbitalSpace.VIRTUAL,
        IndexRole.DUMMY if dummy else IndexRole.FREE,
    )


def general(name: str, *, dummy: bool = False) -> Index:
    return Index(
        name,
        OrbitalSpace.GENERAL,
        IndexRole.DUMMY if dummy else IndexRole.FREE,
    )


def spin(name: str, *, dummy: bool = False) -> Index:
    return Index(
        name,
        OrbitalSpace.SPIN,
        IndexRole.DUMMY if dummy else IndexRole.FREE,
    )


def compatible_spaces(left: OrbitalSpace, right: OrbitalSpace) -> bool:
    """Return whether a Kronecker delta can identify two index spaces."""

    if left is right:
        return True
    if OrbitalSpace.SPIN in (left, right):
        return False
    return OrbitalSpace.GENERAL in (left, right)


def merged_space(spaces) -> OrbitalSpace | None:
    """Return the most specific compatible space, or ``None``."""

    spaces = set(spaces)
    if not spaces:
        raise ValueError("cannot merge an empty set of spaces")
    specific = spaces - {OrbitalSpace.GENERAL}
    if len(specific) > 1:
        return None
    if specific:
        return next(iter(specific))
    return OrbitalSpace.GENERAL
