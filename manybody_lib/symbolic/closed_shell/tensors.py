"""Scalar tensor factors and declared monoterm symmetries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .indices import Index, OrbitalSpace


@dataclass(frozen=True, slots=True)
class PermutationSymmetry:
    """A slot permutation with phase ``+1`` or ``-1``.

    ``permutation[position]`` gives the source slot placed at ``position``.
    """

    permutation: tuple[int, ...]
    phase: int = 1

    def __post_init__(self) -> None:
        if sorted(self.permutation) != list(range(len(self.permutation))):
            raise ValueError("permutation must contain every slot exactly once")
        if self.phase not in (-1, 1):
            raise ValueError("a monoterm symmetry phase must be +1 or -1")

    def apply(self, indices: tuple[Index, ...]) -> tuple[Index, ...]:
        if len(indices) != len(self.permutation):
            raise ValueError("symmetry valence does not match tensor valence")
        return tuple(indices[source] for source in self.permutation)


@dataclass(frozen=True, slots=True)
class TensorSymbol:
    """Metadata shared by scalar tensor elements."""

    name: str
    rank: int
    upper_count: int = 0
    symmetries: tuple[PermutationSymmetry, ...] = ()
    latex_name: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a tensor symbol needs a nonempty name")
        if self.rank < 0:
            raise ValueError("tensor rank must be nonnegative")
        if not 0 <= self.upper_count <= self.rank:
            raise ValueError("upper_count must lie between zero and rank")
        for symmetry in self.symmetries:
            if len(symmetry.permutation) != self.rank:
                raise ValueError("tensor symmetry valence does not match rank")

    def __call__(self, *indices: Index, conjugated: bool = False) -> Tensor:
        return Tensor(self, tuple(indices), conjugated=conjugated)


@dataclass(frozen=True, slots=True)
class Tensor:
    """One indexed scalar tensor factor."""

    symbol: TensorSymbol
    indices: tuple[Index, ...]
    conjugated: bool = False

    def __post_init__(self) -> None:
        if len(self.indices) != self.symbol.rank:
            raise ValueError(
                f"{self.symbol.name} expects {self.symbol.rank} indices, "
                f"got {len(self.indices)}"
            )
        if not all(isinstance(index, Index) for index in self.indices):
            raise TypeError("tensor indices must be Index objects")

    def conjugate(self) -> Tensor:
        return Tensor(self.symbol, self.indices, not self.conjugated)

    def replace_indices(self, substitutions: dict[Index, Index]) -> Tensor:
        return Tensor(
            self.symbol,
            tuple(substitutions.get(index, index) for index in self.indices),
            self.conjugated,
        )


@dataclass(frozen=True, slots=True)
class Delta:
    """A Kronecker delta between two typed indices."""

    left: Index
    right: Index

    def __post_init__(self) -> None:
        if not isinstance(self.left, Index) or not isinstance(self.right, Index):
            raise TypeError("delta arguments must be Index objects")

    @property
    def indices(self) -> tuple[Index, Index]:
        return self.left, self.right

    def replace_indices(self, substitutions: dict[Index, Index]) -> Delta:
        return Delta(
            substitutions.get(self.left, self.left),
            substitutions.get(self.right, self.right),
        )


@dataclass(frozen=True, slots=True)
class OrbitalDimension:
    """The dimension produced by a closed dummy-only delta component."""

    space: OrbitalSpace

    def __post_init__(self) -> None:
        if not isinstance(self.space, OrbitalSpace):
            raise TypeError("dimension space must be an OrbitalSpace")

    @property
    def indices(self) -> tuple[()]:
        return ()

    def replace_indices(self, substitutions: dict[Index, Index]):
        del substitutions
        return self


ScalarFactor: TypeAlias = Tensor | Delta | OrbitalDimension


def simultaneous_column_symmetries(columns: int) -> tuple[PermutationSymmetry, ...]:
    """Return adjacent generators that exchange upper/lower columns together."""

    if columns < 1:
        raise ValueError("columns must be positive")
    generators = []
    for first in range(columns - 1):
        second = first + 1
        permutation = list(range(2 * columns))
        permutation[first], permutation[second] = (
            permutation[second],
            permutation[first],
        )
        lower_first = columns + first
        lower_second = columns + second
        permutation[lower_first], permutation[lower_second] = (
            permutation[lower_second],
            permutation[lower_first],
        )
        generators.append(PermutationSymmetry(tuple(permutation), 1))
    return tuple(generators)


def antisymmetric_group_symmetries(
    rank: int,
    slots: tuple[int, ...],
) -> tuple[PermutationSymmetry, ...]:
    """Return adjacent antisymmetric exchanges within ``slots``."""

    if any(slot < 0 or slot >= rank for slot in slots):
        raise ValueError("antisymmetric slots must lie within the tensor rank")
    generators = []
    for left, right in zip(slots, slots[1:]):
        permutation = list(range(rank))
        permutation[left], permutation[right] = permutation[right], permutation[left]
        generators.append(PermutationSymmetry(tuple(permutation), -1))
    return tuple(generators)
