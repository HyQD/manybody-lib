"""Atomic spin-free closed-shell operators and ordered words."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .indices import Index


@dataclass(frozen=True, slots=True)
class OneBodyOperator:
    """The spin-free substitution operator ``E_pq``."""

    p: Index
    q: Index

    def __post_init__(self) -> None:
        if not isinstance(self.p, Index) or not isinstance(self.q, Index):
            raise TypeError("operator indices must be Index objects")

    @property
    def indices(self) -> tuple[Index, Index]:
        return self.p, self.q

    def replace_indices(
        self,
        substitutions: dict[Index, Index],
    ) -> OneBodyOperator:
        return OneBodyOperator(
            substitutions.get(self.p, self.p),
            substitutions.get(self.q, self.q),
        )


@dataclass(frozen=True, slots=True)
class TwoBodyOperator:
    """The spin-free two-electron substitution operator ``e_pqrs``."""

    p: Index
    q: Index
    r: Index
    s: Index

    def __post_init__(self) -> None:
        if not all(
            isinstance(index, Index)
            for index in (self.p, self.q, self.r, self.s)
        ):
            raise TypeError("operator indices must be Index objects")

    @property
    def indices(self) -> tuple[Index, Index, Index, Index]:
        return self.p, self.q, self.r, self.s

    def replace_indices(
        self,
        substitutions: dict[Index, Index],
    ) -> TwoBodyOperator:
        return TwoBodyOperator(
            *(substitutions.get(index, index) for index in self.indices)
        )


SpinFreeOperator: TypeAlias = OneBodyOperator | TwoBodyOperator


@dataclass(frozen=True, slots=True)
class OperatorWord:
    """An ordered product written in its usual left-to-right order."""

    operators: tuple[SpinFreeOperator, ...] = ()

    def __post_init__(self) -> None:
        if not all(
            isinstance(operator, (OneBodyOperator, TwoBodyOperator))
            for operator in self.operators
        ):
            raise TypeError("an operator word contains an invalid operator")

    @property
    def indices(self) -> tuple[Index, ...]:
        return tuple(
            index
            for operator in self.operators
            for index in operator.indices
        )

    def __bool__(self) -> bool:
        return bool(self.operators)

    def __len__(self) -> int:
        return len(self.operators)

    def __mul__(self, other: OperatorWord | SpinFreeOperator) -> OperatorWord:
        if isinstance(other, OperatorWord):
            return OperatorWord(self.operators + other.operators)
        if isinstance(other, (OneBodyOperator, TwoBodyOperator)):
            return OperatorWord(self.operators + (other,))
        return NotImplemented

    def replace_indices(
        self,
        substitutions: dict[Index, Index],
    ) -> OperatorWord:
        return OperatorWord(
            tuple(
                operator.replace_indices(substitutions)
                for operator in self.operators
            )
        )


IDENTITY_WORD = OperatorWord()
