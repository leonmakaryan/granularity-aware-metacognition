import re
from dataclasses import dataclass

from typing import Generic, TypeVar

T = TypeVar("T")

@dataclass
class Match(Generic[T]):
    start: int
    end: int
    value: T

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "value": (
                self.value.to_dict()
                if hasattr(self.value, "to_dict")
                else self.value
            ),
        }


class BaseParser:
    @staticmethod
    def _find(
        pattern: str,
        text: str,
    ):
        return re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _overlaps(
        start: int,
        end: int,
        occupied: list[tuple[int, int]],
    ) -> bool:
        return any(
            start < occ_end and end > occ_start
            for occ_start, occ_end in occupied
        )
