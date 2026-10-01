from collections import Counter
from dataclasses import dataclass, fields
from math import ceil

from ugc.parser.temporal_constants import LEVEL_FIELDS, PRECISION, LEVEL_EVAL_FIELDS
from ugc.parser.temporal_enums import TemporalModifier, TemporalLevel, Season


@dataclass(frozen=True)
class TemporalValue:

    level: TemporalLevel

    year: int | None = None
    month: int | None = None
    day: int | None = None

    decade: int | None = None
    century: int | None = None

    decade_modifier: TemporalModifier | None = None
    year_modifier: TemporalModifier | None = None
    century_modifier: TemporalModifier | None = None
    season: Season | None = None

    def __post_init__(self) -> None:

        if self.year is not None:

            if self.decade is None:
                object.__setattr__(
                    self,
                    "decade",
                    self._year_to_decade(
                        self.year
                    ),
                )

            if self.century_modifier is None:
                object.__setattr__(
                    self,
                    "century_modifier",
                    self._year_to_century_modifier(
                        self.year
                    ),
                )

            if self.century is None:
                object.__setattr__(
                    self,
                    "century",
                    self._year_to_century(
                        self.year
                    ),
                )

            if self.decade_modifier is None:
                object.__setattr__(
                    self,
                    "decade_modifier",
                    self._year_to_decade_modifier(
                        self.year
                    ),
                )

        if self.decade is not None:

            if self.century is None:
                object.__setattr__(
                    self,
                    "century",
                    self._decade_to_century(
                        self.decade
                    ),
                )

            if self.century_modifier is None:
                object.__setattr__(
                    self,
                    "century_modifier",
                    self._decade_to_century_modifier(
                        self.decade
                    ),
                )

        if self.month is not None:

            if self.season is None:
                object.__setattr__(
                    self,
                    "season",
                    self._month_to_season(
                        self.month
                    ),
                )

            if self.year_modifier is None:
                object.__setattr__(
                    self,
                    "year_modifier",
                    self._month_to_year_modifier(
                        self.month
                    ),
                )

        self._validate_level()

    def _validate_level(self) -> None:

        allowed = LEVEL_FIELDS[self.level]

        for field_name in (
            "year",
            "month",
            "day",
            "decade",
            "century",
            "century_modifier",
            "decade_modifier",
            "year_modifier",
            "season",
        ):

            value = getattr(
                self,
                field_name,
            )

            if (
                value is not None
                and field_name not in allowed
            ):
                raise ValueError(
                    f"{field_name} is not allowed "
                    f"for level {self.level}"
                )

    @staticmethod
    def _year_to_century(
        year: int,
    ) -> int:

        if year > 0:
            return ((year - 1) // 100) + 1

        return -(((abs(year) - 1) // 100) + 1)

    @staticmethod
    def _year_to_decade(
        year: int,
    ) -> int:

        if year >= 0:
            return (year // 10) * 10

        return -((abs(year) // 10) * 10)

    @staticmethod
    def _month_to_season(
            month: int,
    ) -> Season:

        if month in {12, 1, 2}:
            return Season.WINTER

        if month in {3, 4, 5}:
            return Season.SPRING

        if month in {6, 7, 8}:
            return Season.SUMMER

        return Season.AUTUMN

    @staticmethod
    def _month_to_year_modifier(
            month: int,
    ) -> TemporalModifier:

        if month <= 4:
            return TemporalModifier.EARLY

        if month <= 8:
            return TemporalModifier.MID

        return TemporalModifier.LATE

    @staticmethod
    def _year_to_decade_modifier(
            year: int,
    ) -> TemporalModifier:

        offset = abs(year) % 10

        if offset <= 3:
            return TemporalModifier.EARLY

        if offset <= 6:
            return TemporalModifier.MID

        return TemporalModifier.LATE

    @staticmethod
    def _year_to_century_modifier(year: int,) -> TemporalModifier:
        offset = (abs(year) - 1) % 100

        if offset <= 33:
            return TemporalModifier.EARLY
        if offset <= 66:
            return TemporalModifier.MID

        return TemporalModifier.LATE

    @staticmethod
    def _decade_to_century(
        decade: int,
    ) -> int | None:
        if abs(decade) % 100 == 0:  # we cannot know if we have 1900s, as 1900 is 19th century and the rest 20th century
            return None

        return TemporalValue._year_to_century(decade)

    @staticmethod
    def _decade_to_century_modifier(decade: int) -> TemporalModifier | None:
        offset = (abs(decade) // 10) % 10

        if offset == 0:
            return None

        if offset <= 2:
            return TemporalModifier.EARLY

        if offset == 3:
            return None

        if offset <= 5:
            return TemporalModifier.MID

        if offset == 6:
            return None

        return TemporalModifier.LATE

    @property
    def precision(self) -> float:
        return PRECISION[self.level]

    def to_precision(self, precision: int):
        if precision not in {7, 8, 9, 10, 11}:
            raise ValueError('Invalid precision')
        return self.to_level(level = next(k for k, v in PRECISION.items() if v == precision))

    def to_level(
        self,
        level: TemporalLevel,
    ) -> "TemporalValue":

        desired_precision = PRECISION[level]

        if desired_precision > self.precision:
            raise ValueError(
                f"Cannot increase precision from "
                f"{self.precision} to "
                f"{desired_precision}"
            )

        keep_fields = LEVEL_EVAL_FIELDS[level]

        kwargs = {
            "level": level,
        }

        for field in fields(self):

            if field.name == "level":
                continue

            value = getattr(self, field.name)

            if field.name in keep_fields:
                if value is None:
                    raise ValueError(
                        f"Missing value for {field.name}"
                    )

                kwargs[field.name] = value
            else:
                kwargs[field.name] = None

        return TemporalValue(**kwargs)

    def equal_at_level(
        self,
        other: "TemporalValue",
        level: TemporalLevel,
    ) -> bool:

        for field_name in LEVEL_EVAL_FIELDS[level]:

            if (
                getattr(self, field_name)
                != getattr(other, field_name)
            ):
                return False

        return True

    def consistent_with(
        self,
        gold: "TemporalValue",
    ) -> bool:

        if self.precision == gold.precision:

            return self.equal_at_level(
                gold,
                self.level,
            )

        lower, higher = (
            (self, gold)
            if self.precision < gold.precision
            else (gold, self)
        )

        return higher.to_level(
            lower.level
        ).equal_at_level(
            lower,
            lower.level,
        )

    def is_overprecise(
        self,
        gold: "TemporalValue",
    ) -> bool:
        return self.precision > gold.precision

    def max_matching_level(self, other: "TemporalValue") -> TemporalLevel | None:
        common = self.common_value([self, other])
        if not common:
            return None
        return common.level

    def max_matching_precision(
            self,
            other: "TemporalValue",
    ) -> float:

        candidates = sorted(
            set(PRECISION.values()),
            reverse=True,
        )

        for precision in candidates:

            matching_levels = [
                level
                for level, p in PRECISION.items()
                if p == precision
            ]

            for level in matching_levels:

                try:
                    left = self.to_level(level)
                    right = other.to_level(level)

                except ValueError:
                    continue

                if left.equal_at_level(right, level):
                    return precision

        return -1

    @classmethod
    def common_value(
            cls,
            values: list["TemporalValue"],
    ) -> "TemporalValue | None":

        if not values:
            return None

        candidates = list(PRECISION.keys())[::-1]

        for level in candidates:

            try:
                projected = [
                    value.to_level(level)
                    for value in values
                ]
            except ValueError:
                continue

            first = projected[0]

            if all(
                    value.equal_at_level(first, level)
                    for value in projected[1:]
            ):
                return first

        return None

    @classmethod
    def majority_common_value(
            cls,
            values: list["TemporalValue | None"],
            min_share: float = 0.8,
    ) -> "TemporalValue | None":

        if not 0 < min_share <= 1:
            raise ValueError(
                "min_share must be in (0, 1]"
            )

        min_count = ceil(len(values) * min_share)

        values = [
            value
            for value in values
            if value is not None
        ]

        if not values:
            return None

        levels = list(PRECISION.keys())[::-1]

        best = None
        best_count = 0

        for level in levels:

            projected = []

            for value in values:
                try:
                    projected.append(
                        value.to_level(level)
                    )
                except ValueError:
                    continue

            counts = Counter(projected)

            for candidate, count in counts.items():

                if count < min_count:
                    continue

                if (
                        best is None
                        or candidate.precision > best.precision
                        or (
                        candidate.precision == best.precision
                        and count > best_count
                )
                ):
                    best = candidate
                    best_count = count

            if best is not None and best.precision == PRECISION[level]:
                return best

        return best

    def to_text(self) -> str:
        if self.level == TemporalLevel.DAY:
            return (
                f"{self._format_month(self.month)} "
                f"{self.day}, "
                f"{self._format_year(self.year)}"
            )

        if self.level == TemporalLevel.MONTH:
            return (
                f"{self._format_month(self.month)} "
                f"{self._format_year(self.year)}"
            )

        if self.level == TemporalLevel.SEASON:
            return (
                f"{self.season.value} "
                f"{self._format_year(self.year)}"
            )

        if self.level == TemporalLevel.EARLYMIDYEAR:
            return (
                f"{self.year_modifier.value} "
                f"{self._format_year(self.year)}"
            )

        if self.level == TemporalLevel.YEAR:
            return self._format_year(self.year)

        if self.level == TemporalLevel.EARLYMIDDECADE:
            return (
                f"{self.decade_modifier.value} "
                f"{self._format_decade(self.decade)}"
            )

        if self.level == TemporalLevel.DECADE:
            return self._format_decade(self.decade)

        if self.level == TemporalLevel.EARLYMIDCENTURY:
            return (
                f"{self.century_modifier.value} "
                f"{self._format_century(self.century)}"
            )

        if self.level == TemporalLevel.CENTURY:
            return self._format_century(self.century)

        raise ValueError(
            f"Unsupported temporal level: {self.level}"
        )

    @staticmethod
    def _format_year(
        year: int | None,
    ) -> str:
        if year is None:
            raise ValueError("year is required")

        if year < 0:
            return f"{abs(year)} BCE"

        return str(year)

    @staticmethod
    def _format_decade(
        decade: int | None,
    ) -> str:
        if decade is None:
            raise ValueError("decade is required")

        if decade < 0:
            return f"{abs(decade)}s BCE"

        return f"{decade}s"

    @staticmethod
    def _format_century(
        century: int | None,
    ) -> str:
        if century is None:
            raise ValueError("century is required")

        suffix = TemporalValue._ordinal_suffix(
            abs(century)
        )

        if century < 0:
            return f"{abs(century)}{suffix} century BCE"

        return f"{century}{suffix} century"

    @staticmethod
    def _format_month(
        month: int | None,
    ) -> str:
        if month is None:
            raise ValueError("month is required")

        months = {
            1: "January",
            2: "February",
            3: "March",
            4: "April",
            5: "May",
            6: "June",
            7: "July",
            8: "August",
            9: "September",
            10: "October",
            11: "November",
            12: "December",
        }

        return months[month]

    @staticmethod
    def _ordinal_suffix(
        n: int,
    ) -> str:
        if 10 <= n % 100 <= 20:
            return "th"

        return {
            1: "st",
            2: "nd",
            3: "rd",
        }.get(
            n % 10,
            "th",
        )

    def to_dict(self) -> dict:
        data = {
            "level": self.level.value,
        }

        for field_name in LEVEL_EVAL_FIELDS[self.level]:
            value = getattr(self, field_name)

            if hasattr(value, "value"):
                value = value.value
            data[field_name] = value
        return data

    @classmethod
    def from_dict(
            cls,
            data: dict,
    ) -> "TemporalValue":
        data = data.copy()

        data["level"] = TemporalLevel(
            data["level"]
        )

        for field_name in (
                "season",
                "decade_modifier",
                "year_modifier",
                "century_modifier",
        ):
            if field_name in data:
                enum_cls = {
                    "season": Season,
                    "decade_modifier": TemporalModifier,
                    "year_modifier": TemporalModifier,
                    "century_modifier": TemporalModifier,
                }[field_name]

                data[field_name] = enum_cls(
                    data[field_name]
                )

        return cls(**data)