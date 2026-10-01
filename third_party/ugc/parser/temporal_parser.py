import re
from typing import Callable, Tuple

from dateutil import parser

from dataclasses import dataclass

from ugc.parser.base_parser import BaseParser, Match
from ugc.parser.temporal_enums import TemporalLevel, TemporalModifier, Season
from ugc.parser.temporal_value import TemporalValue
from ugc.parser.utils import strip_brackets, extract_markdown_formatting


@dataclass
class TemporalMatch(Match[TemporalValue | None]):
    pass


class TemporalParser(BaseParser):
    MONTHS = (
        "january|february|march|april|may|june|"
        "july|august|september|october|november|december|"
        "jan\\.?|feb\\.?|mar\\.?|apr\\.?|jun\\.?|jul\\.?|"
        "aug\\.?|sep\\.?|sept\\.?|oct\\.?|nov\\.?|dec\\.?"
    )

    ERA_PATTERN = r"(?:b\.c\.e\.|b\.c\.|bce|bc)(?!\w)"

    DATE_PATTERNS = [
        rf"\b(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*,?\s+|\s*\([^)]*\)\s*)\d{{1,4}}\s*(?:{ERA_PATTERN})?(?!\w)",
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})(?:\s+|\s*\([^)]*\)\s*)\d{{1,4}}\s*(?:{ERA_PATTERN})?(?!\w)",
        rf"\b(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?\s+in\s+\d{{1,4}}\s*(?:{ERA_PATTERN})?(?!\w)",
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})\s+in\s+\d{{1,4}}\s*(?:{ERA_PATTERN})?(?!\w)",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{1,4}\b",
        r"\b\d{1,4}-\d{1,2}-\d{1,2}\b",
        r"\b\d{1,2}\.\d{1,2}\.\d{1,4}\b",
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+of\s+(?:{MONTHS})(?:\s+|\s*\([^)]*\)\s*)\d{{1,4}}\s*(?:{ERA_PATTERN})?(?!\w)",
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+of\s+(?:{MONTHS})\s+in\s+\d{{1,4}}\s*(?:{ERA_PATTERN})?(?!\w)",
    ]

    SEASON_PATTERN = rf"\b(spring|summer|autumn|winter)(?:\s+(?:in|of))?\s+(\d{{1,4}})(?![\-–/])\s*({ERA_PATTERN})?(?!\w)"
    EARLY_MID_YEAR_PATTERN = rf"\b(early|mid|late)(?:\s+in)?\s+(\d{{1,4}})(?![\-–/])\s*({ERA_PATTERN})?(?!\w)"
    EARLY_MID_DECADE_PATTERN = rf"\b(early|mid|late)(?:\s+in)?\s+(?:the\s+)?(\d{{1,4}})s\s*({ERA_PATTERN})?(?!\w)"
    DECADE_PATTERN = rf"\b(\d{{1,4}})s\s*({ERA_PATTERN})?(?!\w)"

    CENTURY_PATTERN = rf"\b(\d{{1,2}})(?:st|nd|rd|th)\s+century\s*({ERA_PATTERN})?(?!\w)"

    EARLY_MID_CENTURY_PATTERN = rf"\b(early|mid|late)(?:-|\s+)(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)\s+century\s*({ERA_PATTERN})?(?!\w)"

    YEAR_PATTERN = rf"\b(\d{{1,4}})\s*({ERA_PATTERN})?(?!\w)"
    YEAR_RANGE_PATTERN = rf"\b(\d{{1,4}})\s*(?:and|or|to|-|/)\s*(\d{{1,4}})\s*({ERA_PATTERN})"

    CENTURY_RANGE_PATTERN = (
        rf"\b"
        rf"(\d{{1,2}})(?:st|nd|rd|th)"
        rf"(?:\s+century)?"
        rf"\s*(?:and|or|to|-|/)\s*"
        rf"(\d{{1,2}})(?:st|nd|rd|th)"
        rf"\s+century"
        rf"\s*({ERA_PATTERN})"
    )

    EARLY_MID_CENTURY_RANGE_PATTERN = rf"\b(early|mid|late)(?:-|\s+)(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)\s+century\s*(?:and|or|to|-|/)\s*(early|mid|late)(?:-|\s+)(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)\s+century\s*({ERA_PATTERN})"

    TEMPORAL_CONTEXT_PATTERN =r"\b(born|died|in|around)\b"

    @staticmethod
    def _apply_era(year: int, era: str | None) -> int:
        if era is None:
            return year

        era = era.lower().replace(".", "")
        if era and era.lower() in {"bc", "bce"}:
            return -year

        return year

    @staticmethod
    def _month_number(month_text: str) -> int:
        return parser.parse(
            f"{month_text} 1 2000"
        ).month


    @staticmethod
    def _match(
        m: re.Match,
        value: TemporalValue | None,
        group: int | None = None,
    ) -> TemporalMatch:
        return TemporalMatch(
            start=m.start(group) if group else m.start(),
            end=m.end(group) if group else m.end(),
            value=value,
        )

    @classmethod
    def _parse_wikidata_timestamp(
        cls,
        text: str,
    ) -> list[TemporalMatch]:
        m = re.fullmatch(
            r"([+-]?\d+)-(\d{2})-(\d{2})T\d{2}:\d{2}:\d{2}Z",
            text,
            flags=re.IGNORECASE,
        )

        if not m:
            return []

        year, month, day = map(int, m.groups())

        return [
            cls._match(
                m,
                TemporalValue(
                    level=TemporalLevel.DAY,
                    year=year,
                    month=month,
                    day=day,
                )
            )
        ]

    @classmethod
    def _parse_day(
        cls,
        text: str,
    ) -> list[TemporalMatch]:
        matches = []

        for pattern in cls.DATE_PATTERNS:
            for m in cls._find(pattern, text):

                date_text = m.group(0)

                is_bce = bool(
                    re.search(
                        r"\b(?:bc|bce)\b",
                        date_text,
                        flags=re.IGNORECASE,
                    )
                )

                date_text = re.sub(
                    r"\b(?:bc|bce)\b",
                    "",
                    date_text,
                    flags=re.IGNORECASE,
                )

                date_text = re.sub(
                    r"\([^)]*\)",
                    "",
                    date_text,
                )

                date_text = re.sub(
                    r"\s+in\s+",
                    " ",
                    date_text,
                    flags=re.IGNORECASE,
                )

                try:
                    dt = parser.parse(
                        date_text,
                        fuzzy=False,
                    )

                    year = -dt.year if is_bce else dt.year

                    value = TemporalValue(
                        level=TemporalLevel.DAY,
                        year=year,
                        month=dt.month,
                        day=dt.day,
                    )

                except Exception:
                    value = None

                matches.append(
                    cls._match(m, value)
                )

        return matches

    @classmethod
    def _parse_month(
        cls,
        text: str,
    ) -> list[TemporalMatch]:
        return [
            cls._match(
                m,
                TemporalValue(
                    level=TemporalLevel.MONTH,
                    year=cls._apply_era(
                        int(year),
                        era,
                    ),
                    month=cls._month_number(month_text),
                ),
            )
            for m in cls._find(
                rf"\b({cls.MONTHS})(?:\s+in)?\s+(\d{{1,4}})\s*(bc|bce)?\b",
                text,
            )
            for month_text, year, era in [m.groups()]
        ]

    @classmethod
    def _parse_season(
        cls,
        text: str,
    ) -> list[TemporalMatch]:
        return [
            cls._match(
                m,
                TemporalValue(
                    level=TemporalLevel.SEASON,
                    year=cls._apply_era(
                        int(year),
                        era,
                    ),
                    season=Season(
                        season.lower()
                    ),
                ),
            )
            for m in cls._find(cls.SEASON_PATTERN, text)
            for season, year, era in [m.groups()]
        ]

    @classmethod
    def _parse_early_mid_year(
        cls,
        text: str,
    ) -> list[TemporalMatch]:
        return [
            cls._match(
                m,
                TemporalValue(
                    level=TemporalLevel.EARLYMIDYEAR,
                    year=cls._apply_era(
                        int(year),
                        era,
                    ),
                    year_modifier=TemporalModifier(
                        modifier.lower()
                    )
                ),
            )
            for m in cls._find(cls.EARLY_MID_YEAR_PATTERN, text)
            for modifier, year, era in [m.groups()]
        ]

    @classmethod
    def _parse_early_mid_decade(
        cls,
        text: str,
    ) -> list[TemporalMatch]:
        return [
            cls._match(
                m,
                TemporalValue(
                    level=TemporalLevel.EARLYMIDDECADE,
                    decade=cls._apply_era(
                        int(decade),
                        era,
                    ),
                    decade_modifier=TemporalModifier(
                        modifier.lower()
                    )
                ),
            )
            for m in cls._find(cls.EARLY_MID_DECADE_PATTERN, text)
            for modifier, decade, era in [m.groups()]
        ]

    @classmethod
    def _parse_decade(
        cls,
        text: str,
    ) -> list[TemporalMatch]:
        return [
            cls._match(
                m,
                TemporalValue(
                    level=TemporalLevel.DECADE,
                    decade=cls._apply_era(
                        int(decade),
                        era,
                    ),
                ),
            )
            for m in cls._find(cls.DECADE_PATTERN, text)
            for decade, era in [m.groups()]
        ]

    @classmethod
    def _parse_early_mid_century(
        cls,
        text: str,
    ) -> list[TemporalMatch]:
        matches = []

        for m in cls._find(cls.EARLY_MID_CENTURY_RANGE_PATTERN, text):
            start_modifier, start_century, end_modifier, end_century, era = m.groups()

            matches.extend(
                [
                    cls._match(
                        m,
                        TemporalValue(
                            level=TemporalLevel.EARLYMIDCENTURY,
                            century=cls._apply_era(
                                int(start_century),
                                era,
                            ),
                            century_modifier=TemporalModifier(start_modifier.lower()),
                        ),
                        group=1,
                    ),
                    cls._match(
                        m,
                        TemporalValue(
                            level=TemporalLevel.EARLYMIDCENTURY,
                            century=cls._apply_era(
                                int(end_century),
                                era,
                            ),
                            century_modifier=TemporalModifier(end_modifier.lower()),
                        ),
                        group=2,
                    ),
                ]
            )

        matches.extend([
            cls._match(
                m,
                TemporalValue(
                    level=TemporalLevel.EARLYMIDCENTURY,
                    century=cls._apply_era(
                        int(century),
                        era,
                    ),
                    century_modifier=TemporalModifier(
                        modifier.lower()
                    )
                ),
            )
            for m in cls._find(cls.EARLY_MID_CENTURY_PATTERN, text)
            for modifier, century, era in [m.groups()]
        ])

        return matches

    @classmethod
    def _parse_century(
        cls,
        text: str,
    ) -> list[TemporalMatch]:
        matches = []

        for m in cls._find(cls.CENTURY_RANGE_PATTERN, text):
            start_century, end_century, era = m.groups()

            matches.extend(
                [
                    cls._match(
                        m,
                        TemporalValue(
                            level=TemporalLevel.CENTURY,
                            century=cls._apply_era(
                                int(start_century),
                                era,
                            ),
                        ),
                        group=1,
                    ),
                    cls._match(
                        m,
                        TemporalValue(
                            level=TemporalLevel.CENTURY,
                            century=cls._apply_era(
                                int(end_century),
                                era,
                            ),
                        ),
                        group=2,
                    ),
                ]
            )

        matches.extend([
            cls._match(
                m,
                TemporalValue(
                    level=TemporalLevel.CENTURY,
                    century=cls._apply_era(
                        int(century),
                        era,
                    ),
                ),
            )
            for m in cls._find(cls.CENTURY_PATTERN, text)
            for century, era in [m.groups()]
        ])
        return matches

    @classmethod
    def _parse_year(
        cls,
        text: str,
    ) -> list[TemporalMatch]:
        matches = []

        for m in cls._find(cls.YEAR_RANGE_PATTERN, text):
            start_year, end_year, era = m.groups()

            matches.extend(
                [
                    cls._match(
                        m,
                        TemporalValue(
                            level=TemporalLevel.YEAR,
                            year=cls._apply_era(
                                int(start_year),
                                era,
                            ),
                        ),
                        group=1,
                    ),
                    cls._match(
                        m,
                        TemporalValue(
                            level=TemporalLevel.YEAR,
                            year=cls._apply_era(
                                int(end_year),
                                era,
                            ),
                        ),
                        group=2,
                    ),
                ]
            )

        for m in cls._find(cls.YEAR_PATTERN, text):
            year, era = m.groups()

            if cls._looks_like_list_item(text, year):
                continue

            year_int = int(year)
            prefix = text[max(0, m.start() - 20):m.start()]

            has_temporal_context = bool(
                re.search(
                    cls.TEMPORAL_CONTEXT_PATTERN,
                    prefix,
                    flags=re.IGNORECASE,
                )
            )

            if (
                era is None
                and year_int < 1000
                and not has_temporal_context
            ):
                continue

            matches.append(
                cls._match(
                    m,
                    TemporalValue(
                        level=TemporalLevel.YEAR,
                        year=cls._apply_era(
                            int(year),
                            era,
                        ),
                    )
                )
            )

        return matches

    @staticmethod
    def _looks_like_list_item(
        text: str,
        raw_year: str,
    ) -> bool:
        if int(raw_year) >= 1000:
            return False

        return bool(
            re.search(
                r"(?:^|\n)\s*" + re.escape(raw_year) + r"\.",
                text,
            )
        )

    @classmethod
    def parse_matches(
        cls,
        text: str,
        remove_brackets: bool = True,
    ) -> list[TemporalMatch]:
        if remove_brackets:
            text = strip_brackets(text)

        text = text.strip().lower()

        parse_functions: list[Callable[[str], list[TemporalMatch]]] = [
            cls._parse_wikidata_timestamp,
            cls._parse_day,
            cls._parse_month,
            cls._parse_season,
            cls._parse_early_mid_century,
            cls._parse_century,
            cls._parse_early_mid_decade,
            cls._parse_decade,
            cls._parse_early_mid_year,
            cls._parse_year,
        ]

        occupied = []
        matches = []

        for parse_fn in parse_functions:
            for match in parse_fn(text):
                if cls._overlaps(
                    match.start,
                    match.end,
                    occupied,
                ):
                    continue

                occupied.append(
                    (match.start, match.end)
                )

                if match.value is not None:
                    matches.append(match)

        matches.sort(
            key=lambda match: match.start
        )

        return matches

    @classmethod
    def parse(
            cls,
            text: str,
            remove_brackets: bool = True,
    ) -> list[TemporalValue]:
        return list(
            dict.fromkeys(
                match.value
                for match in cls.parse_matches(
                    text,
                    remove_brackets=remove_brackets,
                )
            )
        )

    @classmethod
    def first_most_specific(cls, text: str) -> TemporalValue | None:
        values = cls.parse(text)
        if not values:
            return None

        max_precision = max(
            v.precision
            for v in values
        )

        for v in values:
            if v.precision == max_precision:
                return v

        return None

    @classmethod
    def replace_value(
            cls,
            text: str,
            old_value: TemporalValue,
            new_value: TemporalValue,
            remove_brackets: bool = True,
            return_replacement: bool = False,
            remove_marker_highlights: bool = True,
    ) -> tuple[str] | str:
        if remove_brackets:
            text = strip_brackets(text)
        matches = cls.parse_matches(text)

        for match in reversed(matches):
            if match.value == old_value:
                start, end, replacement, attended_replacement = cls._adjust_span_for_replacement(
                    text,
                    match,
                    new_value,
                    remove_marker_highlights,
                )

                text = text[:start] + replacement + text[end:]

        if return_replacement:
            return text, attended_replacement
        return text

    @classmethod
    def replace_single_value(
            cls,
            text: str,
            new_value: TemporalValue,
            remove_brackets: bool = True,
            return_replacement: bool = False,
            remove_marker_highlights: bool = True,
    ) -> tuple[str, str] | str:
        if remove_brackets:
            text = strip_brackets(text)

        matches = cls.parse_matches(text)

        if len(matches) == 0:
            raise ValueError(
                "No temporal value found in text."
            )

        if len(matches) > 1:
            raise ValueError(
                f"Expected exactly one temporal value, "
                f"found {len(matches)}."
            )

        match = matches[0]
        return cls.replace_match(text, match, new_value, False, return_replacement, remove_marker_highlights)

    @classmethod
    def replace_match(
            cls,
            text: str,
            old_match: TemporalMatch,
            new_value: TemporalValue,
            remove_brackets: bool = True,
            return_replacement: bool = False,
            remove_marker_highlights: bool = True,
    ):
        if remove_brackets:
            text = strip_brackets(text)

        start, end, replacement, attended_replacement = (
            cls._adjust_span_for_replacement(
                text,
                old_match,
                new_value,
                remove_marker_highlights
            )
        )

        text = (
                text[:start]
                + replacement
                + text[end:]
        )

        if return_replacement:
            return text, attended_replacement

        return text

    @staticmethod
    def _adjust_span_for_replacement(
            text: str,
            match: TemporalMatch,
            new_value: TemporalValue,
            remove_marker_highlights: bool = True,
    ) -> tuple[int, int, str, str]:
        raw_start = match.start
        raw_end = match.end

        start, end, marker = extract_markdown_formatting(
            text,
            raw_start,
            raw_end,
        )

        prefix_before_span = text[:start]

        prep_before_marker = re.search(
            r"\b(in\s+the\s+year|in\s+the|in|on\s+the|during|during\s+the|around|around\s+the|on)\s+$",
            prefix_before_span,
            flags=re.IGNORECASE,
        )
        if prep_before_marker:
            start = prep_before_marker.start()
            prep_before_marker = prep_before_marker.group(1).lower()
        else:
            prep_before_marker = ""

        prep_inside_marker = ""
        if marker:
            prefix_inside_marker = text[start + len(marker):raw_start]
            prep_inside_marker = re.search(
                r"\b(in\s+the\s+year|in\s+the|in|on\s+the|during|during\s+the|around|around\s+the|on|the)\s+$",
                prefix_inside_marker,
                flags=re.IGNORECASE,
            )
            if prep_inside_marker:
                prep_inside_marker = prep_inside_marker.group(1).lower()
            else:
                prep_inside_marker = ""

        if remove_marker_highlights:
            marker = ""

        new_level = new_value.level
        prep_before_marker, prep_inside_marker = TemporalParser.adapt_marker(prep_before_marker, prep_inside_marker, new_level)

        before = f"{prep_before_marker} " if prep_before_marker else ""
        inside = f"{prep_inside_marker} " if prep_inside_marker else ""

        attended_replacement = (
            f"{before}"
            f"{marker}"
            f"{inside}"
            f"{new_value.to_text()}"
        )
        replacement = f"{attended_replacement}{marker}"
        return start, end, replacement, attended_replacement

    @staticmethod
    def adapt_marker(prep_before_marker: str, prep_inside_marker: str, new_level) -> Tuple[str, str]:
        """Possible preps: "in", "in the", "on", "the", "on the", "around", "around the", "during", "during the" """
        combined_marker = " ".join(
            part
            for part in (
                prep_before_marker,
                prep_inside_marker,
            )
            if part
        )
        if combined_marker not in {
            "",
            "in",
            "in the",
            "in the year",
            "on",
            "on the",
            "around",
            "around the",
            "during",
            "during the"
        }:
            raise ValueError(f"Unexpected marker combination: {combined_marker!r}")

        if not combined_marker:
            if new_level not in {
                TemporalLevel.DECADE,
                TemporalLevel.EARLYMIDDECADE,
                TemporalLevel.EARLYMIDCENTURY,
                TemporalLevel.CENTURY,
            }:
                return "", ""

        if combined_marker == "in the year":
            if new_level != TemporalLevel.YEAR:
                combined_marker = "in"

        if new_level in {
            TemporalLevel.DECADE,
            TemporalLevel.EARLYMIDDECADE,
            TemporalLevel.EARLYMIDCENTURY,
            TemporalLevel.CENTURY,
        }:
            if "around" in combined_marker:
                adapted_marker = "around the"
            elif "during" in combined_marker:
                adapted_marker = "during the"
            else:
                adapted_marker = "in the"
        elif new_level == TemporalLevel.DAY:
            adapted_marker = "on"
        else:
            adapted_marker = combined_marker.replace("on", "in")

        before_had_content = bool(prep_before_marker)
        inside_had_content = bool(prep_inside_marker)

        if before_had_content and not inside_had_content:
            return adapted_marker, ""
        if not before_had_content and inside_had_content:
            return "", adapted_marker

        if before_had_content and inside_had_content:
            parts = adapted_marker.split(" ", maxsplit=1)

            if len(parts) == 1:
                return parts[0], ""

            return parts[0], parts[1]

        return adapted_marker, ""
