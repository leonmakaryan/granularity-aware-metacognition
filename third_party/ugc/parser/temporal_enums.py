from enum import StrEnum


class TemporalLevel(StrEnum):
    CENTURY = "CENTURY"
    EARLYMIDCENTURY = "EARLYMIDCENTURY"
    DECADE = "DECADE"
    EARLYMIDDECADE = "EARLYMIDDECADE"
    YEAR = "YEAR"
    SEASON = "SEASON"
    EARLYMIDYEAR = "EARLYMIDYEAR"
    MONTH = "MONTH"
    DAY = "DAY"


class TemporalModifier(StrEnum):
    EARLY = "early"
    MID = "mid"
    LATE = "late"


class Season(StrEnum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"