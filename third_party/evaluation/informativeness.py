import math

from ugc.parser.temporal_enums import TemporalLevel
from ugc.parser.temporal_value import TemporalValue

PRIOR_WIDTH_YEARS = {
    "half_millennium": 500,
    "millennium": 1_000,
    "3_2_millennium": 1_500,
    "two_millennia": 2_000,
}

WIDTH_IN_YEARS = {
    TemporalLevel.CENTURY: 100,
    TemporalLevel.EARLYMIDCENTURY: 100 / 3,
    TemporalLevel.DECADE: 10,
    TemporalLevel.EARLYMIDDECADE: 10 / 3,
    TemporalLevel.YEAR: 1,
    TemporalLevel.SEASON: 1 / 4,
    TemporalLevel.EARLYMIDYEAR: 1 / 3,
    TemporalLevel.MONTH: 1 / 12,
    TemporalLevel.DAY: 1 / 365,
}


def informativeness(
    value: TemporalValue | None,
    prior_width_years: float = 1_000,
) -> float:
    """
    Normalized information gain relative to a prior interval.
    """

    if value is None:
        return 0.0

    max_information = math.log(
        prior_width_years / WIDTH_IN_YEARS[TemporalLevel.DAY]
    )

    return (
        math.log(
            prior_width_years / WIDTH_IN_YEARS[value.level]
        ) / max_information
    )