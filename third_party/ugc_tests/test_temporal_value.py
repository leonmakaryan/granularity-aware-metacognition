import pytest

from ugc.parser.temporal_parser import TemporalValue, TemporalLevel, TemporalModifier, Season


def test_consistent_same_year():

    pred = TemporalValue(
        level="YEAR",
        year=2020,
    )

    gold = TemporalValue(
        level="YEAR",
        year=2020,
    )

    assert pred.consistent_with(gold)


def test_consistent_year_vs_day():

    pred = TemporalValue(
        level="YEAR",
        year=2020,
    )

    gold = TemporalValue(
        level="DAY",
        year=2020,
        month=8,
        day=18,
    )

    assert pred.consistent_with(gold)


def test_season_conflicts_with_month():

    pred = TemporalValue(
        level="SEASON",
        year=2021,
        season="summer",
    )

    gold = TemporalValue(
        level="MONTH",
        year=2021,
        month=1,
    )

    assert not pred.consistent_with(gold)


def test_month_conflicts_with_season():

    pred = TemporalValue(
        level="MONTH",
        year=2021,
        month=1,
    )

    gold = TemporalValue(
        level="SEASON",
        year=2021,
        season="summer",
    )

    assert not pred.consistent_with(gold)


def test_early_year_contains_month():

    pred = TemporalValue(
        level="EARLYMIDYEAR",
        year=2021,
        year_modifier="early",
    )

    gold = TemporalValue(
        level="MONTH",
        year=2021,
        month=3,
    )

    assert pred.consistent_with(gold)


def test_early_year_conflicts_month():

    pred = TemporalValue(
        level="EARLYMIDYEAR",
        year=2021,
        year_modifier="early",
    )

    gold = TemporalValue(
        level="MONTH",
        year=2021,
        month=8,
    )

    assert not pred.consistent_with(gold)


def test_early_decade_contains_year():

    pred = TemporalValue(
        level="EARLYMIDDECADE",
        decade=1970,
        decade_modifier="early",
    )

    gold = TemporalValue(
        level="YEAR",
        year=1972,
    )

    assert pred.consistent_with(gold)


def test_early_decade_conflicts_year():

    pred = TemporalValue(
        level="EARLYMIDDECADE",
        decade=1970,
        decade_modifier="early",
    )

    gold = TemporalValue(
        level="YEAR",
        year=1978,
    )

    assert not pred.consistent_with(gold)


def test_is_overprecise():

    pred = TemporalValue(
        level="MONTH",
        year=2020,
        month=8,
    )

    gold = TemporalValue(
        level="YEAR",
        year=2020,
    )

    assert pred.is_overprecise(gold)


def test_is_not_overprecise():

    pred = TemporalValue(
        level="DECADE",
        decade=2020,
    )

    gold = TemporalValue(
        level="YEAR",
        year=2023,
    )

    assert not pred.is_overprecise(gold)


def test_season_overprecise_year():

    pred = TemporalValue(
        level="SEASON",
        year=2020,
        season="summer",
    )

    gold = TemporalValue(
        level="YEAR",
        year=2020,
    )

    assert pred.is_overprecise(gold)


def test_early_mid_decade_overprecise_decade():

    pred = TemporalValue(
        level="EARLYMIDDECADE",
        decade=1970,
        decade_modifier="early",
    )

    gold = TemporalValue(
        level="DECADE",
        decade=1970,
    )

    assert pred.is_overprecise(gold)


def test_to_precision_cannot_increase_precision():

    value = TemporalValue(
        level="YEAR",
        year=2021,
        decade=2020,
        century=21,
    )

    with pytest.raises(ValueError):
        value.to_level("DAY")


def test_to_precision_day_to_year():

    value = TemporalValue(
        level="DAY",
        year=2020,
        month=8,
        day=18,
        decade=2020,
        century=21,
    )

    result = value.to_level("YEAR")

    assert result.level == "YEAR"

    assert result.year == 2020
    assert result.decade == 2020
    assert result.century == 21

    assert result.month is None
    assert result.day is None


def test_to_precision_day_to_middecade():

    value = TemporalValue(
        level="DAY",
        year=2020,
        month=8,
        day=18,
        decade=2020,
        century=21,
    )

    result = value.to_level("EARLYMIDDECADE")

    assert result.level == "EARLYMIDDECADE"

    assert result.decade_modifier == "early"
    assert result.decade == 2020
    assert result.century == 21

    assert result.year is None
    assert result.month is None
    assert result.day is None


def test_to_precision_day_to_season():

    value = TemporalValue(
        level="DAY",
        year=2020,
        month=8,
        day=18,
        decade=2020,
        century=21,
    )

    result = value.to_level("SEASON")

    assert result.level == "SEASON"

    assert result.season == "summer"
    assert result.year == 2020

    assert result.month is None
    assert result.day is None


def test_common_temporal_value_identical_day():
    values = [
        TemporalValue(
            level=TemporalLevel.DAY,
            year=1971,
            month=6,
            day=28,
        ),
        TemporalValue(
            level=TemporalLevel.DAY,
            year=1971,
            month=6,
            day=28,
        ),
    ]

    result = TemporalValue.common_value(values)

    assert result == TemporalValue(
        level=TemporalLevel.DAY,
        year=1971,
        month=6,
        day=28,
    )


def test_common_temporal_value_month():
    values = [
        TemporalValue(
            level=TemporalLevel.DAY,
            year=1971,
            month=6,
            day=28,
        ),
        TemporalValue(
            level=TemporalLevel.DAY,
            year=1971,
            month=6,
            day=30,
        ),
    ]

    result = TemporalValue.common_value(values)

    assert result == TemporalValue(
        level=TemporalLevel.MONTH,
        year=1971,
        month=6,
    )


def test_common_temporal_value_year():
    values = [
        TemporalValue(
            level=TemporalLevel.MONTH,
            year=1971,
            month=6,
        ),
        TemporalValue(
            level=TemporalLevel.MONTH,
            year=1971,
            month=8,
        ),
    ]

    result = TemporalValue.common_value(values)

    assert result == TemporalValue(
        level=TemporalLevel.EARLYMIDYEAR,
        year_modifier="mid",
        year=1971,
    )


def test_common_temporal_value_decade():
    values = [
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=1971,
        ),
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=1978,
        ),
    ]

    result = TemporalValue.common_value(values)

    assert result == TemporalValue(
        level=TemporalLevel.DECADE,
        decade=1970,
    )


def test_common_temporal_value_century():
    values = [
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=1971,
        ),
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=1998,
        ),
    ]

    result = TemporalValue.common_value(values)

    assert result == TemporalValue(
        level=TemporalLevel.EARLYMIDCENTURY,
        century_modifier=TemporalModifier.LATE,
        century=20,
    )


def test_common_temporal_value_none():
    values = [
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=1971,
        ),
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=2071,
        ),
    ]

    result = TemporalValue.common_value(values)

    assert result is None


def test_common_temporal_value_bce_decade():
    values = [
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=-431,
        ),
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=-438,
        ),
    ]

    result = TemporalValue.common_value(values)

    assert result == TemporalValue(
        level=TemporalLevel.DECADE,
        decade=-430,
    )


def test_common_temporal_value_mixed_precisions():
    values = [
        TemporalValue(
            level=TemporalLevel.DAY,
            year=1971,
            month=6,
            day=28,
        ),
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=1971,
        ),
    ]

    result = TemporalValue.common_value(values)

    assert result == TemporalValue(
        level=TemporalLevel.YEAR,
        year=1971,
    )


def test_common_temporal_value_20_identical_samples():
    values = [
        TemporalValue(
            level=TemporalLevel.DAY,
            year=1971,
            month=6,
            day=28,
        )
        for _ in range(20)
    ]

    result = TemporalValue.common_value(values)

    assert result.precision == 11


def test_common_temporal_value_year_modifier():
    values = [
        TemporalValue(
            level=TemporalLevel.EARLYMIDYEAR,
            year=1971,
            year_modifier=TemporalModifier.EARLY,
        ),
        TemporalValue(
            level=TemporalLevel.MONTH,
            year=1971,
            month=1,
        ),
        TemporalValue(
            level=TemporalLevel.MONTH,
            year=1971,
            month=4,
        ),
    ]

    result = TemporalValue.common_value(values)

    assert result == TemporalValue(
        level=TemporalLevel.EARLYMIDYEAR,
        year=1971,
        year_modifier=TemporalModifier.EARLY,
    )


def test_common_temporal_value_season():
    values = [
        TemporalValue(
            level=TemporalLevel.SEASON,
            year=1971,
            season=Season.SUMMER,
        ),
        TemporalValue(
            level=TemporalLevel.MONTH,
            year=1971,
            month=6,
        ),
        TemporalValue(
            level=TemporalLevel.MONTH,
            year=1971,
            month=7,
        ),
    ]

    result = TemporalValue.common_value(values)

    assert result == TemporalValue(
        level=TemporalLevel.SEASON,
        year=1971,
        season=Season.SUMMER,
    )


def test_common_temporal_value_season_and_year_modifier_with_month():
    values = [
        TemporalValue(
            level=TemporalLevel.SEASON,
            year=1971,
            season=Season.SUMMER,
        ),
        TemporalValue(
            level=TemporalLevel.EARLYMIDYEAR,
            year=1971,
            year_modifier=TemporalModifier.MID,
        ),
        TemporalValue(
            level=TemporalLevel.MONTH,
            year=1971,
            month=7,
        ),
    ]

    result = TemporalValue.common_value(values)

    assert result == TemporalValue(
        level=TemporalLevel.YEAR,
        year=1971,
    )


def test_majority_common_prefers_largest_cluster():
    values = [
        TemporalValue(
            level=TemporalLevel.DAY,
            year=1971,
            month=6,
            day=28,
        ),
        TemporalValue(
            level=TemporalLevel.DAY,
            year=1971,
            month=7,
            day=1,
        ),
        TemporalValue(
            level=TemporalLevel.DAY,
            year=1971,
            month=8,
            day=3,
        ),
        TemporalValue(
            level=TemporalLevel.DAY,
            year=1971,
            month=1,
            day=5,
        ),
    ] * 4

    values += [
        TemporalValue(
            level=TemporalLevel.DAY,
            year=1980,
            month=1,
            day=1,
        ),
        TemporalValue(
            level=TemporalLevel.DAY,
            year=1980,
            month=3,
            day=4,
        ),
    ] * 2

    result = TemporalValue.majority_common_value(
        values,
        min_share=0.8
    )

    assert result == TemporalValue(
        level=TemporalLevel.YEAR,
        year=1971,
    )


def test_early_mid_century_text():
    value = TemporalValue(
        level=TemporalLevel.EARLYMIDCENTURY,
        century_modifier=TemporalModifier.EARLY,
        century=19
    )
    text = value.to_text()
    assert text == "early 19th century"


def test_decade_to_century():
    value = TemporalValue(
        level=TemporalLevel.DECADE,
        decade=1730,
    )

    assert value.to_level(
        TemporalLevel.CENTURY
    ) == TemporalValue(
        level=TemporalLevel.CENTURY,
        century=18,
    )


def test_decade_to_early_century():
    value = TemporalValue(
        level=TemporalLevel.DECADE,
        decade=1720,
    )

    assert value.to_level(
        TemporalLevel.EARLYMIDCENTURY
    ) == TemporalValue(
        level=TemporalLevel.EARLYMIDCENTURY,
        century=18,
        century_modifier=TemporalModifier.EARLY,
    )


def test_common_decade_to_early_century():
    values = [
        TemporalValue(
            level=TemporalLevel.DECADE,
            decade=1720,
        ),
        TemporalValue(
            level=TemporalLevel.DECADE,
            decade=1730
        )
    ]

    assert TemporalValue.common_value(values) == TemporalValue(
        level=TemporalLevel.CENTURY,
        century=18,
    )


def test_common_year():
    values = [
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=1720,
        ),
        TemporalValue(
            level=TemporalLevel.DECADE,
            decade=1730
        ),
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=1731
        ),
    ]

    assert TemporalValue.common_value(values) == TemporalValue(
        level=TemporalLevel.CENTURY,
        century=18,
    )


def test_common_year2():
    values = [
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=1720,
        ),
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=1732
        ),
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=1731
        ),
    ]

    assert TemporalValue.common_value(values) == TemporalValue(
        level=TemporalLevel.EARLYMIDCENTURY,
        century=18,
        century_modifier=TemporalModifier.EARLY,
    )


def test_border_temporal_value1():
    value = TemporalValue(
            level=TemporalLevel.YEAR,
            year=1900
    )

    assert value == TemporalValue(
        level=TemporalLevel.YEAR,
        year=1900,
        decade=1900,
        decade_modifier=TemporalModifier.EARLY,
        century_modifier=TemporalModifier.LATE,
        century=19
    )

def test_border_temporal_value2():
    value = TemporalValue(
            level=TemporalLevel.YEAR,
            year=1901
    )

    assert value.level == TemporalLevel.YEAR
    assert value.year == 1901
    assert value.decade == 1900
    assert value.decade_modifier == TemporalModifier.EARLY
    assert value.century_modifier == TemporalModifier.EARLY
    assert value.century == 20

def test_border_temporal_value3():
    value = TemporalValue(
            level=TemporalLevel.DECADE,
            decade=1900
    )

    assert value.level == TemporalLevel.DECADE
    assert value.decade == 1900
    assert value.decade_modifier is None
    assert value.century_modifier is None
    assert value.century is None


def test_decade_1900_to_century_fails():
    value = TemporalValue(
        level=TemporalLevel.DECADE,
        decade=1900,
    )

    with pytest.raises(ValueError):
        value.to_level(TemporalLevel.CENTURY)


def test_border_temporal_value_decade_1930():
    value = TemporalValue(
        level=TemporalLevel.DECADE,
        decade=1930,
    )

    assert value.century == 20
    assert value.century_modifier is None


def test_border_temporal_value_decade_1960():
    value = TemporalValue(
        level=TemporalLevel.DECADE,
        decade=1960,
    )

    assert value.century == 20
    assert value.century_modifier is None


def test_decade_1920_has_early_century_modifier():
    value = TemporalValue(
        level=TemporalLevel.DECADE,
        decade=1920,
    )

    assert value.century_modifier == TemporalModifier.EARLY


def test_decade_1940_has_mid_century_modifier():
    value = TemporalValue(
        level=TemporalLevel.DECADE,
        decade=1940,
    )

    assert value.century_modifier == TemporalModifier.MID


def test_decade_1970_has_late_century_modifier():
    value = TemporalValue(
        level=TemporalLevel.DECADE,
        decade=1970,
    )

    assert value.century_modifier == TemporalModifier.LATE


def test_border_temporal_value_bce1():
    value = TemporalValue(
        level=TemporalLevel.YEAR,
        year=-100,
    )

    assert value.century == -1
    assert value.century_modifier == TemporalModifier.LATE


def test_border_temporal_value_bce2():
    value = TemporalValue(
        level=TemporalLevel.YEAR,
        year=-101,
    )

    assert value.century == -2
    assert value.century_modifier == TemporalModifier.EARLY


def test_bce_decade_boundary():
    assert (
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=-100,
        ).decade
        == -100
    )

    assert (
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=-101,
        ).decade
        == -100
    )

    assert (
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=-110,
        ).decade
        == -110
    )


def test_year_cannot_have_month():
    with pytest.raises(ValueError):
        TemporalValue(
            level=TemporalLevel.YEAR,
            year=2020,
            month=1,
        )


def test_century_cannot_have_year():
    with pytest.raises(ValueError):
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=20,
            year=1901,
        )


def test_common_temporal_value_empty():
    assert TemporalValue.common_value([]) is None


def test_majority_common_temporal_value_empty():
    assert (
        TemporalValue.majority_common_value([])
        is None
    )


def test_max_matching_precision_day_month():
    a = TemporalValue(
        level=TemporalLevel.DAY,
        year=1971,
        month=6,
        day=28,
    )

    b = TemporalValue(
        level=TemporalLevel.DAY,
        year=1971,
        month=6,
        day=30,
    )

    assert (
        a.max_matching_precision(b)
        == TemporalValue(
            level=TemporalLevel.MONTH,
            year=1971,
            month=6,
        ).precision
    )


def test_max_matching_precision_none():
    a = TemporalValue(
        level=TemporalLevel.YEAR,
        year=1971,
    )

    b = TemporalValue(
        level=TemporalLevel.YEAR,
        year=2071,
    )

    assert a.max_matching_precision(b) == -1


def test_bce_year_text():
    value = TemporalValue(
        level=TemporalLevel.YEAR,
        year=-431,
    )

    assert value.to_text() == "431 BCE"


def test_bce_century_text():
    value = TemporalValue(
        level=TemporalLevel.CENTURY,
        century=-5,
    )

    assert value.to_text() == "5th century BCE"


def test_month_to_season_boundaries():
    assert (
        TemporalValue(
            level=TemporalLevel.MONTH,
            year=2020,
            month=2,
        ).season
        == Season.WINTER
    )

    assert (
        TemporalValue(
            level=TemporalLevel.MONTH,
            year=2020,
            month=3,
        ).season
        == Season.SPRING
    )

    assert (
        TemporalValue(
            level=TemporalLevel.MONTH,
            year=2020,
            month=5,
        ).season
        == Season.SPRING
    )

    assert (
        TemporalValue(
            level=TemporalLevel.MONTH,
            year=2020,
            month=6,
        ).season
        == Season.SUMMER
    )

    assert (
        TemporalValue(
            level=TemporalLevel.MONTH,
            year=2020,
            month=8,
        ).season
        == Season.SUMMER
    )

    assert (
        TemporalValue(
            level=TemporalLevel.MONTH,
            year=2020,
            month=9,
        ).season
        == Season.AUTUMN
    )

    assert (
        TemporalValue(
            level=TemporalLevel.MONTH,
            year=2020,
            month=11,
        ).season
        == Season.AUTUMN
    )

    assert (
        TemporalValue(
            level=TemporalLevel.MONTH,
            year=2020,
            month=12,
        ).season
        == Season.WINTER
    )


def test_day_to_year_preserves_year():
    value = TemporalValue(
        level=TemporalLevel.DAY,
        year=1971,
        month=6,
        day=28,
    )

    assert (
        value.to_level(
            TemporalLevel.YEAR
        ).year
        == 1971
    )


def test_day_to_year_consistent():
    value = TemporalValue(
        level=TemporalLevel.DAY,
        year=1971,
        month=6,
        day=28,
    )

    assert value.consistent_with(
        value.to_level(
            TemporalLevel.YEAR
        )
    )