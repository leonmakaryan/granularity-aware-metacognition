import pytest

from ugc.parser.temporal_parser import TemporalParser, TemporalLevel, TemporalValue, TemporalModifier, Season


def test_parse_year():

    value = TemporalParser.parse("1971")
    assert len(value) == 1

    value = value[0]

    assert value.level == "YEAR"
    assert value.year == 1971
    assert value.decade == 1970
    assert value.century == 20


def test_parse_decade():

    value = TemporalParser.parse("1970s")
    assert len(value) == 1

    value = value[0]

    assert value.level == "DECADE"
    assert value.decade == 1970
    assert value.century == 20


def test_parse_early_decade():

    value = TemporalParser.parse("early 1970s")

    assert len(value) == 1

    value = value[0]
    assert value.level == "EARLYMIDDECADE"
    assert value.decade_modifier == "early"
    assert value.decade == 1970


def test_parse_mid_decade():

    value = TemporalParser.parse("mid 1970s")

    assert len(value) == 1

    value = value[0]
    assert value.level == "EARLYMIDDECADE"
    assert value.decade_modifier == "mid"
    assert value.decade == 1970


def test_parse_late_decade():

    value = TemporalParser.parse("late 1970s")

    assert len(value) == 1

    value = value[0]
    assert value.level == "EARLYMIDDECADE"
    assert value.decade_modifier == "late"
    assert value.decade == 1970


def test_parse_season():

    value = TemporalParser.parse("summer 1971")

    assert len(value) == 1

    value = value[0]
    assert value.level == "SEASON"
    assert value.season == "summer"
    assert value.year == 1971


def test_parse_month_year():

    value = TemporalParser.parse("June 1971")

    assert len(value) == 1

    value = value[0]
    assert value.level == "MONTH"
    assert value.month == 6
    assert value.year == 1971


def test_parse_full_date():

    value = TemporalParser.parse("June 28, 1971")

    assert len(value) == 1

    value = value[0]
    assert value.level == "DAY"
    assert value.day == 28
    assert value.month == 6
    assert value.year == 1971


def test_parse_full_date_alternative_format():

    value = TemporalParser.parse("28 June 1971")

    assert len(value) == 1

    value = value[0]
    assert value.level == "DAY"
    assert value.day == 28
    assert value.month == 6
    assert value.year == 1971


def test_parse_early_year():

    value = TemporalParser.parse("early 1971")

    assert len(value) == 1

    value = value[0]
    assert value.level == "EARLYMIDYEAR"
    assert value.year_modifier == "early"
    assert value.year == 1971


def test_parse_mid_year():

    value = TemporalParser.parse("mid 1971")

    assert len(value) == 1

    value = value[0]
    assert value.level == "EARLYMIDYEAR"
    assert value.year_modifier == "mid"
    assert value.year == 1971


def test_parse_late_year():

    value = TemporalParser.parse("late 1971")

    assert len(value) == 1

    value = value[0]
    assert value.level == "EARLYMIDYEAR"
    assert value.year_modifier == "late"
    assert value.year == 1971


def test_invalid_text():

    assert len(TemporalParser.parse("I don't know")) == 0


def test_person_name():

    assert len(TemporalParser.parse("Elon Musk")) == 0


def test_empty_string():

    assert len(TemporalParser.parse("")) == 0


def test_no_default_day_month_for_year():

    value = TemporalParser.parse("1971")

    assert len(value) == 1

    value = value[0]
    assert value.level == "YEAR"
    assert value.year == 1971

    # critical: parser must not invent a day/month
    assert value.day is None
    assert value.month is None


def test_no_default_day_for_month_year():

    value = TemporalParser.parse("May 1971")

    assert len(value) == 1

    value = value[0]
    assert value.level == "MONTH"
    assert value.month == 5
    assert value.year == 1971

    # critical: parser must not invent a day
    assert value.day is None


def test_complex():

    value = TemporalParser.parse("He was born in May 1971")

    assert len(value) == 1

    value = value[0]
    assert value.level == "MONTH"
    assert value.month == 5
    assert value.year == 1971

    # critical: parser must not invent a day
    assert value.day is None


def test_numeric_complex():

    value = TemporalParser.parse("He was born in 23.8.1971")

    assert len(value) == 1

    value = value[0]
    assert value.level == "DAY"
    assert value.day == 23
    assert value.month == 8
    assert value.year == 1971


def test_full_date_preferred_over_year():

    value = TemporalParser.parse(
        "Born on June 28, 1971."
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "DAY"


def test_full_date_preferred_over_month():

    value = TemporalParser.parse(
        "June 28, 1971"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "DAY"


def test_decade_not_year():

    value = TemporalParser.parse(
        "Born in the 1970s."
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "DECADE"


def test_early_decade_not_decade():

    value = TemporalParser.parse(
        "Born in the early 1970s."
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "EARLYMIDDECADE"


def test_llm_birth_answer():

    value = TemporalParser.parse(
        "Elon Musk was born on June 28, 1971."
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "DAY"
    assert value.day == 28
    assert value.month == 6
    assert value.year == 1971


def test_markdown_date():

    value = TemporalParser.parse(
        "**June 28, 1971**"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "DAY"


def test_approximate_year():

    value = TemporalParser.parse(
        "around 1971"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "YEAR"
    assert value.year == 1971


def test_parenthesized_date():

    value = TemporalParser.parse(
        "(June 28, 1971)"
    )

    assert len(value) == 0


def test_invalid_date():

    value = TemporalParser.parse(
        "31 February 1971"
    )

    assert len(value) == 0


def test_iso_date():

    value = TemporalParser.parse(
        "1971-06-28"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "DAY"
    assert value.day == 28
    assert value.month == 6
    assert value.year == 1971


def test_year_with_extra_text():

    value = TemporalParser.parse(
        "1971 (approximate)"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "YEAR"
    assert value.year == 1971


def test_decade_with_extra_text():

    value = TemporalParser.parse(
        "probably 1970s"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "DECADE"
    assert value.decade == 1970


def test_numeric_date_with_dots_uses_us_format():

    value = TemporalParser.parse(
        "06.07.1971"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "DAY"
    assert value.day == 7
    assert value.month == 6
    assert value.year == 1971


def test_numeric_date_uses_us_format():

    value = TemporalParser.parse(
        "06-07-1971"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "DAY"
    assert value.day == 7
    assert value.month == 6
    assert value.year == 1971


def test_numeric_date_uses_us_format_three():

    value = TemporalParser.parse(
        "06-07-197"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "DAY"
    assert value.day == 7
    assert value.month == 6
    assert value.year == 197


def test_early_decade_preferred_over_year():

    value = TemporalParser.parse(
        "early 1970s"
    )
    assert len(value) == 1

    value = value[0]
    assert value.level == "EARLYMIDDECADE"


def test_decade_preferred_over_year():

    value = TemporalParser.parse(
        "1970s"
    )
    assert len(value) == 1

    value = value[0]
    assert value.level == "DECADE"


def test_iso_date_inside_text():

    value = TemporalParser.parse(
        "Born on 1971-06-28."
    )
    assert len(value) == 1

    value = value[0]
    assert value.level == "DAY"


def test_invalid_numeric_date():

    value = TemporalParser.parse(
        "02/31/1971"
    )

    assert len(value) == 0


def test_bce():
    value = TemporalParser.parse(
        "Śāriputra was born in the year 623 BCE in what is now Afghanistan"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "YEAR"
    assert value.year == -623


def test_month_year_bce():

    value = TemporalParser.parse(
        "Śāriputra was born in June 623 BCE"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "MONTH"
    assert value.year == -623
    assert value.month == 6
    assert value.decade == -620
    assert value.century == -7


def test_season_bce():

    value = TemporalParser.parse(
        "Śāriputra was born in summer 623 BCE"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "SEASON"
    assert value.year == -623
    assert value.season == "summer"
    assert value.decade == -620
    assert value.century == -7


def test_late_year_bce():

    value = TemporalParser.parse(
        "late 623 BCE"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "EARLYMIDYEAR"
    assert value.year == -623
    assert value.year_modifier == "late"
    assert value.decade == -620
    assert value.century == -7


def test_full_date_bce():

    value = TemporalParser.parse(
        "June 15, 623 BCE"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "DAY"
    assert value.year == -623
    assert value.month == 6
    assert value.day == 15
    assert value.decade == -620
    assert value.century == -7


def test_full_date_bce_alternative():

    value = TemporalParser.parse(
        "15 June 623 BCE"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "DAY"
    assert value.year == -623
    assert value.month == 6
    assert value.day == 15
    assert value.decade == -620
    assert value.century == -7


def test_early_decade_bce():

    value = TemporalParser.parse(
        "early 620s BCE"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "EARLYMIDDECADE"
    assert value.decade == -620
    assert value.decade_modifier == "early"


def test_decade_bce():

    value = TemporalParser.parse(
        "620s BCE"
    )

    assert len(value) == 1

    value = value[0]
    assert value.level == "DECADE"
    assert value.decade == -620


def test_enumeration():
    value = TemporalParser.parse(
        'ing:\n\n1. **Pre", "In B'
    )
    assert len(value) == 0


def test_wikidata():
    value = TemporalParser.parse(
        "1572-01-01T00:00:00Z"
    )
    assert len(value) == 1

    value = value[0]
    assert value.level == "DAY"
    assert value.year == 1572
    assert value.month == 1
    assert value.day == 1

    value = TemporalParser.parse(
        "-0489-01-01T00:00:00Z"
    )
    assert len(value) == 1

    value = value[0]
    assert value.level == "DAY"
    assert value.year == -489
    assert value.month == 1
    assert value.day == 1


def test_multiple_dates():

    values = TemporalParser.parse(
        "Xerxes I, who reigned from 486 to 465 BCE, was born in 519 BCE."
    )

    years = sorted(
        v.year
        for v in values
        if v.year is not None
    )

    assert years == [-519, -486, -465]


def test_multiple_full_dates():

    values = TemporalParser.parse(
        "Born June 28, 1971 and died May 10, 2020."
    )
    assert values != []
    assert len(values) == 2

    assert values[0].level == "DAY"
    assert values[0].year == 1971
    assert values[0].month == 6
    assert values[0].day == 28

    assert values[1].level == "DAY"
    assert values[1].year == 2020
    assert values[1].month == 5
    assert values[1].day == 10


def test_prefer_longest_match():
    values = TemporalParser.parse(
        "Born on June 28, 1971."
    )

    assert len(values) == 1

    value = values[0]
    assert value.level == "DAY"
    assert value.year == 1971
    assert value.month == 6
    assert value.day == 28


def test_invalid_date_does_not_block_other_dates():

    values = TemporalParser.parse(
        "31 February 1971 and June 28, 1980"
    )

    assert len(values) == 1

    assert values[0].year == 1980
    assert values[0].month == 6
    assert values[0].day == 28


def test_overlap_resolution():

    values = TemporalParser.parse(
        "June 28, 1971"
    )

    assert len(values) == 1

    assert values[0].level == "DAY"


def test_multiple_granularities():

    values = TemporalParser.parse(
        "He was born on June 28, 1971 and died in summer 2020."
    )

    assert len(values) == 2

    assert values[0].level == "DAY"
    assert values[0].year == 1971
    assert values[0].month == 6
    assert values[0].day == 28

    assert values[1].level == "SEASON"
    assert values[1].year == 2020
    assert values[1].season == "summer"


def test_multiple_mixed_granularities():

    values = TemporalParser.parse(
        "Born in 1971, active in the 1990s, died June 28, 2020."
    )

    assert len(values) == 3

    assert values[0].level == "YEAR"
    assert values[1].level == "DECADE"
    assert values[2].level == "DAY"


def test_full_with_comma():
    values = TemporalParser.parse(
        "496/495 BCE"
    )
    assert len(values) == 2
    assert values[0].year == -496
    assert values[1].year == -495


def test_full_with_():
    values = TemporalParser.parse(
        "496-495 BCE"
    )
    assert len(values) == 2
    assert values[0].year == -496
    assert values[1].year == -495


def test_full_with_s():
    values = TemporalParser.parse(
        "496 - 495 BCE"
    )
    assert len(values) == 2
    assert values[0].year == -496
    assert values[1].year == -495


def test_bce_with_and():
    values = TemporalParser.parse(
        "496 and 495 BCE"
    )
    assert len(values) == 2
    assert values[0].year == -496
    assert values[1].year == -495


def test_abb_month():
    values = TemporalParser.parse(
        "Jean Renoir was born on **Aug. 9, 1888**."
    )
    assert len(values) == 1
    assert values[0].year == 1888
    assert values[0].day == 9
    assert values[0].month == 8


def test_day_in_year():

    values = TemporalParser.parse(
        "Charles V, Holy Roman Emperor was born on February 21 in 1500."
    )

    assert values == [
        TemporalValue(
            level=TemporalLevel.DAY,
            year=1500,
            month=2,
            day=21,
            decade=1500,
            century=15,
            decade_modifier=TemporalModifier.EARLY,
            year_modifier=TemporalModifier.EARLY,
            season=Season.WINTER,
        )
    ]

def test_in_with_range():
    values = TemporalParser.parse(
        "to fall in 322–319"
    )
    assert len(values) == 2
    assert values[0].level == TemporalLevel.YEAR
    assert values[0].year == 322

    assert values[1].level == TemporalLevel.YEAR
    assert values[1].year == 319

def test_month_in_year():

    values = TemporalParser.parse(
        "Charles V, Holy Roman Emperor was born February in 1500."
    )

    assert values == [
        TemporalValue(
            level=TemporalLevel.MONTH,
            year=1500,
            month=2,
            decade=1500,
            century=15,
            decade_modifier=TemporalModifier.EARLY,
            year_modifier=TemporalModifier.EARLY,
            season=Season.WINTER,
        )
    ]


def test_season_in_year():

    values = TemporalParser.parse(
        "Charles V, Holy Roman Emperor was born summer in 1500."
    )

    assert values == [
        TemporalValue(
            level=TemporalLevel.SEASON,
            year=1500,
            season=Season.SUMMER,
            decade=1500,
            century=15,
            decade_modifier=TemporalModifier.EARLY,
        )
    ]


def test_early_in_decade():

    values = TemporalParser.parse(
        "Charles V, Holy Roman Emperor was born early in 1970s."
    )

    assert values == [
        TemporalValue(
            level=TemporalLevel.EARLYMIDDECADE,
            decade=1970,
            century=20,
            decade_modifier=TemporalModifier.EARLY,
        )
    ]

def test_brackets_between():
    values = TemporalParser.parse(
        "October 5 (September in the Russian calendar) 1763"
    )
    assert len(values) == 1
    assert values[0].year == 1763
    assert values[0].month == 10
    assert values[0].day == 5


def test_without_year():
    values = TemporalParser.parse(
        "February 14"
    )

    assert len(values) == 1

    assert values[0].level == TemporalLevel.MONTH
    # This gets parses as February in Year 14. But is fine for now


def test_bce_point():
    values = TemporalParser.parse(
        "1350 B.C. and 1352 B.C."
    )

    assert len(values) == 2
    assert values[0].year == -1350
    assert values[1].year == -1352


def test_to_text1():
    text = "Ingrid Bergman was born **May 24, 1920** (October 23, 1923 in Stockholm, Sweden)."
    old_value = TemporalParser.parse("May 24, 1920")[0]
    new_text = TemporalParser.replace_value(text, old_value, old_value.to_level(TemporalLevel.DECADE), remove_marker_highlights=False)

    assert new_text == "Ingrid Bergman was born in the **1920s**."

def test_to_text2():
    text = "Ingrid Bergman was born **May 24, 1920** (October 23, 1923 in Stockholm, Sweden)."
    old_value = TemporalParser.parse("May 24, 1920")[0]
    new_text = TemporalParser.replace_value(text, old_value, old_value.to_level(TemporalLevel.DECADE))

    assert new_text == "Ingrid Bergman was born in the 1920s."


def test_to_text_year():
    text = "Cary Grant was born on April 12, 1905."
    old_value = TemporalParser.parse("April 12, 1905")[0]
    new_text = TemporalParser.replace_value(text, old_value, old_value.to_level(TemporalLevel.YEAR))

    assert new_text == "Cary Grant was born in 1905."


def replace_century_text():
    text = "Cary Grant was born on April 12, 1905."
    old_value = TemporalParser.parse("April 12, 1905")[0]
    new_text = TemporalParser.replace_value(text, old_value, old_value.to_level(TemporalLevel.CENTURY))

    assert new_text == "Cary Grant was born in the 20th century."


def test_replace_early_mid_century_text():
    text = "Cary Grant was born on April 12, 1905."
    old_value = TemporalParser.parse("April 12, 1905")[0]
    new_text = TemporalParser.replace_value(text, old_value, old_value.to_level(TemporalLevel.EARLYMIDCENTURY))

    assert new_text == "Cary Grant was born in the early 20th century."



def test_to_intervall_text_year():
    """
    This does not fully resolve to a logic sentence,
    but since for training we focus on sentences with only one temp value this is fine.
    """
    text = "Cary Grant was born in 1955 or 1958 BCE."
    old_value = TemporalParser.parse("1955 BCE")[0]
    new_text = TemporalParser.replace_value(text, old_value, old_value.to_level(TemporalLevel.DECADE))

    assert new_text == 'Cary Grant was born in the 1950s BCE or 1958 BCE.'


def test_early_century():

    values = TemporalParser.parse(
        "Charles V, Holy Roman Emperor was born in the early 19th century."
    )

    assert values == [
        TemporalValue(
            level=TemporalLevel.EARLYMIDCENTURY,
            century=19,
            century_modifier=TemporalModifier.EARLY,
        )
    ]


def test_mid_century():
    values = TemporalParser.parse(
        "He lived during the mid-19th century."
    )

    assert values == [
        TemporalValue(
            level=TemporalLevel.EARLYMIDCENTURY,
            century=19,
            century_modifier=TemporalModifier.MID,
        )
    ]


def test_multiple_centuries():
    values = TemporalParser.parse(
        "between the 17th century and the 19th century"
    )

    assert values == [
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=17,
        ),
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=19,
        ),
    ]


def test_century():
    values = TemporalParser.parse(
        "Charles V, Holy Roman Emperor was born between the 22nd century."
    )

    assert values == [
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=22,
        ),
    ]


def test_century_frame():
    values = TemporalParser.parse(
        "Charles V, Holy Roman Emperor was born between the late 20th century and early 18th century BC"
    )

    assert values == [
        TemporalValue(
            level=TemporalLevel.EARLYMIDCENTURY,
            century=-20,
            century_modifier=TemporalModifier.LATE,
        ),
        TemporalValue(
            level=TemporalLevel.EARLYMIDCENTURY,
            century=-18,
            century_modifier=TemporalModifier.EARLY,
        ),
    ]


def test_century_frame_easy():
    values = TemporalParser.parse(
        "Charles V, Holy Roman Emperor was born between the 20th and 18th century BC"
    )

    assert values == [
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=-20,
        ),
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=-18,
        ),
    ]


def test_century_frame_easy2():
    values = TemporalParser.parse(
        "Charles V, Holy Roman Emperor was born between the 20th century and 18th century BC"
    )

    assert values == [
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=-20,
        ),
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=-18,
        ),
    ]


def century_minus_test():
    values = TemporalParser.parse("early 15th-century")
    assert values == [
        TemporalValue(
            level=TemporalLevel.EARLYMIDCENTURY,
            century=15,
        )
    ]


def century_minus_test2():
    values = TemporalParser.parse("15th-century")
    assert values == [
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=15,
        )
    ]


def test_weird_ones():
    values = TemporalParser.parse("Yohannes IV was born in the 19th century. More specifically, he was born in the 1840s. However, the exact year of his birth is not readily available in historical records. Yohannes IV was the 22")
    assert values == [
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=19,
        ),
        TemporalValue(
            level=TemporalLevel.DECADE,
            century=19,
            decade=1840,
        )
    ]


def test_first_most_specifc():
    value = TemporalParser.first_most_specific("Yohannes IV was born in the 19th century. More specifically, he was born in the 1840s or 1950s.")
    assert value == TemporalValue(
            level=TemporalLevel.DECADE,
            century=19,
            decade=1840,
        )

def test_remove_brackets_in_replace():
    value = TemporalValue(level=TemporalLevel.YEAR, year=1742)
    text = 'Nicéphore Niépce (Coined as \"Nicéphore Niépce, French [Nizan] (1732), born in 1742\") was born'
    new_text = TemporalParser.replace_value(text, old_value=value, new_value=value.to_level(TemporalLevel.DECADE))
    assert new_text == 'Nicéphore Niépce was born'


def test_duplicate_in1():
    value = TemporalValue(level=TemporalLevel.YEAR, year=1950)
    text = "Florence Griffith Joyner was born in **1950**"
    new_text, replacement = TemporalParser.replace_value(text, old_value=value, new_value=value.to_level(TemporalLevel.CENTURY), return_replacement=True, remove_marker_highlights=False)
    assert new_text == 'Florence Griffith Joyner was born in the **20th century**'
    assert replacement == 'in the **20th century'


def test_duplicate_in2():
    value = TemporalValue(level=TemporalLevel.YEAR, year=1950)
    text = "Florence Griffith Joyner was born in **1950**"
    new_text, replacement = TemporalParser.replace_value(text, old_value=value, new_value=value.to_level(TemporalLevel.CENTURY), return_replacement=True)
    assert new_text == 'Florence Griffith Joyner was born in the 20th century'
    assert replacement == 'in the 20th century'


def test_replace_single_value_year_to_century():
    text = "Florence Griffith Joyner was born in 1950"

    new_text = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=20,
        ),
    )

    assert (
        new_text
        == "Florence Griffith Joyner was born in the 20th century"
    )


def test_replace_single_value_day_to_season():
    text = "Florence Griffith Joyner was born on September 25, 1906"

    new_text = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.SEASON,
            year=1906,
            season=Season.AUTUMN,
        ),
    )

    assert (
        new_text
        == "Florence Griffith Joyner was born in autumn 1906"
    )


def test_replace_single_value_no_temporal_value():
    text = "Florence Griffith Joyner was born somewhere in California"

    with pytest.raises(
        ValueError,
        match="No temporal value found",
    ):
        TemporalParser.replace_single_value(
            text,
            TemporalValue(
                level=TemporalLevel.CENTURY,
                century=20,
            ),
        )


def test_replace_single_value_multiple_temporal_values():
    text = (
        "Florence Griffith Joyner was born in 1950 "
        "and died in 1998"
    )

    with pytest.raises(
        ValueError,
        match="Expected exactly one temporal value",
    ):
        TemporalParser.replace_single_value(
            text,
            TemporalValue(
                level=TemporalLevel.CENTURY,
                century=20,
            ),
        )


def test_replace_single_value_return_replacement():
    text = "Florence Griffith Joyner was born in 1950"

    new_text, replacement = (
        TemporalParser.replace_single_value(
            text,
            TemporalValue(
                level=TemporalLevel.CENTURY,
                century=20,
            ),
            return_replacement=True,
        )
    )

    assert (
        new_text
        == "Florence Griffith Joyner was born in the 20th century"
    )

    assert replacement == "in the 20th century"


def test_replace_single_value_preserves_markdown1():
    text = "Florence Griffith Joyner was born in **1950**"

    new_text, replacement = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=15,
        ),
        return_replacement=True,
        remove_marker_highlights=False,
    )

    assert (
        new_text
        == "Florence Griffith Joyner was born in the **15th century**"
    )
    assert replacement == "in the **15th century"


def test_replace_single_value_preserves_markdown2():
    text = "Florence Griffith Joyner was born in **1950**"

    new_text, replacement = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=15,
        ),
        return_replacement=True,
    )

    assert (
        new_text
        == "Florence Griffith Joyner was born in the 15th century"
    )
    assert replacement == "in the 15th century"


def test_replace_single_value_removes_brackets():
    text = 'was born in 1742'

    new_text, replacement = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.DAY,
            day=12,
            month=12,
            year=1842,
        ),
        return_replacement=True
    )

    assert (
        new_text
        == "was born on December 12, 1842"
    )
    assert replacement == "on December 12, 1842"



def test_replace_single_value_without_bracket_removal():
    text = 'Nicéphore Niépce was born in 1742 although he likes the 12.3.2026'

    with pytest.raises(
        ValueError,
        match="Expected exactly one temporal value",
    ):
        TemporalParser.replace_single_value(
            text,
            TemporalValue(
                level=TemporalLevel.DECADE,
                decade=1740,
            ),
        )

def test_strange_marker():
    text = "Claude Monet was born **March 14, 1840**"

    new_text, replacement = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.DAY,
            day=12,
            month=12,
            year=1842,
        ),
        return_replacement=True,
        remove_marker_highlights=False,
    )

    assert (
        new_text
        == "Claude Monet was born **December 12, 1842**"
    )
    assert replacement == "**December 12, 1842"


def test_strange_no_marker():
    text = "Claude Monet was born **March 14, 1840**"

    new_text, replacement = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.DAY,
            day=12,
            month=12,
            year=1842,
        ),
        return_replacement=True
    )

    assert (
        new_text
        == "Claude Monet was born December 12, 1842"
    )
    assert replacement == "December 12, 1842"



def test_strange2():
    text = "Tokhtamysh was likely born in **the mid-14th century**"

    new_text, replacement = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=19,
        ),
        return_replacement=True,
        remove_marker_highlights=False,
    )

    assert new_text == "Tokhtamysh was likely born in **the 19th century**"
    assert replacement == "in **the 19th century"


def test_strange3():
    text = "Tokhtamysh was likely born in the mid-14th century"

    new_text, replacement = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=19,
        ),
        return_replacement=True
    )

    assert new_text == "Tokhtamysh was likely born in the 19th century"
    assert replacement == "in the 19th century"


def test_strange4():
    text = "Tokhtamysh was likely born in **the mid-14th century**"

    new_text, replacement = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.DAY,
            day=12,
            month=12,
            year=1842,
        ),
        return_replacement=True,
        remove_marker_highlights=False,
    )

    assert new_text == "Tokhtamysh was likely born on **December 12, 1842**"
    assert replacement == "on **December 12, 1842"


def test_strange5():
    text = "Tokhtamysh was likely born in **1948**"

    new_text, replacement = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.DAY,
            day=12,
            month=12,
            year=1842,
        ),
        return_replacement=True,
        remove_marker_highlights=False,
    )

    assert new_text == "Tokhtamysh was likely born on **December 12, 1842**"
    assert replacement == "on **December 12, 1842"

def test_strange6():
    text = "Paco de Lucía was born on the 12th of April 1947."

    new_text, replacement = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.DAY,
            day=12,
            month=12,
            year=1842,
        ),
        return_replacement=True
    )

    assert new_text == "Paco de Lucía was born on December 12, 1842."
    assert replacement == "on December 12, 1842"


def test_parse_strange6():
    text = "Paco de Lucía was born on the 12th of April 1947."

    values = TemporalParser.parse(text)

    assert len(values) == 1
    assert values[0].level == TemporalLevel.DAY
    assert values[0].day == 12
    assert values[0].month == 4
    assert values[0].year == 1947


def test_replace_on_the_to_century():
    text = "Paco de Lucía was born on the 12th of April 1947."

    new_text = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=20,
        ),
    )

    assert (
        new_text
        == "Paco de Lucía was born in the 20th century."
    )


def test_replace_full_marker_inside_markdown1():
    text = "Tokhtamysh was likely born **in the mid-14th century**"

    new_text, replacement = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.DAY,
            day=12,
            month=12,
            year=1842,
        ),
        return_replacement=True,
        remove_marker_highlights=False
    )

    assert (
        new_text
        == "Tokhtamysh was likely born **on December 12, 1842**"
    )
    assert replacement == "**on December 12, 1842"


def test_replace_full_marker_inside_markdown2():
    text = "Tokhtamysh was likely born **in the mid-14th century**"

    new_text, replacement = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.DAY,
            day=12,
            month=12,
            year=1842,
        ),
        return_replacement=True,
    )

    assert (
        new_text
        == "Tokhtamysh was likely born on December 12, 1842"
    )
    assert replacement == "on December 12, 1842"


def test_replace_multiple_occurrences():
    text = "Born in 1950 and died in 1998."

    value = TemporalValue(
        level=TemporalLevel.YEAR,
        year=1950,
    )

    new_text = TemporalParser.replace_value(
        text,
        value,
        value.to_level(TemporalLevel.DECADE),
    )

    assert (
        new_text
        == "Born in the 1950s and died in 1998."
    )


def test_replace_bce_year_to_decade():
    text = "Śāriputra was born in 623 BCE."

    new_text = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.DECADE,
            decade=-620,
        ),
    )

    assert (
        new_text
        == "Śāriputra was born in the 620s BCE."
    )


def test_replace_day_with_day():
    text = "Born on April 12, 1905."

    new_text = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.DAY,
            day=13,
            month=4,
            year=1905,
        ),
    )

    assert (
        new_text
        == "Born on April 13, 1905."
    )


def test_replace_preserves_non_temporal_preposition():
    text = "Émile Zola was born on **Paul Barret in 1845**."

    new_text = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.DAY,
            year=1845,
            month=12,
            day=12,
        ),
    )

    assert (
        new_text
        == "Émile Zola was born on **Paul Barret on December 12, 1845**."
    )


def test_replace_in_the_year():
    text = "Muhammad was born on the 24th of Ramadan in the year 870"

    new_text = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.DAY,
            year=1845,
            month=12,
            day=12,
        ),
    )

    assert (
        new_text
        == "Muhammad was born on the 24th of Ramadan on December 12, 1845"
    )


def test_replace_in_the_year_season():
    text = "Muhammad was born on the 24th of Ramadan in the year 870"

    new_text = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.SEASON,
            year=1845,
            season=Season.WINTER
        ),
    )

    assert (
        new_text
        == "Muhammad was born on the 24th of Ramadan in winter 1845"
    )

def test_fictional_character_no_false_positive_year():
    values = TemporalParser.parse(
        "Zara Yaqobi is a fictional character from *The 100* television series, not a real person."

    )
    assert values == []


def test_around():
    text = 'Askia Muhammad I (known as Askia the Great) was born around **1440**'

    new_text = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=15,
        ),
    )

    assert (
        new_text
        == "Askia Muhammad I was born around the 15th century"
    )


def test_around2():
    text = 'Askia Muhammad I was born around the 15th century'

    new_text = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.DAY,
            day=14,
            month=12,
            year=1554,
        ),
    )

    assert (
        new_text
        == "Askia Muhammad I was born on December 14, 1554"
    )


def test_during():
    text = 'Askia Muhammad I (known as Askia the Great) was born during **1440**'

    new_text = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.CENTURY,
            century=15,
        ),
    )

    assert (
        new_text
        == "Askia Muhammad I was born during the 15th century"
    )


def test_during2():
    text = 'Askia Muhammad I was born during the 15th century'

    new_text = TemporalParser.replace_single_value(
        text,
        TemporalValue(
            level=TemporalLevel.DAY,
            day=14,
            month=12,
            year=1554,
        ),
    )

    assert (
        new_text
        == "Askia Muhammad I was born on December 14, 1554"
    )


def test_parse():
    text = 'Shaka, the transformative leader credited with founding the Zulu Kingdom, died in 1828. His exact date of death is not definitively recorded in historical accounts, but historians generally place his passing sometime around the spring of 1828.'

    values = TemporalParser.parse_matches(text)
    assert values[0].value.level == TemporalLevel.YEAR
    assert values[1].value.level == TemporalLevel.SEASON

