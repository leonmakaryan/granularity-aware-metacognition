from ugc.parser.temporal_enums import TemporalModifier, Season, TemporalLevel

YEAR_MODIFIER_MONTHS = {
    TemporalModifier.EARLY: {1, 2, 3, 4},
    TemporalModifier.MID: {5, 6, 7, 8},
    TemporalModifier.LATE: {9, 10, 11, 12},
}

SEASON_MONTHS = {
    Season.SPRING: {3, 4, 5},
    Season.SUMMER: {6, 7, 8},
    Season.AUTUMN: {9, 10, 11},
    Season.WINTER: {12, 1, 2},
}

PRECISION = {
    TemporalLevel.CENTURY: 7,
    TemporalLevel.EARLYMIDCENTURY: 7.5,
    TemporalLevel.DECADE: 8,
    TemporalLevel.EARLYMIDDECADE: 8.5,
    TemporalLevel.YEAR: 9,
    TemporalLevel.SEASON: 9.5,
    TemporalLevel.EARLYMIDYEAR: 9.5,
    TemporalLevel.MONTH: 10,
    TemporalLevel.DAY: 11,
}

LEVEL_TO_PRECISION = {
    level.name: precision
    for level, precision in PRECISION.items()
}

PRECISION_TO_RANK = {
    precision: rank
    for rank, precision in enumerate(sorted(PRECISION.values()))
}

LEVEL_FIELDS = {
    TemporalLevel.CENTURY: {
        "century",
    },
    TemporalLevel.EARLYMIDCENTURY: {
        "century",
        "century_modifier",
    },
    TemporalLevel.DECADE: {
        "century",
        "century_modifier",
        "decade",
    },
    TemporalLevel.EARLYMIDDECADE: {
        "century",
        "century_modifier",
        "decade",
        "decade_modifier",
    },
    TemporalLevel.YEAR: {
        "century",
        "century_modifier",
        "decade",
        "decade_modifier",
        "year",
    },
    TemporalLevel.SEASON: {
        "century",
        "century_modifier",
        "decade",
        "decade_modifier",
        "year",
        "season",
    },
    TemporalLevel.EARLYMIDYEAR: {
        "century",
        "century_modifier",
        "decade",
        "decade_modifier",
        "year",
        "year_modifier",
    },
    TemporalLevel.MONTH: {
        "century",
        "century_modifier",
        "decade",
        "decade_modifier",
        "year",
        "month",
        "season",
        "year_modifier",
    },
    TemporalLevel.DAY: {
        "century",
        "century_modifier",
        "decade",
        "decade_modifier",
        "year",
        "month",
        "day",
        "season",
        "year_modifier",
    },
}

LEVEL_EVAL_FIELDS = {
    TemporalLevel.CENTURY: {
        "century",
    },
    TemporalLevel.EARLYMIDCENTURY: {
        "century",
        "century_modifier",
    },
    TemporalLevel.DECADE: {
        "decade",
    },
    TemporalLevel.EARLYMIDDECADE: {
        "decade",
        "decade_modifier",
    },
    TemporalLevel.YEAR: {
        "year",
    },
    TemporalLevel.SEASON: {
        "year",
        "season",
    },
    TemporalLevel.EARLYMIDYEAR: {
        "year",
        "year_modifier",
    },
    TemporalLevel.MONTH: {
        "year",
        "month",
    },
    TemporalLevel.DAY: {
        "year",
        "month",
        "day",
    },
}