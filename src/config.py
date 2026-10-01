"""Dataset and scoring settings shared by the domain helpers in src/."""

import os
from pathlib import Path


def _load_env():
    """Read HF_TOKEN and similar from a .env file in the repository root, if present."""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


_load_env()

DATASET = "lukasellinger/granola-eq"
RELATION = "P131"

# The six location relations. They share one place hierarchy (town, region, country).
LOCATION_RELATIONS = ["P131", "P19", "P20", "P276", "P159", "P740"]
RELATIONS = [RELATION]   # default filter of load_split

# Informativeness of a correct place answer, following Lukas's measure for dates. His
# widths are in years; the dataset's normalized levels are already a log scale, so the
# measure is linear in the level:
#   info(level) = (INFO_PRIOR_NL - level) / (INFO_PRIOR_NL - INFO_FINEST_NL)
# The prior sits one step beyond the country (4.0): city 1.0, region 0.625, country 0.25.
INFO_FINEST_NL = 1.0
INFO_PRIOR_NL = 5.0

# Two scoring fixes, on by default: the exact-match-first place matcher
# (text.find_match_level_v3) and the date parse retry with "in " + answer.
# CORRECTED_SCORING=0 restores the older rules.
CORRECTED_SCORING = os.environ.get("CORRECTED_SCORING", "1") == "1"
