"""Informativeness of place answers, and the estimated level of a wrong place answer."""

from .config import INFO_FINEST_NL, INFO_PRIOR_NL, LOCATION_RELATIONS
from .data import Question, load_split
from .prompt import finest_element
from .text import normalize


def informativeness(norm_level: float | None) -> float:
    """Information of a correct answer at this normalized level: 1 at the finest
    level, less for coarser ones, 0 for None (see config.INFO_*)."""
    if norm_level is None:
        return 0.0
    info = (INFO_PRIOR_NL - norm_level) / (INFO_PRIOR_NL - INFO_FINEST_NL)
    return max(0.0, info)


def build_level_pool(qs: list[Question] | None = None) -> dict[str, float]:
    """Map each place name in the dataset to its mean normalized level.

    A wrong answer matches nothing in its own hierarchy, so its level is looked up
    here. Levels are relative to each hierarchy, so the mean over all hierarchies
    a place appears in gives its usual granularity."""
    if qs is None:
        qs = load_split("train", LOCATION_RELATIONS) + load_split("test", LOCATION_RELATIONS)
    sums: dict[str, list[float]] = {}
    for q in qs:
        for ent, lv in zip(q.hierarchy, q.levels):
            key = normalize(ent)
            if key:
                sums.setdefault(key, []).append(lv)
    return {k: sum(v) / len(v) for k, v in sums.items()}


# Level of a wrong answer that appears nowhere in the dataset: the middle of the scale.
WRONG_FALLBACK_NL = 2.5


def answer_norm_level(answer: str, pool: dict[str, float],
                      fallback: float = WRONG_FALLBACK_NL) -> float:
    """Estimated normalized level of an answer, right or wrong.

    Looks up the most specific part of the answer, then the whole answer; for a
    sentence, takes the finest place named in it; otherwise returns `fallback`."""
    for cand in (finest_element(answer), answer):
        lv = pool.get(normalize(cand))
        if lv is not None:
            return lv
    na = f" {normalize(answer)} "
    if na.strip():
        hits = [lv for key, lv in pool.items() if f" {key} " in na]
        if hits:
            return min(hits)
    return fallback
