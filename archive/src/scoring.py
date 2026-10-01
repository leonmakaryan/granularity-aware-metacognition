"""GranuScore (Lukas's granularity scorer, 0-100, higher is coarser), for evaluation.
It scores the most specific part of an answer ("Tokyo, Japan" -> "Tokyo"), and is
imported only when used, because it does not install on Python 3.13.
"""

from .prompt import finest_element

_SCORER = None


def _scorer():
    global _SCORER
    if _SCORER is None:
        from granuscore import GranuScore
        _SCORER = GranuScore()
        _SCORER("warmup")
    return _SCORER


def gscore(texts: list[str]) -> list[float]:
    """GranuScore of the most specific part of each answer; an empty answer scores 0."""
    if not texts:
        return []
    clean, empty = [], set()
    for i, t in enumerate(texts):
        f = finest_element(t) if t else ""
        if f and f.strip():
            clean.append(f)
        else:
            clean.append("placeholder")
            empty.add(i)
    raw = _scorer()(clean)
    return [0.0 if i in empty else float(raw[i]) for i in range(len(clean))]
