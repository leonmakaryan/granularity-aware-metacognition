"""One question type for every evaluation set."""
import hashlib
import os
from dataclasses import dataclass, field

from src.config import LOCATION_RELATIONS
from src.data import load_split
from src.dates import load_dates_split
from src.entity import ENTITY_RELATIONS


@dataclass
class Item:
    domain: str                 # locations | dates | entity
    qid: str
    question: str
    unit: str                   # bootstrap resampling unit
    hierarchy: list = field(default_factory=list)
    levels: list = field(default_factory=list)
    gold: object = None         # TemporalValue, dates only
    relation: str = ""


def load(domain: str, split: str, historical_bce: bool = True) -> list[Item]:
    """Items in a fixed order (sorted qid), so batches and seeds are reproducible.

    Dates use historical BCE year numbering (see src/dates._gold_value).
    Locations `val` is the confirmation set, used once at the end, so reading it
    needs CONFIRM=1."""
    if split == "val" and os.environ.get("CONFIRM") != "1":
        raise RuntimeError("locations val is the confirmation set; it is read only by "
                           "the frozen confirmation phase (CONFIRM=1)")
    if domain == "locations":
        items = [Item("locations", str(q.qid), q.question, str(q.qid),
                      q.hierarchy, q.levels, relation=q.relation)
                 for q in load_split(split, LOCATION_RELATIONS)]
    elif domain == "dates":
        if split not in ("train", "test"):
            raise ValueError("dates has only train and test")
        # A person's birth and death questions share one resampling unit.
        items = [Item("dates", q.qid, q.question, q.qid.rsplit("_", 1)[0], gold=q.gold)
                 for q in load_dates_split(split, historical_bce=historical_bce)]
    elif domain == "entity":
        if split != "test":
            raise ValueError("the entity probe uses the test split only")
        items = [Item("entity", str(q.qid), q.question, str(q.qid),
                      q.hierarchy, q.levels, relation=q.relation)
                 for q in load_split("test", ENTITY_RELATIONS)]
    else:
        raise ValueError(domain)
    return sorted(items, key=lambda it: it.qid)


def holdout(items: list[Item], frac: float, seed: int) -> tuple[list[Item], list[Item]]:
    """Deterministic (fit, held-out) split by resampling unit, for the pilot only."""
    def held(unit: str) -> bool:
        h = int(hashlib.sha256(f"{seed}|{unit}".encode()).hexdigest()[:8], 16)
        return h / 0xFFFFFFFF < frac
    return ([it for it in items if not held(it.unit)],
            [it for it in items if held(it.unit)])
