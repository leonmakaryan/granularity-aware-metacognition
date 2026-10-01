"""GRANOLA-EQ questions with their answer hierarchies.

`hierarchy` runs from the most specific answer to the coarsest, and `levels[i]` is
the dataset's normalized granularity of `hierarchy[i]`.
"""

from dataclasses import dataclass

from datasets import load_dataset

from .config import DATASET, RELATIONS


@dataclass
class Question:
    qid: int
    question: str
    hierarchy: list[str]   # most specific -> coarsest
    levels: list[float]    # normalized_levels, parallel to hierarchy
    relation: str = ""     # source relation (P131, P19, ...), for per-relation eval


def load_split(split: str, relations: list[str] | None = None) -> list[Question]:
    rels = set(relations) if relations is not None else set(RELATIONS)
    ds = load_dataset(DATASET, split=split)
    qs = []
    for row in ds:
        if row["relation"] not in rels:
            continue
        hier = list(row["granola_answers"])
        levs = list(row["normalized_levels"])
        qs.append(Question(qid=row["id"], question=row["question"],
                           hierarchy=hier, levels=levs, relation=row["relation"]))
    return qs
