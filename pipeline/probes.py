"""Secondary probes through the same generation code. Neither touches locations val.

conflict  two synthetic sources agree only down to a level (src/conflict.py builds the
          cases and classifies behaviour).
entity    unseen entity relations, graded by the shared scorer.
"""
import pandas as pd

from src.conflict import build_cases, classify, make_conflict_prompt, summarize
from src.dates import TemporalLevel, level_info, load_dates_split

from . import config as C
from .data import load
from .generate import generate
from .score import grade, prompt_for


def run_conflict(model, tok, out_csv, batch_size: int, limit=None, log=None) -> dict:
    cases = build_cases(load_dates_split("test", historical_bce=True), seed=C.GLOBAL_SEED)
    if limit:
        keep = {c["qid"] for c in cases[:limit]}
        cases = [c for c in cases if c["qid"] in keep]
    outs = generate(model, tok, [make_conflict_prompt(c) for c in cases],
                    sample=False, batch_size=batch_size, log=log)
    rows = []
    for c, o in zip(cases, outs):
        row = classify(c, o[0])
        if not o[0].strip():
            # Same rule as the shared scorer: an empty completion is a format failure.
            row.update(idk=False, behaviour="fmt_fail", correct_strict=False,
                       signed_strict=-level_info(TemporalLevel.YEAR))
        row["raw"] = o[0]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return {"n": len(rows), "summary": summarize(rows)}


def run_entity(model, tok, out_csv, batch_size: int, limit=None, log=None) -> dict:
    items = load("entity", "test")
    if limit:
        items = items[:limit]
    outs = generate(model, tok, [prompt_for(it) for it in items],
                    sample=False, batch_size=batch_size, log=log)
    rows = []
    for it, o in zip(items, outs):
        g = grade(it, o[0])
        rows.append(dict(qid=it.qid, relation=it.relation, question=it.question, raw=o[0],
                         kind=g["kind"], text=g["text"], correct=g["correct"],
                         level=g["level"], score=g["score"]))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    def rates(d):
        return dict(n=len(d), correct=float(d.correct.mean()),
                    abstain=float((d.kind == "abstained").mean()),
                    wrong=float(((d.kind != "abstained") & ~d.correct).mean()),
                    fmt_fail=int((d.kind == "fmt_fail").sum()), score=float(d.score.mean()))
    return {"all": rates(df),
            "per_relation": {rel: rates(d) for rel, d in df.groupby("relation")}}
