"""Check that the first probe files grade the same under the final scorer, so probe
seeds run on different days share one scorer.
"""
import pandas as pd
from src.conflict import build_cases, classify
from src.dates import load_dates_split
from pipeline import config as C
from pipeline.data import load
from pipeline.score import grade
items = {str(it.qid): it for it in load("entity", "test")}
cases = build_cases(load_dates_split("test", historical_bce=True), seed=C.GLOBAL_SEED)
for a in ("base", "locations_s0", "dates_s0", "joint_s0"):
    e = pd.read_csv(f"results/sep11/7b/probes/entity__{a}.csv", dtype={"qid": str}, keep_default_na=False)
    de = sum(abs(grade(items[r.qid], r.raw)["score"] - float(r.score)) > 1e-9 for r in e.itertuples())
    c = pd.read_csv(f"results/sep11/7b/probes/conflict__{a}.csv", keep_default_na=False)
    assert len(c) == len(cases)
    dc = 0
    for case, r in zip(cases, c.to_dict("records")):
        if not str(r["raw"]).strip():
            continue
        new = classify(case, r["raw"])
        dc += (new["behaviour"] != r["behaviour"]) or abs(float(new["signed_strict"]) - float(r["signed_strict"])) > 1e-9
    print(f"{a:13s} entity rows changed: {de}/{len(e)}   conflict rows changed: {dc}/{len(c)}")
