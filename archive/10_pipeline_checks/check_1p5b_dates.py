import csv
from collections import Counter

import pandas as pd

from pipeline import config as C
from pipeline.data import load
from pipeline.generate import generate, load_model
from pipeline.score import grade, prompt_for

items = {it.qid: it for it in load("dates", "test")}
new = pd.read_csv("results/sep11/1p5b/preds/base__dates_test__deploy.csv", dtype={"qid": str},
                  keep_default_na=False)
old = {r["qid"]: r for r in csv.DictReader(open("phase2_results/transfer_1p5b_base_on_dates.csv"))}

shared = [q for q in new.qid if q in old]
same_raw = sum(1 for q, raw in zip(new.qid, new.raw) if q in old and raw == old[q]["raw"])
old_kind = {q: grade(items[q], old[q]["raw"])["kind"] for q in shared}
print(f"A. Jul 29 1.5B base on dates vs today: {len(shared)} shared questions, identical raw answers {same_raw}")
print("   kinds, July answers graded by today's scorer:", dict(Counter(old_kind.values())))
print("   kinds today:", dict(Counter(new.kind)))
print("   transitions July -> today:", dict(Counter((old_kind[q], k) for q, k in zip(new.qid, new.kind) if q in old)))
examples = [(old[q]["raw"][:38], raw[:38]) for q, k, raw in zip(new.qid, new.kind, new.raw)
            if q in old and k == "abstained" and old_kind[q] == "answered"][:5]
print("   examples July answered -> today abstained:", examples)

qs = [items[q] for q in sorted(items)][:150]
model, tok = load_model(C.MODELS["1p5b"])
prompts = [prompt_for(it) for it in qs]
one = generate(model, tok, prompts, sample=False, batch_size=1)
many = generate(model, tok, prompts, sample=False, batch_size=32)


def outcome(it, raw):
    g = grade(it, raw)
    return g["kind"], g["correct"], g["level"]


diffs = sum(1 for it, a, b in zip(qs, one, many) if outcome(it, a[0]) != outcome(it, b[0]))
stored = dict(zip(new.qid, new.raw))
match_stored = sum(1 for it, b in zip(qs, many) if b[0] == stored[it.qid])
print(f"B. 1.5B base, first 150 dates test questions, bf16:")
print("   batch 1 kinds:", dict(Counter(outcome(it, o[0])[0] for it, o in zip(qs, one))))
print("   batch 32 kinds:", dict(Counter(outcome(it, o[0])[0] for it, o in zip(qs, many))))
print(f"   scorer-outcome differences batch 1 vs 32: {diffs}/150; batch 32 identical to stored preds: {match_stored}/150")
print("   July kinds on the same 150:", dict(Counter(old_kind[it.qid] for it in qs if it.qid in old_kind)))
