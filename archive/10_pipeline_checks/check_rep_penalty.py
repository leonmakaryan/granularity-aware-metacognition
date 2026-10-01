import csv
from collections import Counter

import pandas as pd
from transformers import GenerationConfig

from pipeline import config as C
from pipeline.data import load
from pipeline.generate import generate, load_model
from pipeline.score import grade, prompt_for

print("A. shipped generation settings")
shipped = {}
for tag, name in C.MODELS.items():
    g = GenerationConfig.from_pretrained(name)
    shipped[tag] = g
    print(f"   {tag}: repetition_penalty {g.repetition_penalty}, top_k {g.top_k}, top_p {g.top_p}, "
          f"temperature {g.temperature}")
print(f"   pipeline config uses repetition_penalty {C.REPETITION_PENALTY}, top_k {C.TOP_K} for every model")

print("B. 1.5B base on locations test, July vs today (both graded by today's scorer)")
items = {it.qid: it for it in load("locations", "test")}
try:
    old = {r["qid"]: r["raw"] for r in csv.DictReader(open("phase2_results/transfer_1p5b_base_on_locations.csv"))}
    new = pd.read_csv("results/sep11/1p5b/preds/base__locations_test__deploy.csv", dtype={"qid": str}, keep_default_na=False)
    print("   July kinds:", dict(Counter(grade(items[q], old[q])["kind"] for q in new.qid if q in old)))
    print("   today kinds:", dict(Counter(new.kind)))
except FileNotFoundError as e:
    print("   missing file:", e)

print("C. 1.5B base, first 150 dates test questions, batch 32, with the shipped penalty")
ditems = load("dates", "test")[:150]
today = pd.read_csv("results/sep11/1p5b/preds/base__dates_test__deploy.csv", dtype={"qid": str}, keep_default_na=False)
today_kind = dict(zip(today.qid, today.kind))
model, tok = load_model(C.MODELS["1p5b"])
prompts = [prompt_for(it) for it in ditems]
for penalty in (shipped["1p5b"].repetition_penalty, C.REPETITION_PENALTY):
    C.REPETITION_PENALTY = penalty
    outs = generate(model, tok, prompts, sample=False, batch_size=32)
    kinds = Counter(grade(it, o[0])["kind"] for it, o in zip(ditems, outs))
    print(f"   repetition_penalty {penalty}: {dict(kinds)}")
print("   today's stored run (1.05) on the same 150:", dict(Counter(today_kind[it.qid] for it in ditems)))
