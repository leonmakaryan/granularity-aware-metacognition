"""Conflicting sources probe. Evaluation only; the cases and the scoring are in
src/conflict.py.

    ADAPTER=base MODEL_NAME=Qwen/Qwen2.5-7B-Instruct python 7_probes/run_conflict.py
"""
import os

os.environ.setdefault("MODEL_DTYPE", "bfloat16")

import json
import time

import pandas as pd
import torch

from src.config import OUT_DIR, MODEL_NAME, MAX_COMPLETION_LEN, SEED
from src.inference import load_tokenizer, load_base_model, load_trained_model
from src.dates import load_dates_split
from src.conflict import build_cases, make_conflict_prompt, classify, summarize

ADAPTER = os.environ.get("ADAPTER", "base")
SMOKE = bool(os.environ.get("SMOKE"))
MTAG = "7b" if "7B" in MODEL_NAME else "1p5b"
# Smoke output gets its own tag: a 12-case file sitting at the real path once made
# the sweep's skip-if-exists check treat the smoke as a finished run.
TAG = os.environ.get("TAG", f"{MTAG}_{ADAPTER}" + ("_smoke" if SMOKE else ""))
CASE_SEED = int(os.environ.get("CASE_SEED", "0"))

OUT_CSV = OUT_DIR / f"conflict_{TAG}.csv"
OUT_JSON = OUT_DIR / f"conflict_{TAG}.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def generate_raw(model, tok, prompt_text):
    """Greedy, full completion (not marker-extracted) so the scorer can tell a
    missing marker from an abstention."""
    chat = tok.apply_chat_template([{"role": "user", "content": prompt_text}],
                                   tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_COMPLETION_LEN,
                             do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def main():
    torch.manual_seed(SEED)
    log(f"MODEL={MODEL_NAME} ADAPTER={ADAPTER} case_seed={CASE_SEED}")

    test = load_dates_split("test")
    cases = build_cases(test, seed=CASE_SEED)
    if SMOKE:
        # Keep all three conditions represented in a smoke run.
        keep = {c["qid"] for c in cases[:12]}
        cases = [c for c in cases if c["qid"] in keep]
    log(f"cases: {len(cases)} ({len(cases)//3} questions x 3 conditions)")

    tok = load_tokenizer()
    if ADAPTER == "base":
        model = load_base_model()
    else:
        model, tok = load_trained_model(ADAPTER)

    rows = []
    for i, c in enumerate(cases):
        raw = generate_raw(model, tok, make_conflict_prompt(c))
        r = classify(c, raw)
        r["raw"] = raw
        rows.append(r)
        if i % 100 == 0:
            log(f"  {i+1}/{len(cases)}")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    summary = summarize(rows)

    fmt = sum(1 for r in rows if r["behaviour"] == "fmt_fail")
    log(f"FMT_FAIL {fmt}/{len(rows)} ({100*fmt/len(rows):.1f}%)")
    for k, v in summary.items():
        log(f"  {k:18s} n={v['n']:4d} signed={v['signed_strict']:+.3f} "
            f"acc={v['acc_strict']:.3f} {v['behaviour']}")

    with open(OUT_JSON, "w") as f:
        json.dump({"model": MODEL_NAME, "adapter": ADAPTER,
                   "case_seed": CASE_SEED, "n_cases": len(cases),
                   "fmt_fail": fmt, "summary": summary}, f, indent=2)
    log(f"wrote {OUT_CSV.name} / {OUT_JSON.name}")


if __name__ == "__main__":
    main()
