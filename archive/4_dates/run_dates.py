"""Dates: the base model on the test set, then the sample cache on the training set
(8 samples and one greedy answer per question). MODEL_NAME selects the model.

    SMOKE=1 python 4_dates/run_dates.py
"""
import os
os.environ.setdefault("MODEL_DTYPE", "bfloat16")

import json
import time
from statistics import mean, pstdev

import pandas as pd
import torch

from src.config import OUT_DIR, MODEL_NAME, MAX_COMPLETION_LEN, SEED
from src.inference import load_tokenizer, load_base_model
from src.eval import metacognition
from src.dates import (load_dates_split, make_date_prompt, parse_answer,
                       credited_level, signed_date_info, level_info, DATE_LADDER)

SMOKE = os.environ.get("SMOKE") == "1"
TAG = "7b_" if "7B" in MODEL_NAME else ""
K = 8

EVAL_CSV = OUT_DIR / f"dates_{TAG}base_eval.csv"
EVAL_JSON = OUT_DIR / f"dates_{TAG}base_eval.json"
CAP_PATH = OUT_DIR / f"dates_{TAG}capability.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def generate(model, tok, question, n=1, greedy=True):
    chat = tok.apply_chat_template(
        [{"role": "user", "content": make_date_prompt(question)}],
        tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    kwargs = dict(max_new_tokens=MAX_COMPLETION_LEN, pad_token_id=tok.pad_token_id)
    if greedy:
        kwargs["do_sample"] = False
    else:
        kwargs.update(do_sample=True, temperature=0.7, top_p=0.95,
                      num_return_sequences=n)
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]


def eval_metrics(recs):
    """recs: (q, marker_text, idk, value). Returns the dates metric block."""
    n = len(recs)
    len_lvls = [credited_level(v, q.gold, strict=False) if not idk else None
                for q, _, idk, v in recs]
    str_lvls = [credited_level(v, q.gold, strict=True) if not idk else None
                for q, _, idk, v in recs]
    signed_len = [signed_date_info(v, idk, q.gold, strict=False) for q, _, idk, v in recs]
    signed_str = [signed_date_info(v, idk, q.gold, strict=True) for q, _, idk, v in recs]
    committed = [r for r in recs if not r[2]]
    unparseable = sum(1 for _, _, idk, v in recs if not idk and v is None)
    out_prec = [v.precision for _, _, idk, v in recs if not idk and v is not None]
    mc = metacognition([(not idk, l is not None) for (_, _, idk, _), l in zip(recs, len_lvls)])
    return dict(
        n=n,
        acc=sum(1 for l in len_lvls if l is not None) / n,
        acc_strict=sum(1 for l in str_lvls if l is not None) / n,
        info=sum(level_info(l) for l in len_lvls if l is not None) / n,
        info_signed=sum(signed_len) / n,
        info_signed_strict=sum(signed_str) / n,
        idk=(n - len(committed)) / n,
        wrong=sum(1 for l, r in zip(len_lvls, recs) if l is None and not r[2]) / n,
        unparseable=unparseable,
        sel=(sum(1 for l in len_lvls if l is not None) / len(committed)) if committed else 0.0,
        d=mc["d_prime"],
        out_precision_mean=mean(out_prec) if out_prec else None,
        out_precision_std=pstdev(out_prec) if out_prec else None,
    )


def main():
    torch.manual_seed(SEED)
    test = load_dates_split("test")
    train = load_dates_split("train")
    if SMOKE:
        test, train = test[:8], train[:8]
        print("*** SMOKE: 8 questions/stage ***", flush=True)
    log(f"MODEL={MODEL_NAME}  dates train={len(train)} test={len(test)}")
    tok = load_tokenizer()
    model = load_base_model()

    # ---- stage 1: base + IDK prompt on dates test ----
    log("stage 1: base eval on dates test")
    recs = []
    for i, q in enumerate(test):
        text, idk, v = parse_answer(generate(model, tok, q.question)[0])
        recs.append((q, text, idk, v))
        if i % 50 == 0:
            log(f"  eval {i+1}/{len(test)}")
    pd.DataFrame([{
        "qid": q.qid, "question": q.question, "gold": q.gold.to_text(),
        "gold_level": q.gold.level.name, "answer": text, "is_idk": idk,
        "parsed": v.to_text() if v else None,
        "parsed_level": v.level.name if v else None,
        "lenient_level": (l.name if (l := credited_level(v, q.gold, False)) else None) if not idk else None,
        "strict_level": (l.name if (l := credited_level(v, q.gold, True)) else None) if not idk else None,
    } for q, text, idk, v in recs]).to_csv(EVAL_CSV, index=False)
    m = eval_metrics(recs)
    with open(EVAL_JSON, "w") as f:
        json.dump({"model": MODEL_NAME, "base": m}, f, indent=2)
    log(f"BASE DATES: acc={m['acc']:.3f} acc_strict={m['acc_strict']:.3f} "
        f"info={m['info']:.3f} signed={m['info_signed']:+.3f} "
        f"signed_strict={m['info_signed_strict']:+.3f} idk={m['idk']:.3f} "
        f"wrong={m['wrong']:.3f} unparseable={m['unparseable']} d'={m['d']}")

    # ---- stage 2: capability cache on dates train ----
    log("stage 2: dates capability cache (k samples + greedy per question)")
    out = {"model": MODEL_NAME, "k": K, "questions": {}}
    for i, q in enumerate(train):
        samples = generate(model, tok, q.question, n=K, greedy=False)
        greedy = generate(model, tok, q.question)[0]
        srecs = []
        for s in samples:
            text, idk, v = parse_answer(s)
            lvl = credited_level(v, q.gold, strict=False) if not idk else None
            srecs.append({"text": text, "idk": idk,
                          "parsed_level": v.level.name if v else None,
                          "lenient_level": lvl.name if lvl else None})
        gtext, gidk, gv = parse_answer(greedy)
        glvl = credited_level(gv, q.gold, strict=False) if not gidk else None
        out["questions"][q.qid] = {
            "question": q.question, "kind": q.kind,
            "gold": q.gold.to_text(), "gold_level": q.gold.level.name,
            "samples": srecs,
            "greedy": {"text": gtext, "idk": gidk,
                       "parsed_level": gv.level.name if gv else None,
                       "lenient_level": glvl.name if glvl else None},
        }
        if i % 50 == 0:
            log(f"  cap {i+1}/{len(train)}")
        if i % 200 == 0:
            with open(CAP_PATH, "w") as f:
                json.dump(out, f)
    with open(CAP_PATH, "w") as f:
        json.dump(out, f, indent=1)
    log(f"saved {CAP_PATH}")
    log("DONE")


if __name__ == "__main__":
    main()
