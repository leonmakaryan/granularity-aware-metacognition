"""Transfer without further training: a dates-trained adapter on
the location test set, and the other way round.

A model that fails to transfer writes the other domain's marker, which the parsers
would read as an abstention and score as perfect caution. So answers are split
into answered, abstained and format failure, and a format failure counts as a
wrong answer.

    ADAPTER=dates_sft_7b_s0w2 DOMAIN=locations MODEL_NAME=Qwen/Qwen2.5-7B-Instruct python 6_transfer_joint/run_transfer.py
"""
import os
os.environ.setdefault("MODEL_DTYPE", "bfloat16")

import json
import time
from statistics import mean, pstdev

import pandas as pd
import torch

from src.config import (OUT_DIR, MODEL_NAME, MAX_COMPLETION_LEN, SEED,
                        LOCATION_RELATIONS)
from src.inference import load_tokenizer, load_base_model, load_trained_model
from src.eval import (metacognition, build_level_pool, signed_informativeness,
                      strict_match_level, output_level, level_mean_std)
from src.text import match_level
from src.data import load_split
from src.prompt import make_prompt, extract_answer, is_idk
from src.dates import (load_dates_split, make_date_prompt, extract_date_text,
                       credited_level, signed_date_info, level_info)
from src.dates import _PARSER

ADAPTER = os.environ.get("ADAPTER", "base")
DOMAIN = os.environ.get("DOMAIN", "locations")
SMOKE = os.environ.get("SMOKE") == "1"
MTAG = "7b" if "7B" in MODEL_NAME else "1p5b"
TAG = os.environ.get("TAG", f"{MTAG}_{ADAPTER}_on_{DOMAIN}")

OUT_CSV = OUT_DIR / f"transfer_{TAG}.csv"
OUT_JSON = OUT_DIR / f"transfer_{TAG}.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def generate_raw(model, tok, prompt_text):
    """Greedy, returns the FULL completion (not marker-extracted) so the caller
    can tell a missing marker from an abstention."""
    chat = tok.apply_chat_template([{"role": "user", "content": prompt_text}],
                                   tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_COMPLETION_LEN,
                             do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def classify(raw: str, extracted: str) -> str:
    """answered | abstained | fmt_fail -- the three-way split described up top."""
    if extracted and not is_idk(extracted):
        return "answered"
    if extracted and is_idk(extracted):
        return "abstained"
    # extraction empty: was it a real abstention, or the wrong marker entirely?
    return "abstained" if is_idk(raw) else "fmt_fail"


# ---------------------------------------------------------------- locations
def eval_locations(model, tok):
    # All six location relations (303 questions), not the P131-only default (61).
    test = load_split("test", LOCATION_RELATIONS)
    if SMOKE:
        test = test[:8]
    log(f"locations test: {len(test)} questions")
    pool = build_level_pool()

    rows = []
    for i, q in enumerate(test):
        raw = generate_raw(model, tok, make_prompt(q.question))
        ans = extract_answer(raw)
        kind = classify(raw, ans)
        # fmt_fail is a commit, not an abstention: idk=False so it is penalized.
        idk = (kind == "abstained")
        ml = match_level(ans, q.hierarchy) if kind == "answered" else None
        sml = strict_match_level(ans, q.hierarchy) if kind == "answered" else None
        rows.append(dict(qid=q.qid, question=q.question, raw=raw, answer=ans,
                         kind=kind, is_idk=idk, match_level=ml, strict_level=sml))
        if i % 50 == 0:
            log(f"  {i+1}/{len(test)}")

    n = len(rows)
    kinds = pd.Series([r["kind"] for r in rows]).value_counts().to_dict()
    fmt_fail = kinds.get("fmt_fail", 0)

    signed = [signed_informativeness(q, r["match_level"], r["is_idk"], r["answer"], pool)
              for q, r in zip(test, rows)]
    signed_str = [signed_informativeness(q, r["strict_level"], r["is_idk"], r["answer"], pool)
                  for q, r in zip(test, rows)]
    keep = [j for j, r in enumerate(rows) if r["kind"] != "fmt_fail"]
    out_lv = [output_level(q, r["match_level"], r["is_idk"], r["answer"], pool)
              for q, r in zip(test, rows)]
    lv_mean, lv_std = level_mean_std(out_lv)
    meta = metacognition([(not r["is_idk"], r["match_level"] is not None) for r in rows])

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    return dict(
        n=n, domain="locations",
        fmt_fail=fmt_fail, fmt_fail_rate=fmt_fail / n,
        answered=kinds.get("answered", 0), abstained=kinds.get("abstained", 0),
        acc=sum(1 for r in rows if r["match_level"] is not None) / n,
        acc_strict=sum(1 for r in rows if r["strict_level"] is not None) / n,
        idk=sum(1 for r in rows if r["is_idk"]) / n,
        wrong=sum(1 for r in rows if not r["is_idk"] and r["match_level"] is None) / n,
        info_signed=sum(signed) / n,
        info_signed_strict=sum(signed_str) / n,
        info_signed_parsed_only=(sum(signed[j] for j in keep) / len(keep)) if keep else None,
        out_level_mean=lv_mean, out_level_std=lv_std,
        d=meta["d_prime"],
    )


# -------------------------------------------------------------------- dates
def eval_dates(model, tok):
    test = load_dates_split("test")
    if SMOKE:
        test = test[:8]
    log(f"dates test: {len(test)} questions")

    rows, recs = [], []
    for i, q in enumerate(test):
        raw = generate_raw(model, tok, make_date_prompt(q.question))
        text = extract_date_text(raw)
        kind = classify(raw, text)
        idk = (kind == "abstained")
        v = _PARSER.first_most_specific(text) if kind == "answered" else None
        rows.append(dict(qid=q.qid, question=q.question, raw=raw, answer=text,
                         kind=kind, is_idk=idk, gold=q.gold.to_text(),
                         gold_level=q.gold.level.name,
                         parsed=v.to_text() if v else None,
                         parsed_level=v.level.name if v else None))
        recs.append((q, text, idk, v))
        if i % 50 == 0:
            log(f"  {i+1}/{len(test)}")

    n = len(rows)
    kinds = pd.Series([r["kind"] for r in rows]).value_counts().to_dict()
    fmt_fail = kinds.get("fmt_fail", 0)

    len_lv = [credited_level(v, q.gold, False) if not idk else None for q, _, idk, v in recs]
    str_lv = [credited_level(v, q.gold, True) if not idk else None for q, _, idk, v in recs]
    signed = [signed_date_info(v, idk, q.gold, False) for q, _, idk, v in recs]
    signed_str = [signed_date_info(v, idk, q.gold, True) for q, _, idk, v in recs]
    keep = [j for j, r in enumerate(rows) if r["kind"] != "fmt_fail"]
    prec = [v.precision for _, _, idk, v in recs if not idk and v is not None]
    meta = metacognition([(not idk, l is not None) for (_, _, idk, _), l in zip(recs, len_lv)])

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    return dict(
        n=n, domain="dates",
        fmt_fail=fmt_fail, fmt_fail_rate=fmt_fail / n,
        answered=kinds.get("answered", 0), abstained=kinds.get("abstained", 0),
        acc=sum(1 for l in len_lv if l is not None) / n,
        acc_strict=sum(1 for l in str_lv if l is not None) / n,
        idk=sum(1 for r in rows if r["is_idk"]) / n,
        wrong=sum(1 for l, r in zip(len_lv, rows) if l is None and not r["is_idk"]) / n,
        info_signed=sum(signed) / n,
        info_signed_strict=sum(signed_str) / n,
        info_signed_parsed_only=(sum(signed[j] for j in keep) / len(keep)) if keep else None,
        out_level_mean=mean(prec) if prec else None,
        out_level_std=pstdev(prec) if prec else None,
        d=meta["d_prime"],
    )


def main():
    torch.manual_seed(SEED)
    log(f"MODEL={MODEL_NAME} ADAPTER={ADAPTER} DOMAIN={DOMAIN}")
    tok = load_tokenizer()
    if ADAPTER == "base":
        model = load_base_model()
    else:
        model, tok = load_trained_model(ADAPTER)
    m = eval_locations(model, tok) if DOMAIN == "locations" else eval_dates(model, tok)

    with open(OUT_JSON, "w") as f:
        json.dump({"model": MODEL_NAME, "adapter": ADAPTER, "domain": DOMAIN,
                   "metrics": m}, f, indent=2)
    log(f"FMT_FAIL {m['fmt_fail']}/{m['n']} ({m['fmt_fail_rate']:.1%})  "
        f"answered={m['answered']} abstained={m['abstained']}")
    log(f"acc={m['acc']:.3f} acc_strict={m['acc_strict']:.3f} idk={m['idk']:.3f} "
        f"wrong={m['wrong']:.3f} signed={m['info_signed']:+.3f} "
        f"signed_strict={m['info_signed_strict']:+.3f} d'={m['d']}")
    log(f"wrote {OUT_CSV.name} / {OUT_JSON.name}")


if __name__ == "__main__":
    main()
