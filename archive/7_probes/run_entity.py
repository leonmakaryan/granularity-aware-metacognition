"""The base and trained models on relations without a place or date hierarchy
(author, creator, owner, ...), without training on them.

    ADAPTER=base MODEL_NAME=Qwen/Qwen2.5-7B-Instruct python 7_probes/run_entity.py
"""
import os

os.environ.setdefault("MODEL_DTYPE", "float16")

import json
import time

import pandas as pd
import torch

from src.config import OUT_DIR, MODEL_NAME, MAX_COMPLETION_LEN, SEED
from src.data import load_split
from src.inference import load_tokenizer, load_base_model, load_trained_model
from src.entity import (ENTITY_RELATIONS, make_entity_prompt,
                        extract_entity_answer, entity_match_level)
from src.eval import metacognition, informativeness
from src.prompt import is_idk, finest_element

ADAPTER = os.environ.get("ADAPTER", "base")
SMOKE = bool(os.environ.get("SMOKE"))
MTAG = "7b" if "7B" in MODEL_NAME else "1p5b"
TAG = os.environ.get("TAG", f"{MTAG}_{ADAPTER}" + ("_smoke" if SMOKE else ""))
RELS = os.environ.get("ENTITY_RELS", "").split(",") if os.environ.get("ENTITY_RELS") else ENTITY_RELATIONS

OUT_CSV = OUT_DIR / f"entity_{TAG}.csv"
OUT_JSON = OUT_DIR / f"entity_{TAG}.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def generate_raw(model, tok, prompt_text):
    chat = tok.apply_chat_template([{"role": "user", "content": prompt_text}],
                                   tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_COMPLETION_LEN,
                             do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def classify(raw: str, extracted: str) -> str:
    if extracted and not is_idk(extracted):
        return "answered"
    if extracted and is_idk(extracted):
        return "abstained"
    return "abstained" if is_idk(raw) else "fmt_fail"


def main():
    torch.manual_seed(SEED)
    log(f"MODEL={MODEL_NAME} ADAPTER={ADAPTER} relations={RELS}")

    test = load_split("test", RELS)
    if SMOKE:
        test = test[:12]
    log(f"entity test: {len(test)} questions")

    tok = load_tokenizer()
    if ADAPTER == "base":
        model = load_base_model()
    else:
        model, tok = load_trained_model(ADAPTER)

    rows = []
    for i, q in enumerate(test):
        raw = generate_raw(model, tok, make_entity_prompt(q.question))
        ans = extract_entity_answer(raw)
        kind = classify(raw, ans)
        idk = (kind == "abstained")
        ml = entity_match_level(ans, q.hierarchy) if kind == "answered" else None
        sl = (entity_match_level(finest_element(ans), q.hierarchy)
              if kind == "answered" else None)
        rows.append(dict(qid=q.qid, relation=q.relation, question=q.question,
                         raw=raw, answer=ans, kind=kind, is_idk=idk,
                         match_level=ml, strict_level=sl,
                         gold_fine=q.hierarchy[0] if q.hierarchy else "",
                         gold_coarse=q.hierarchy[-1] if q.hierarchy else ""))
        if i % 50 == 0:
            log(f"  {i+1}/{len(test)}")

    n = len(rows)
    kinds = pd.Series([r["kind"] for r in rows]).value_counts().to_dict()
    fmt_fail = kinds.get("fmt_fail", 0)

    def signed(level_key):
        tot = 0.0
        for q, r in zip(test, rows):
            if r["is_idk"]:
                continue
            L = r[level_key]
            if L is not None and L < len(q.levels):
                tot += informativeness(q.levels[L])
            else:
                # Wrong: penalize at the coarsest level, the most conservative
                # estimate available without a level pool for this family.
                tot -= informativeness(q.levels[-1]) if q.levels else 1.0
        return tot / n

    meta = metacognition([(not r["is_idk"], r["match_level"] is not None) for r in rows])
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    per_rel = {}
    for rel in sorted({r["relation"] for r in rows}):
        sel = [r for r in rows if r["relation"] == rel]
        per_rel[rel] = {
            "n": len(sel),
            "acc": round(sum(1 for r in sel if r["match_level"] is not None) / len(sel), 4),
            "idk": round(sum(1 for r in sel if r["is_idk"]) / len(sel), 4),
            "fmt_fail": round(sum(1 for r in sel if r["kind"] == "fmt_fail") / len(sel), 4),
        }

    m = dict(
        n=n, fmt_fail=fmt_fail, fmt_fail_rate=fmt_fail / n,
        answered=kinds.get("answered", 0), abstained=kinds.get("abstained", 0),
        acc=sum(1 for r in rows if r["match_level"] is not None) / n,
        acc_strict=sum(1 for r in rows if r["strict_level"] is not None) / n,
        idk=sum(1 for r in rows if r["is_idk"]) / n,
        wrong=sum(1 for r in rows if not r["is_idk"] and r["match_level"] is None) / n,
        info_signed=signed("match_level"),
        info_signed_strict=signed("strict_level"),
        d=meta["d_prime"], per_relation=per_rel,
    )

    with open(OUT_JSON, "w") as f:
        json.dump({"model": MODEL_NAME, "adapter": ADAPTER,
                   "relations": RELS, "metrics": m}, f, indent=2)
    log(f"FMT_FAIL {fmt_fail}/{n} ({m['fmt_fail_rate']:.1%})  "
        f"answered={m['answered']} abstained={m['abstained']}")
    log(f"acc={m['acc']:.3f} acc_strict={m['acc_strict']:.3f} idk={m['idk']:.3f} "
        f"wrong={m['wrong']:.3f} signed={m['info_signed']:+.3f} "
        f"signed_strict={m['info_signed_strict']:+.3f} "
        f"d'={'n/a' if m['d'] is None else format(m['d'], '.3f')}")
    for rel, v in per_rel.items():
        log(f"    {rel}: n={v['n']:3d} acc={v['acc']:.3f} idk={v['idk']:.3f} fmt={v['fmt_fail']:.3f}")
    log(f"wrote {OUT_CSV.name} / {OUT_JSON.name}")


if __name__ == "__main__":
    main()
