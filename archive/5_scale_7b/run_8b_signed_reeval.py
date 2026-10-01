"""Evaluate the 7B base, SFT and surgical models again, this time saving every answer,
so the signed informativeness can be computed. Same settings as the first run.

    MODEL_NAME=Qwen/Qwen2.5-7B-Instruct python 5_scale_7b/run_8b_signed_reeval.py
"""
import gc
import json
import os
import time

import pandas as pd
import torch

from src.config import LOCATION_RELATIONS, OUT_DIR, CKPT_DIR, MODEL_NAME
from src.data import load_split
from src.inference import load_tokenizer, load_base_model, greedy_answers
from src.prompt import is_idk
from src.text import find_match_level
from src.eval import (info_weighted_acc, metacognition, build_level_pool,
                      signed_informativeness, output_level, level_mean_std,
                      strict_match_level)

LOAD_4BIT = os.environ.get("LOAD_4BIT") == "1"
RESULTS = OUT_DIR / "results_8b_signed.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_base():
    if not LOAD_4BIT:
        return load_base_model()
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=qcfg, device_map="auto")
    model.eval()
    return model


def load_adapter(name):
    from peft import PeftModel
    base = load_base()
    model = PeftModel.from_pretrained(base, str(CKPT_DIR / name))
    model.eval()
    return model


def eval_model(model, tok, qs, pool, csv_name):
    answers = greedy_answers(model, tok, qs)
    recs = []
    for q, a in zip(qs, answers):
        idk = is_idk(a)
        ml = find_match_level(a, q.hierarchy) if (a and not idk) else None
        recs.append((q, a, idk, ml))
    pd.DataFrame([{"qid": q.qid, "question": q.question, "hierarchy": q.hierarchy,
                   "answer": a, "is_idk": idk, "match_level": ml}
                  for q, a, idk, ml in recs]).to_csv(OUT_DIR / csv_name, index=False)

    n = len(recs)
    nls = [q.levels[ml] if (ml is not None and ml < len(q.levels)) else None
           for q, _, _, ml in recs]
    signed = [signed_informativeness(q, ml, idk, a, pool) for q, a, idk, ml in recs]
    lvl_mean, lvl_std = level_mean_std(
        [output_level(q, ml, idk, a, pool) for q, a, idk, ml in recs])
    # B4 strict policy (primary claim only) next to the lenient columns.
    smls = [strict_match_level(a, q.hierarchy) if (a and not idk) else None
            for q, a, idk, _ in recs]
    snls = [q.levels[m] if (m is not None and m < len(q.levels)) else None
            for (q, _, _, _), m in zip(recs, smls)]
    ssigned = [signed_informativeness(q, m, idk, a, pool)
               for (q, a, idk, _), m in zip(recs, smls)]
    committed = sum(1 for _, _, idk, _ in recs if not idk)
    on_hier = sum(1 for _, _, _, ml in recs if ml is not None)
    m = dict(n=n, acc=on_hier / n, info=info_weighted_acc(nls),
             info_signed=sum(signed) / n,
             info_strict=info_weighted_acc(snls),
             info_signed_strict=sum(ssigned) / n,
             level_mean=lvl_mean, level_std=lvl_std,
             idk=sum(1 for _, _, idk, _ in recs if idk) / n,
             wrong=sum(1 for _, _, idk, ml in recs if (not idk) and ml is None) / n,
             sel=on_hier / committed if committed else 0.0,
             d=metacognition([(not idk, ml is not None) for _, _, idk, ml in recs])["d_prime"])
    byrel = {}
    for rel in sorted({q.relation for q, _, _, _ in recs if q.relation}):
        sub = [i for i, (q, _, _, _) in enumerate(recs) if q.relation == rel]
        byrel[rel] = dict(n=len(sub),
                          acc=sum(1 for i in sub if recs[i][3] is not None) / len(sub),
                          info=info_weighted_acc([nls[i] for i in sub]),
                          info_signed=sum(signed[i] for i in sub) / len(sub),
                          wrong=sum(1 for i in sub if (not recs[i][2]) and recs[i][3] is None) / len(sub))
    m["per_relation"] = byrel
    return m


def main():
    log(f"MODEL={MODEL_NAME}  4bit={LOAD_4BIT}")
    test_qs = load_split("test", LOCATION_RELATIONS)
    tok = load_tokenizer()
    pool = build_level_pool()
    results = {"model": MODEL_NAME, "load_4bit": LOAD_4BIT}

    stages = [("base", None, "eval_7b_base.csv"),
              ("sft", "sft_8b", "eval_7b_sft.csv"),
              ("surgical", "sft_surgical_8b", "eval_7b_surgical.csv")]
    for tag, adapter, csv_name in stages:
        log(f"eval {tag}" + (f" (adapter {adapter})" if adapter else ""))
        model = load_base() if adapter is None else load_adapter(adapter)
        results[tag] = eval_model(model, tok, test_qs, pool, csv_name)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        m = results[tag]
        log(f"{tag}: acc={m['acc']:.3f} info={m['info']:.3f} "
            f"info_signed={m['info_signed']:+.3f} signed_strict={m['info_signed_strict']:+.3f} "
            f"lvl={m['level_mean']:.2f}±{m['level_std']:.2f} "
            f"idk={m['idk']:.3f} wrong={m['wrong']:.3f} d'={m['d']:.3f}")
        with open(RESULTS, "w") as f:
            json.dump(results, f, indent=2)

    old = OUT_DIR / "results_8b.json"
    if old.exists() and not LOAD_4BIT:
        with open(old) as f:
            prev = json.load(f)
        print("\nsanity vs Jul 1 run (must match in fp16):")
        for tag in ("base", "sft", "surgical"):
            if tag in prev and tag in results:
                print(f"  {tag:9s} acc {prev[tag]['acc']:.3f} -> {results[tag]['acc']:.3f}   "
                      f"info {prev[tag]['info']:.3f} -> {results[tag]['info']:.3f}")
    print(f"\nresults -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
