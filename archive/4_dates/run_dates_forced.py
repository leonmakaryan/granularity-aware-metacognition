"""Control for the dates SFT: the base model told to answer at one fixed level ("Give
the year only."). If this matched the SFT, training would add nothing over a prompt.
It did not.
"""
import os
os.environ.setdefault("MODEL_DTYPE", "bfloat16")

import json
import time

import pandas as pd
import torch

from src.config import OUT_DIR, MODEL_NAME, MAX_COMPLETION_LEN, SEED
from src.inference import load_tokenizer, load_base_model
from src.dates import (load_dates_split, parse_answer, credited_level,
                       DATE_DIRECT_TEMPLATE, DATE_FORCE_INSTRUCTION)
from ugc.parser.temporal_enums import TemporalLevel
from run_dates import eval_metrics

SMOKE = os.environ.get("SMOKE") == "1"
LEVELS = [TemporalLevel.YEAR, TemporalLevel.DECADE, TemporalLevel.CENTURY]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    torch.manual_seed(SEED)
    test = load_dates_split("test")
    if SMOKE:
        test = test[:8]
    log(f"MODEL={MODEL_NAME} forced-level dates baselines on {len(test)} Qs")
    tok = load_tokenizer()
    model = load_base_model()
    results = {}
    for lvl in LEVELS:
        tag = lvl.name.lower()
        log(f"=== base + forced {tag} ===")
        recs = []
        for i, q in enumerate(test):
            chat = tok.apply_chat_template(
                [{"role": "user", "content": DATE_DIRECT_TEMPLATE.format(
                    instruction=DATE_FORCE_INSTRUCTION[lvl], q=q.question)}],
                tokenize=False, add_generation_prompt=True)
            inputs = tok(chat, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_COMPLETION_LEN,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            comp = tok.decode(out[0, inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)
            text, idk, v = parse_answer(comp)
            recs.append((q, text, idk, v))
            if i % 100 == 0:
                log(f"  {tag} {i+1}/{len(test)}")
        if not SMOKE:
            pd.DataFrame([{
                "qid": q.qid, "answer": text, "is_idk": idk,
                "parsed": v.to_text() if v else None,
                "parsed_level": v.level.name if v else None,
                "lenient_level": (l.name if (l := credited_level(v, q.gold, False)) else None) if not idk else None,
                "strict_level": (l.name if (l := credited_level(v, q.gold, True)) else None) if not idk else None,
            } for q, text, idk, v in recs]).to_csv(
                OUT_DIR / f"dates_base_forced_{tag}.csv", index=False)
        m = eval_metrics(recs)
        results[tag] = m
        log(f"forced-{tag}: acc={m['acc']:.3f} acc_strict={m['acc_strict']:.3f} "
            f"signed={m['info_signed']:+.3f} signed_strict={m['info_signed_strict']:+.3f} "
            f"idk={m['idk']:.3f} wrong={m['wrong']:.3f} unparseable={m['unparseable']}")
    if not SMOKE:
        with open(OUT_DIR / "dates_base_forced.json", "w") as f:
            json.dump({"model": MODEL_NAME, **results}, f, indent=2)
    log("DONE")


if __name__ == "__main__":
    main()
