"""Dates SFT: targets from the sample cache, training records from the model's own
answers, LoRA training, and evaluation on the test set.

    SMOKE=1 python 4_dates/run_dates_sft.py      python 4_dates/run_dates_sft.py [seed]

DEP_WEIGHT repeats the deployment examples; it is part of the adapter name.
"""
import os
os.environ.setdefault("MODEL_DTYPE", "bfloat16")

import json
import sys
import time

import pandas as pd
import torch
from datasets import Dataset

from src.config import (OUT_DIR, CKPT_DIR, MODEL_NAME, MODEL_DTYPE, LORA_R,
                        LORA_ALPHA, BATCH_SIZE, GRAD_ACCUM, SFT_LR, SFT_EPOCHS,
                        SFT_MAX_LEN, SEED)
from src.inference import load_tokenizer, load_trained_model
from src.dates import (load_dates_split, derive_dates_targets,
                       build_dates_sft_records, parse_answer, credited_level)
from run_dates import generate, eval_metrics

SMOKE = os.environ.get("SMOKE") == "1"
RUN_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
DEP_WEIGHT = int(os.environ.get("DEP_WEIGHT", "1"))
# Model tag mirrors run_dates.py so the 7B run reads the 7B capability cache and
# writes 7B-tagged artifacts (no clash with the 1.5B files).
MTAG = "7b_" if "7B" in MODEL_NAME else ""
CAP_PATH = OUT_DIR / f"dates_{MTAG}capability.json"
_tag = f"{MTAG}s{RUN_SEED}" + (f"w{DEP_WEIGHT}" if DEP_WEIGHT != 1 else "")
ADAPTER = "dates_sft_smoke" if SMOKE else f"dates_sft_{_tag}"
EVAL_CSV = OUT_DIR / ("dates_sft_eval_smoke.csv" if SMOKE else f"dates_sft_eval_{_tag}.csv")
EVAL_JSON = OUT_DIR / ("dates_sft_eval_smoke.json" if SMOKE else f"dates_sft_eval_{_tag}.json")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log(f"MODEL={MODEL_NAME} dtype={MODEL_DTYPE} adapter={ADAPTER}")
    with open(CAP_PATH) as f:
        cap = json.load(f)
    if SMOKE:
        cap["questions"] = dict(list(cap["questions"].items())[:40])

    # ---- targets + training data (offline) ----
    derive_dates_targets(cap)
    golds = {q.qid: q.gold for q in load_dates_split("train")}
    records = build_dates_sft_records(cap, golds, seed=RUN_SEED, dep_weight=DEP_WEIGHT)
    if SMOKE:
        records = records[:80]

    # ---- train (same light recipe as the locations SFT) ----
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig
    lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"],
                          lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    kwargs = dict(output_dir=str(CKPT_DIR / "dates_sft_run"), learning_rate=SFT_LR,
                  per_device_train_batch_size=BATCH_SIZE,
                  gradient_accumulation_steps=GRAD_ACCUM,
                  num_train_epochs=SFT_EPOCHS, max_length=SFT_MAX_LEN,
                  completion_only_loss=True, packing=False, logging_steps=25,
                  save_strategy="no", bf16=(MODEL_DTYPE == "bfloat16"),
                  fp16=(MODEL_DTYPE != "bfloat16"), seed=RUN_SEED, report_to="none")
    if SMOKE:
        kwargs["max_steps"] = 20
    trainer = SFTTrainer(model=MODEL_NAME, args=SFTConfig(**kwargs),
                         train_dataset=Dataset.from_list(records),
                         peft_config=lora_cfg)
    trainer.train()
    trainer.save_model(str(CKPT_DIR / ADAPTER))
    del trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    log(f"saved {CKPT_DIR / ADAPTER}")

    # ---- eval on dates test ----
    test = load_dates_split("test")
    if SMOKE:
        test = test[:8]
    model, tok = load_trained_model(ADAPTER)
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
    # The base row of the same model (the untagged file is the 1.5B one).
    base_path = OUT_DIR / f"dates_{MTAG}base_eval.json"
    if not base_path.exists():
        base_path = OUT_DIR / "dates_base_eval.json"
    base = json.load(open(base_path))["base"] if base_path.exists() else None
    with open(EVAL_JSON, "w") as f:
        json.dump({"model": MODEL_NAME, "adapter": ADAPTER, "sft": m, "base": base}, f, indent=2)
    log(f"DATES SFT: acc={m['acc']:.3f} acc_strict={m['acc_strict']:.3f} "
        f"info={m['info']:.3f} signed={m['info_signed']:+.3f} "
        f"signed_strict={m['info_signed_strict']:+.3f} idk={m['idk']:.3f} "
        f"wrong={m['wrong']:.3f} unparseable={m['unparseable']} d'={m['d']}")
    if base:
        log(f"BASE      : acc={base['acc']:.3f} acc_strict={base['acc_strict']:.3f} "
            f"info={base['info']:.3f} signed={base['info_signed']:+.3f} "
            f"signed_strict={base['info_signed_strict']:+.3f} idk={base['idk']:.3f} "
            f"wrong={base['wrong']:.3f}")
    if not SMOKE:
        with open("dates_multiseed.jsonl", "a") as f:
            f.write(json.dumps(dict(seed=RUN_SEED, dep_weight=DEP_WEIGHT,
                                    adapter=ADAPTER, **m)) + "\n")
    log("DONE")


if __name__ == "__main__":
    main()
