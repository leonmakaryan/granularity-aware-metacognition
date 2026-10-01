"""Joint SFT on dates and locations together, evaluated on both
test sets. The larger domain is subsampled to MIX_RATIO times the smaller one, so
one domain does not dominate. The two domains have different prompts and markers.

    SMOKE=1 python 6_transfer_joint/run_joint_sft.py      python 6_transfer_joint/run_joint_sft.py [seed]
"""
import os
os.environ.setdefault("MODEL_DTYPE", "bfloat16")

import json
import random
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset

from src.config import (OUT_DIR, CKPT_DIR, MODEL_NAME, MODEL_DTYPE, LORA_R,
                        LORA_ALPHA, BATCH_SIZE, GRAD_ACCUM, SFT_LR, SFT_EPOCHS,
                        SFT_MAX_LEN, SEED)
from src.sft import build_sft_records
from src.dates import load_dates_split, derive_dates_targets, build_dates_sft_records

SMOKE = os.environ.get("SMOKE") == "1"
RUN_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
MIX_RATIO = float(os.environ.get("MIX_RATIO", "1.0"))
DEP_WEIGHT = int(os.environ.get("DEP_WEIGHT", "2"))

# Caches follow the model size, mirroring run_dates_sft.py's MTAG pattern. The 7B
# locations cache only exists after run_cap7b_loc.sh rebuilds it (see the module
# docstring); both are overridable so a rebuild under a different name still works.
MTAG = "7b_" if "7B" in MODEL_NAME else ""
LOC_CAP = Path(os.environ.get(
    "LOC_CAP", OUT_DIR / ("capability_loc_7b.json" if MTAG else "capability_loc_v2.json")))
DATE_CAP = Path(os.environ.get(
    "DATE_CAP", OUT_DIR / f"dates_{MTAG}capability.json"))
_tag = f"{MTAG}s{RUN_SEED}w{DEP_WEIGHT}"
# ADAPTER_NAME avoids overwriting an existing adapter of the same derived name.
ADAPTER = os.environ.get(
    "ADAPTER_NAME", "joint_smoke" if SMOKE else f"joint_{_tag}")
OUT_JSON = OUT_DIR / f"{ADAPTER}.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_joint(seed: int):
    """-> (records, stats). Each half uses its own domain's tuned recipe."""
    with open(LOC_CAP) as f:
        loc_cap = json.load(f)
    with open(DATE_CAP) as f:
        date_cap = json.load(f)
    if SMOKE:
        loc_cap["questions"] = dict(list(loc_cap["questions"].items())[:40])
        date_cap["questions"] = dict(list(date_cap["questions"].items())[:40])

    log("building locations half (self-distilled, balanced)")
    loc = build_sft_records(loc_cap, balance=True, seed=seed,
                            self_distill=True, dep_weight=DEP_WEIGHT)

    log("building dates half (self-distilled, balanced)")
    derive_dates_targets(date_cap)
    golds = {q.qid: q.gold for q in load_dates_split("train")}
    dat = build_dates_sft_records(date_cap, golds, seed=seed, dep_weight=DEP_WEIGHT)

    n_loc_raw, n_dat_raw = len(loc), len(dat)
    rng = random.Random(seed)
    # Subsample the larger half so neither domain dominates the gradient.
    if n_loc_raw > n_dat_raw * MIX_RATIO:
        loc = rng.sample(loc, int(n_dat_raw * MIX_RATIO))
    elif n_dat_raw > n_loc_raw * MIX_RATIO:
        dat = rng.sample(dat, int(n_loc_raw * MIX_RATIO))

    records = loc + dat
    rng.shuffle(records)
    stats = dict(loc_raw=n_loc_raw, dates_raw=n_dat_raw,
                 loc_used=len(loc), dates_used=len(dat),
                 total=len(records), mix_ratio=MIX_RATIO, dep_weight=DEP_WEIGHT)
    log(f"JOINT SET: locations {n_loc_raw}->{len(loc)}  dates {n_dat_raw}->{len(dat)}  "
        f"total={len(records)}")
    return records, stats


def main():
    torch.manual_seed(RUN_SEED)
    log(f"MODEL={MODEL_NAME} dtype={MODEL_DTYPE} adapter={ADAPTER} seed={RUN_SEED}")
    records, stats = build_joint(RUN_SEED)
    if SMOKE:
        records = records[:120]

    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig
    lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"],
                          lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    kwargs = dict(output_dir=str(CKPT_DIR / "joint_run"), learning_rate=SFT_LR,
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

    with open(OUT_JSON, "w") as f:
        json.dump({"model": MODEL_NAME, "adapter": ADAPTER, "seed": RUN_SEED,
                   "dataset": stats}, f, indent=2)
    log("DONE training. Eval both domains with:")
    log(f"  ADAPTER={ADAPTER} DOMAIN=locations MODEL_DTYPE=float16 python run_transfer.py")
    log(f"  ADAPTER={ADAPTER} DOMAIN=dates     MODEL_DTYPE=bfloat16 python run_transfer.py")


if __name__ == "__main__":
    main()
