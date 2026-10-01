"""The location experiments at 7B (Qwen2.5-7B-Instruct; the file names say 8b):
the base model, the capability curve, SFT, and the "surgical" target, which keeps
the base model's correct answers and abstains where it was confidently wrong.

    MODEL_NAME=Qwen/Qwen2.5-7B-Instruct python 5_scale_7b/run_8b.py
"""
import os
os.environ.setdefault("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

import gc
import json
import time

import torch
from datasets import Dataset

from src.config import (LOCATION_RELATIONS, OUT_DIR, CKPT_DIR, MODEL_NAME, LORA_R,
                        LORA_ALPHA, BATCH_SIZE, GRAD_ACCUM, SFT_LR, SFT_EPOCHS, SFT_MAX_LEN, SEED)
from src.data import load_split
from src.inference import load_tokenizer, load_base_model, load_trained_model, greedy_answers
from src.prompt import is_idk, make_prompt
from src.text import find_match_level
from src.eval import info_weighted_acc, metacognition
from src.capability import step_capability
from src.sft import step_sft
from src.eval import step_eval

CAP_8B = OUT_DIR / "capability_8b.json"
RESULTS = OUT_DIR / "results_8b.json"
results = {"model": MODEL_NAME}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save():
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)


def _recs(model, tok, qs):
    ans = greedy_answers(model, tok, qs)
    out = []
    for q, a in zip(qs, ans):
        idk = is_idk(a)
        ml = find_match_level(a, q.hierarchy) if (a and not idk) else None
        out.append((q, a, idk, ml))
    return out


def _metrics(recs):
    n = len(recs)
    nls = [q.levels[ml] if (ml is not None and ml < len(q.levels)) else None
           for q, _, _, ml in recs]
    pairs = [(not idk, ml is not None) for _, _, idk, ml in recs]
    committed = sum(1 for _, _, idk, _ in recs if not idk)
    on_hier = sum(1 for _, _, _, ml in recs if ml is not None)
    m = dict(n=n, acc=on_hier / n, info=info_weighted_acc(nls),
             idk=sum(1 for _, _, idk, _ in recs if idk) / n,
             wrong=sum(1 for _, _, idk, ml in recs if (not idk) and ml is None) / n,
             sel=on_hier / committed if committed else 0.0,
             d=metacognition(pairs)["d_prime"])
    byrel = {}
    for rel in sorted({q.relation for q, _, _, _ in recs if q.relation}):
        sub = [(q, a, idk, ml) for q, a, idk, ml in recs if q.relation == rel]
        nn = len(sub)
        byrel[rel] = dict(n=nn, acc=sum(1 for _, _, _, ml in sub if ml is not None) / nn,
                          info=info_weighted_acc([q.levels[ml] if (ml is not None and ml < len(q.levels)) else None
                                                  for q, _, _, ml in sub]),
                          wrong=sum(1 for _, _, idk, ml in sub if (not idk) and ml is None) / nn)
    m["per_relation"] = byrel
    return m


def _print(tag, m):
    log(f"{tag}: acc={m['acc']:.3f} info={m['info']:.3f} idk={m['idk']:.3f} "
        f"wrong={m['wrong']:.3f} sel={m['sel']:.3f} d'={m['d']}")


SMOKE = os.environ.get("SMOKE") == "1"   # logic check on the small model
train_qs = load_split("train", LOCATION_RELATIONS)
test_qs = load_split("test", LOCATION_RELATIONS)
if SMOKE:
    train_qs, test_qs = train_qs[:40], test_qs[:40]
    print("*** SMOKE MODE: 40 train / 40 test, tiny capability + SFT ***", flush=True)
log(f"MODEL={MODEL_NAME}  train={len(train_qs)}  test={len(test_qs)}")
tok = load_tokenizer()

# ---------------- STAGE 1: base + IDK baseline ----------------
log("STAGE 1: base + IDK prompt on combined test")
base = load_base_model()
base_recs = _recs(base, tok, test_qs)
del base; gc.collect(); torch.cuda.empty_cache()
results["base"] = _metrics(base_recs)
_print("BASE", results["base"])
save()

# ---------------- STAGE 2: capability curve ----------------
log("STAGE 2: capability curve on combined train (slow)")
step_capability(limit=(8 if SMOKE else None), out_path=CAP_8B, relations=LOCATION_RELATIONS)
results["capability_path"] = str(CAP_8B)
save()

# ---------------- STAGE 3: capability-SFT + eval ----------------
log("STAGE 3: SFT on capability target + eval")
step_sft(save_name="sft_8b", balance=True, capability_path=CAP_8B, smoke_test=SMOKE)
sft, _ = load_trained_model("sft_8b")
sft_recs = _recs(sft, tok, test_qs)
del sft; gc.collect(); torch.cuda.empty_cache()
results["sft"] = _metrics(sft_recs)
_print("SFT ", results["sft"])
save()

# ---------------- STAGE 4: surgical abstention ----------------
try:
    log("STAGE 4: surgical-abstention target (keep base-correct, IDK on confident-wrong)")
    base = load_base_model()
    tr = _recs(base, tok, train_qs)
    del base; gc.collect(); torch.cuda.empty_cache()
    n_keep = sum(1 for _, _, _, ml in tr if ml is not None)
    log(f"  surgical target: keep {n_keep}/{len(tr)} base-correct, IDK the rest")
    records = []
    for q, a, idk, ml in tr:
        comp = f"Location: {a}" if ml is not None else "Location: I don't know"
        records.append({"prompt": [{"role": "user", "content": make_prompt(q.question)}],
                        "completion": [{"role": "assistant", "content": comp}]})
    ds = Dataset.from_list(records)

    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig
    lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"],
                          lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    sft_args = dict(output_dir=str(CKPT_DIR / "sft_surgical_8b_run"), learning_rate=SFT_LR,
                    per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
                    num_train_epochs=SFT_EPOCHS, max_length=SFT_MAX_LEN, completion_only_loss=True,
                    packing=False, logging_steps=50, save_strategy="no",
                    bf16=False, fp16=True, seed=SEED, report_to="none")
    if SMOKE:
        sft_args["max_steps"] = 15
    trainer = SFTTrainer(model=MODEL_NAME, args=SFTConfig(**sft_args),
                         train_dataset=ds, peft_config=lora_cfg)
    trainer.train()
    trainer.save_model(str(CKPT_DIR / "sft_surgical_8b"))
    del trainer; gc.collect(); torch.cuda.empty_cache()

    surg, _ = load_trained_model("sft_surgical_8b")
    surg_recs = _recs(surg, tok, test_qs)
    del surg; gc.collect(); torch.cuda.empty_cache()
    results["surgical"] = _metrics(surg_recs)
    _print("SURG", results["surgical"])
    save()
except Exception as e:
    log(f"STAGE 4 failed (stages 1-3 saved): {e!r}")
    results["surgical_error"] = repr(e)
    save()

# ---------------- summary ----------------
log("DONE. Summary vs 1.5B (base acc0.531/info0.269/d'3.04, sft acc0.548/info0.241/d'2.82):")
for k in ("base", "sft", "surgical"):
    if k in results:
        m = results[k]
        print(f"  {k:9s} acc={m['acc']:.3f} info={m['info']:.3f} idk={m['idk']:.3f} "
              f"wrong={m['wrong']:.3f} d'={m['d']}", flush=True)
print(f"\nresults -> {RESULTS}", flush=True)
