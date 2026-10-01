"""Early stopping: train 2 epochs, evaluate every checkpoint on a slice of the
training set, keep the best, and test it. Asked whether informativeness ever rises
above the base model early in training; it did not.
"""
import json, gc, random, glob, os

import torch
from datasets import Dataset

from src.config import (LOCATION_RELATIONS, OUT_DIR, CKPT_DIR, MODEL_NAME, LORA_R,
                        LORA_ALPHA, BATCH_SIZE, GRAD_ACCUM, SFT_LR, SFT_MAX_LEN, SEED)
from src.data import load_split
from src.sft import build_sft_records
from src.inference import load_tokenizer, load_base_model, greedy_answers
from src.prompt import is_idk
from src.text import find_match_level
from src.eval import info_weighted_acc, metacognition

CAP = OUT_DIR / "capability_loc.json"
RUN_DIR = CKPT_DIR / "earlystop_run"
VAL_FRAC = 0.12
EPOCHS = 2

# ---------- split train into train'/val by qid ----------
cap = json.load(open(CAP))
qids = list(cap["questions"].keys())
random.Random(SEED).shuffle(qids)
n_val = int(len(qids) * VAL_FRAC)
val_qids = set(qids[:n_val])
tr_qids = set(qids[n_val:])
cap_tr = {**cap, "questions": {q: r for q, r in cap["questions"].items() if q in tr_qids}}

train_all = load_split("train", LOCATION_RELATIONS)
val_qs = [q for q in train_all if str(q.qid) in val_qids]
test_qs = load_split("test", LOCATION_RELATIONS)
print(f"train'={len(tr_qids)}  val={len(val_qs)}  test={len(test_qs)}", flush=True)

tok = load_tokenizer()


def eval_on(qs, adapter_path=None):
    """Greedy deployment answers -> info / acc / idk / d'. Reloads base each call
    (correct over clever; a handful of checkpoints)."""
    base = load_base_model()
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(adapter_path))
        model.eval()
    else:
        model = base
    ans = greedy_answers(model, tok, qs)
    nls, pairs = [], []
    for q, a in zip(qs, ans):
        idk = is_idk(a)
        ml = find_match_level(a, q.hierarchy) if (a and not idk) else None
        nls.append(q.levels[ml] if (ml is not None and ml < len(q.levels)) else None)
        pairs.append((not idk, ml is not None))
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return dict(info=info_weighted_acc(nls),
                acc=sum(1 for _, c in pairs if c) / len(pairs),
                idk=sum(1 for c, _ in pairs if not c) / len(pairs),
                d=metacognition(pairs)["d_prime"])


# ---------- step 0 = base, on val ----------
base_val = eval_on(val_qs)
print(f"\n[val] step0 BASE: info={base_val['info']:.3f} acc={base_val['acc']:.3f} "
      f"idk={base_val['idk']:.3f} d'={base_val['d']:.3f}", flush=True)

# ---------- train, saving the trajectory ----------
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

records = build_sft_records(cap_tr, balance=True, seed=SEED)
ds = Dataset.from_list(records)
steps_per_epoch = -(-len(ds) // (BATCH_SIZE * GRAD_ACCUM))
total_steps = steps_per_epoch * EPOCHS
save_steps = max(50, total_steps // 8)
print(f"records={len(ds)} steps/epoch={steps_per_epoch} total={total_steps} save_every={save_steps}", flush=True)

lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
trainer = SFTTrainer(
    model=MODEL_NAME,
    args=SFTConfig(output_dir=str(RUN_DIR), learning_rate=SFT_LR,
                   per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
                   num_train_epochs=EPOCHS, max_length=SFT_MAX_LEN, completion_only_loss=True,
                   packing=False, logging_steps=50, save_strategy="steps", save_steps=save_steps,
                   bf16=False, fp16=True, seed=SEED, report_to="none"),
    train_dataset=ds, peft_config=lora_cfg)
trainer.train()
del trainer
gc.collect()
torch.cuda.empty_cache()

# ---------- evaluate the trajectory on val ----------
ckpts = sorted(glob.glob(str(RUN_DIR / "checkpoint-*")),
               key=lambda p: int(p.split("-")[-1]))
print(f"\n=== val trajectory ({len(ckpts)} checkpoints + base) ===", flush=True)
traj = [("base(step0)", base_val)]
for c in ckpts:
    step = int(c.split("-")[-1])
    m = eval_on(val_qs, c)
    traj.append((f"step{step}", m))
    print(f"  {f'step{step}':>12s}: info={m['info']:.3f} acc={m['acc']:.3f} "
          f"idk={m['idk']:.3f} d'={m['d']:.3f}", flush=True)

# best by val info (base counts as a candidate = pure early stop to step 0)
best_name, best = max(traj, key=lambda kv: kv[1]["info"])
best_ckpt = None if best_name == "base(step0)" else \
    str(RUN_DIR / f"checkpoint-{best_name[4:]}")
print(f"\nBEST by val info: {best_name} (info={best['info']:.3f})", flush=True)

# ---------- final readout: best vs base, on TEST ----------
print("\n=== TEST readout ===", flush=True)
base_test = eval_on(test_qs, None)
best_test = base_test if best_ckpt is None else eval_on(test_qs, best_ckpt)
print(f"  base+IDK : info={base_test['info']:.3f} acc={base_test['acc']:.3f} "
      f"idk={base_test['idk']:.3f} d'={base_test['d']:.3f}")
print(f"  {best_name:>9s}: info={best_test['info']:.3f} acc={best_test['acc']:.3f} "
      f"idk={best_test['idk']:.3f} d'={best_test['d']:.3f}")
print(f"\nVERDICT: early-stopped SFT {'BEATS' if best_test['info'] > base_test['info'] else 'does NOT beat'} "
      f"base on test info ({best_test['info']:.3f} vs {base_test['info']:.3f})")
