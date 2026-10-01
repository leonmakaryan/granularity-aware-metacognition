"""Train and evaluate one seed of the second SFT recipe in a fresh process.

    for s in 0 1 2 3; do MODEL_DTYPE=bfloat16 python 3_sft/run_oneseed_v2.py $s; done
"""
import os
os.environ.setdefault("MODEL_DTYPE", "bfloat16")

import json
import sys

from src.config import CKPT_DIR, OUT_DIR, LOCATION_RELATIONS
from src.sft import step_sft
from src.eval import step_eval

seed = int(sys.argv[1])
name = f"sft_v2_s{seed}"
TGT = OUT_DIR / "targets_loc_v2.json"

if not (CKPT_DIR / name).exists():
    step_sft(save_name=name, balance=True, seed=seed, capability_path=TGT,
             self_distill=True, dep_weight=2)
else:
    print(f"[skip train] {CKPT_DIR/name} exists, re-eval only")
r = step_eval(name, LOCATION_RELATIONS,
              out_csv=f"trained_eval_v2_s{seed}.csv", cmp_out=f"comparison_v2_s{seed}.json")
rec = dict(seed=seed, name=name, acc_hier=r.acc_on_hier, sel_acc=r.selective_acc,
           info=r.info_wacc, info_signed=r.info_signed,
           info_signed_strict=r.info_signed_strict,
           level_mean=r.level_mean, level_std=r.level_std,
           idk=r.idk_rate, d_prime=r.d_prime)
with open("phase2_v2_multiseed.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print("RESULT", json.dumps(rec))
