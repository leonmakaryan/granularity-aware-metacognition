"""Train and evaluate one balanced-SFT seed in a fresh process (TRL fails when a
second trainer starts in the same process).

    for s in 0 1 2 3; do python 3_sft/run_oneseed_sft.py $s; done
"""
import sys, json
from src.sft import step_sft
from src.eval import step_eval
from src.config import CKPT_DIR

seed = int(sys.argv[1])
name = sys.argv[2] if len(sys.argv) > 2 else f"sft_bal_s{seed}"
results_file = sys.argv[3] if len(sys.argv) > 3 else "phase2_sft_multiseed_results.jsonl"

if not (CKPT_DIR / name).exists():
    step_sft(smoke_test=False, save_name=name, balance=True, seed=seed)
else:
    print(f"[skip train] {CKPT_DIR/name} exists, re-eval only")
r = step_eval(name)
rec = dict(seed=seed, name=name,
           acc_hier=r.acc_on_hier, sel_acc=r.selective_acc, spec_acc=r.specific_acc,
           info=r.info_wacc, idk=r.idk_rate, d_prime=r.d_prime, nfr=r.nfr,
           avg_lvl=r.avg_norm_level)
with open(results_file, "a") as f:
    f.write(json.dumps(rec) + "\n")
print("RESULT", json.dumps(rec))
