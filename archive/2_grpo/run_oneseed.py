"""Train and evaluate one GRPO seed in a fresh process (TRL fails when a second
trainer starts in the same process). Args: seed [checkpoint name] [results file].
"""
import sys, json
from src.train import step_train
from src.eval import step_eval
from src.config import CKPT_DIR, IDK_REWARD

seed = int(sys.argv[1])
name = sys.argv[2] if len(sys.argv) > 2 else f"spec_s{seed}"
results_file = sys.argv[3] if len(sys.argv) > 3 else "phase2_multiseed_results.jsonl"

if not (CKPT_DIR / name).exists():
    step_train(smoke_test=False, seed=seed, save_name=name)
else:
    print(f"[skip train] {CKPT_DIR/name} exists, re-eval only")
r = step_eval(name)
rec = dict(seed=seed, idk_reward=IDK_REWARD, name=name,
           acc_hier=r.acc_on_hier, sel_acc=r.selective_acc, spec_acc=r.specific_acc,
           idk=r.idk_rate, d_prime=r.d_prime, nfr=r.nfr, avg_lvl=r.avg_norm_level)
with open(results_file, "a") as f:
    f.write(json.dumps(rec) + "\n")
print("RESULT", json.dumps(rec))
