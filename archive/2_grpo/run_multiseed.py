"""GRPO with the specificity reward over 4 seeds: mean and spread. The results were
bimodal, which is why GRPO was dropped as the main method.
"""
import numpy as np
from src.config import REWARD_VARIANT
from src.train import step_train
from src.eval import step_eval

assert REWARD_VARIANT == "spec", f"set REWARD_VARIANT='spec' (got {REWARD_VARIANT})"
SEEDS = [0, 1, 2, 3]

results = []
for s in SEEDS:
    name = f"spec_s{s}"
    print(f"\n############## SEED {s} -> {name} ##############", flush=True)
    step_train(smoke_test=False, seed=s, save_name=name)
    r = step_eval(name)
    results.append((s, r))

print("\n\n================ MULTI-SEED B'-spec SUMMARY ================", flush=True)
for s, r in results:
    d = r.d_prime if r.d_prime is not None else float("nan")
    print(f"seed {s}: acc_hier={r.acc_on_hier:.3f}  sel={r.selective_acc:.3f}  "
          f"spec={r.specific_acc:.3f}  idk={r.idk_rate:.3f}  d'={d:.3f}  NFR={r.nfr}")

def col(attr):
    return [getattr(r, attr) for _, r in results if getattr(r, attr) is not None]

print("\nstat       mean    std     min     max")
for attr, label in [("acc_on_hier","acc_hier"),("selective_acc","sel_acc"),
                    ("specific_acc","spec_acc"),("idk_rate","idk"),("d_prime","d'")]:
    v = col(attr)
    print(f"{label:9s} {np.mean(v):.3f}  {np.std(v):.3f}  {min(v):.3f}  {max(v):.3f}")
