"""Second SFT recipe on all location relations, with five changes at once: the model's
own answers instead of gold strings, the v2 target rule, the deployment task
weighted twice, bf16, and one direct example per level.

    SMOKE=1 python 3_sft/run_v2.py       python 3_sft/run_v2.py
"""
import os
os.environ.setdefault("MODEL_DTYPE", "bfloat16")   # A4; must precede src imports

import json
import time

from src.config import LOCATION_RELATIONS, OUT_DIR
from src.capability import step_capability, retarget
from src.sft import step_sft
from src.eval import step_eval

SMOKE = os.environ.get("SMOKE") == "1"
CAP_V2 = OUT_DIR / ("capability_loc_v2_smoke.json" if SMOKE else "capability_loc_v2.json")
TGT_V2 = OUT_DIR / ("targets_loc_v2_smoke.json" if SMOKE else "targets_loc_v2.json")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- stage 1: capability v2 (skipped if the cache already exists) ----
if CAP_V2.exists():
    log(f"stage 1: {CAP_V2} exists, reusing")
else:
    log("stage 1: capability v2 (samples + greedy anchor, matcher v2)")
    step_capability(limit=(6 if SMOKE else None), out_path=CAP_V2,
                    relations=LOCATION_RELATIONS)

# ---- stage 2: retarget sweep (offline) ----
log("stage 2: retarget sweep")
with open(CAP_V2) as f:
    cap = json.load(f)

# Sweep for the record; the chosen setting is applied last so it is what's saved.
for (tau, mc, mw) in [(0.5, 3, 3), (0.5, 4, 2), (0.75, 3, 2), (0.5, 3, 2)]:
    retarget(cap, tau=tau, min_commit=mc, max_wrong=mw)

# Greedy-consistency check for the chosen rule (0.5, 3, 2): how often is the
# target coarser than a CORRECT greedy answer (estimator overruling ability)?
overruled = sum(1 for r in cap["questions"].values()
                if r["greedy_match_level"] is not None
                and (r["target_level"] is None or r["target_level"] > r["greedy_match_level"]))
log(f"  targets coarser than a correct greedy answer: {overruled}/{len(cap['questions'])}")
with open(TGT_V2, "w") as f:
    json.dump(cap, f, indent=2)
log(f"  saved {TGT_V2}")

# ---- stage 3: self-distill SFT ----
log("stage 3: self-distill SFT (dep_weight=2)")
step_sft(save_name=("sft_v2_smoke" if SMOKE else "sft_v2_loc"), balance=True,
         capability_path=TGT_V2, self_distill=True, dep_weight=2,
         smoke_test=SMOKE)

# ---- stage 4: eval (isolated outputs) ----
log("stage 4: eval")
step_eval("sft_v2_smoke" if SMOKE else "sft_v2_loc", LOCATION_RELATIONS,
          out_csv="trained_eval_v2.csv", cmp_out="comparison_v2.json")
log("DONE")
