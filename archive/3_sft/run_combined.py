"""Balanced SFT on all six location relations, evaluated per relation. Needs the
capability curve of all six relations (src.capability.step_capability).
"""
from src.config import LOCATION_RELATIONS, OUT_DIR
from src.sft import step_sft
from src.eval import step_eval

CAP_LOC = OUT_DIR / "capability_loc.json"

print("############## COMBINED: balanced SFT -> sft_bal_loc ##############", flush=True)
step_sft(save_name="sft_bal_loc", balance=True, capability_path=CAP_LOC)

print("\n############## COMBINED: eval (per-relation) ##############", flush=True)
step_eval("sft_bal_loc", relations=LOCATION_RELATIONS)
