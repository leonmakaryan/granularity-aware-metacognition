"""Settings of the earlier experiments (GRPO, SFT, DPO, evaluation)."""

import os
from pathlib import Path


def _load_env():
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


_load_env()

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
# float16 by default so older evaluations reproduce; later runs set MODEL_DTYPE=bfloat16.
MODEL_DTYPE = os.environ.get("MODEL_DTYPE", "float16")
DATASET = "lukasellinger/granola-eq"
RELATION = "P131"

# The six location relations share one place hierarchy (town, region, country).
LOCATION_RELATIONS = ["P131", "P19", "P20", "P276", "P159", "P740"]
RELATIONS = [RELATION]   # active filter for load_split; set to LOCATION_RELATIONS to extend

OUT_DIR = Path("phase2_results")
OUT_DIR.mkdir(exist_ok=True)
CKPT_DIR = OUT_DIR / "checkpoint"
CALIB_PATH = OUT_DIR / "calibration.json"   # bins + metadata
TARGETS_PATH = OUT_DIR / "targets.json"     # per-qid: entropy, n_hits, target_level, target_bin

# Rollout / generation
NUM_ROLLOUTS = 8
TEMPERATURE = 0.7
MAX_PROMPT_LEN = 256
MAX_COMPLETION_LEN = 32

# Training (same as Formulation B)
LEARNING_RATE = 2e-5
KL_BETA = 0.002
LORA_R = 16
LORA_ALPHA = 32
BATCH_SIZE = 4
GRAD_ACCUM = 4
NUM_EPOCHS = 1

# GRPO rewards: correct and specific > correct and coarse > "I don't know" when the
# model is uncertain > "I don't know" when the model should know > wrong.
IDK_REWARD = float(os.environ.get("IDK_REWARD", "0.4"))
IDK_WHEN_KNOW = -0.5
WRONG_PENALTY = -1.0
LEVEL_FLOOR = 0.1        # minimum reward for any correct on-hierarchy answer
LEVEL_DIFF_K = 1.5       # penalty slope on |normalized level - target normalized level|

# GranuScore only as a small extra reward term; it is noisy on raw outputs.
W_LEVEL = 1.0            # primary: correctness-gated level closeness
W_GRAN = 0.25           # secondary: GranuScore closeness to target gscore (small weight)

# Keep a question for GRPO only if this many base rollouts are correct at some level.
MIN_HITS = 1

# Capability curve (capability.py): the finest level with reliability >= tau, else IDK.
CAPABILITY_PATH = OUT_DIR / "capability.json"
CAPABILITY_K = NUM_ROLLOUTS      # open samples per question used to build the curve
CAPABILITY_TAU = 0.5             # reliability threshold for "reliable at this level"

# SFT, kept light: 3 epochs on ~200 pairs destroyed the model's factual knowledge.
SFT_LR = 1e-5
SFT_EPOCHS = 1
SFT_MAX_LEN = 320
SFT_BALANCE_CAP = 3.0   # max per-label oversample factor for the meta/direct skill tasks

# DPO on top of the SFT adapter (dpo.py).
DPO_INIT = "sft_bal"    # SFT adapter to continue from
DPO_LR = 5e-6
DPO_BETA = 0.1
DPO_EPOCHS = 1
DPO_COUNTRY_OVER_IDK = False   # commit-country>abstain pairs; off (they removed abstention)

# Reward variant: "target" = B' level-target match; "spec" = maximize specificity.
REWARD_VARIANT = "spec"

# Informativeness of a correct place answer, following Lukas's measure for dates:
#   info(level) = (INFO_PRIOR_NL - level) / (INFO_PRIOR_NL - INFO_FINEST_NL)
# city 1.0, region 0.625, country 0.25. The signed version scores
# a wrong answer as minus the information of its estimated level.
INFO_FINEST_NL = 1.0     # finest granularity present in every hierarchy (analog of Lukas's DAY)
INFO_PRIOR_NL = 5.0      # prior anchor, coarser than country (analog of his prior_width_years)

# Two scoring fixes, on by default: the exact-match-first place matcher and the date
# parse retry with "in " + answer. CORRECTED_SCORING=0 restores the older rules.
CORRECTED_SCORING = os.environ.get("CORRECTED_SCORING", "1") == "1"

SEED = 0                 # GRPO seed (varied by the multi-seed driver)
MAX_ANSWER_WORDS = 6     # answers longer than this look like rambling, not a place name
BREVITY_PENALTY = 0.0    # off: it punished the specific answers, which were the longer ones
