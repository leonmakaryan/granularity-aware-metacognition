"""Every experimental setting, in one place.

Nothing here reads an environment variable, so two runs of the same stage cannot
silently differ. The one deliberate exception is the confirmation guard in data.py.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MODELS = {"7b": "Qwen/Qwen2.5-7B-Instruct", "1p5b": "Qwen/Qwen2.5-1.5B-Instruct",
          "3b": "Qwen/Qwen2.5-3B-Instruct"}

# One number format for sampling, for the base weights during training, and for evaluation.
DTYPE = "bfloat16"

# Decoding. Qwen's generation_config.json ships top_k=20 and a repetition penalty, which
# generate() applies unless told otherwise, so they are stated explicitly here.
MAX_NEW_TOKENS = 32
K_SAMPLES = 8
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 20
# Each model keeps its shipped repetition penalty (load_model checks this). It matters:
# the prompts contain "I don't know", and a weaker penalty makes copying that phrase
# cheaper. At 1.05 instead of its shipped 1.1, the 1.5B base abstained on 64 of 150 date
# questions instead of 11 (results/1p5b_rep_penalty_1.05).
REPETITION_PENALTY = {"Qwen/Qwen2.5-7B-Instruct": 1.05, "Qwen/Qwen2.5-1.5B-Instruct": 1.1,
                      "Qwen/Qwen2.5-3B-Instruct": 1.05}

# 50%-support rule: the most specific level supported by at least this share of the
# K samples, else abstain.
SUPPORT_THRESHOLD = 0.5

GLOBAL_SEED = 0
SEEDS = (0, 1, 2)
DOMAINS = ("locations", "dates")
CONDITIONS = ("locations", "dates", "joint")

# LoRA fine-tuning on the deployment prompt only, effective batch 16, the same for every run.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 1e-5
EPOCHS = 2
MICRO_BATCH = 4
GRAD_ACCUM = 4
GRADIENT_CHECKPOINTING = False
MAX_LEN = 320

SAMPLE_BATCH = 16        # questions per sampling call, each expanded to K_SAMPLES
SMOKE_EVAL_BATCH = 32    # full runs use the batch size fixed by Gate 4
GATE4_QUESTIONS = 100
GATE4_MAX_DIFFS = 3      # batched path canonical if scorer outcomes differ on <= 3/100

HOLDOUT_FRAC = 0.10      # pilot only
BOOTSTRAP_B = 10000

# Draft format, an experiment not used in the thesis: the model first writes its own guess
# on a "Guess:" line, then the answer line, which alone is graded.
DRAFT_PREFIX = "Guess: "
DRAFT_MAX_NEW_TOKENS = 64   # two lines instead of one; base and plain SFT keep MAX_NEW_TOKENS

# Confirmation slice of each train split, held out from fitting and selection (used only
# by experiments not in the thesis).
CONFIRM_FRAC = 0.20
CONFIRM_SEED = 11
