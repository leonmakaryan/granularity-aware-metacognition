"""First GRPO version: the reward compares the GranuScore of the answer with a
per-question target granularity (from the entropy of the base answers), plus a
correctness reward. The model no longer writes its own "Level:" field, because it
learned to keep that field constant whatever it answered.
"""

import json
import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from granuscore import GranuScore

# ---------- config ----------

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET = "lukasellinger/granola-eq"
RELATION = "P131"

OUT_DIR = Path("phase2_results")
OUT_DIR.mkdir(exist_ok=True)
CKPT_DIR = OUT_DIR / "checkpoint"
CALIB_PATH = OUT_DIR / "calibration.json"   # bins + metadata
TARGETS_PATH = OUT_DIR / "targets.json"     # per-qid: entropy, n_hits, target_gscore

NUM_ROLLOUTS = 8
TEMPERATURE = 0.7
LEARNING_RATE = 2e-5
KL_BETA = 0.002
LORA_R = 16
LORA_ALPHA = 32
MAX_PROMPT_LEN = 256
MAX_COMPLETION_LEN = 32
BATCH_SIZE = 4
GRAD_ACCUM = 4
NUM_EPOCHS = 1

# Reward weights
W_GRAN = 2.0      # granularity closeness (the metacognition signal)
W_CORRECT = 2.0   # correctness at any level

# Filter: keep a question only if at least this many of NUM_ROLLOUTS base
# rollouts land on the hierarchy at any level.
MIN_HITS = 1

# GranuScore is loaded once at import. Range 0-100, higher = coarser.
_SCORER = GranuScore()
_SCORER("warmup")


def gscore(texts: list[str]) -> list[float]:
    """Batch-score granularity. Empty answers -> 0 (fully specific) so a
    non-answer never reads as coarse."""
    if not texts:
        return []
    clean, empty = [], set()
    for i, t in enumerate(texts):
        if t and t.strip():
            clean.append(t)
        else:
            clean.append("placeholder")
            empty.add(i)
    raw = _SCORER(clean)
    return [0.0 if i in empty else float(raw[i]) for i in range(len(clean))]


# ---------- text utils (from Phase 1) ----------

_ARTICLES = {"the", "a", "an"}
_FILLER = {"of", "the", "a", "an", "and"}


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    tokens = s.split()
    while tokens and tokens[0] in _ARTICLES:
        tokens = tokens[1:]
    return " ".join(tokens)


def content_tokens(s: str) -> set[str]:
    return {t for t in normalize(s).split() if t and t not in _FILLER}


def match_at_level(answer: str, hierarchy_entry: str) -> bool:
    if not answer or not hierarchy_entry:
        return False
    na, nh = normalize(answer), normalize(hierarchy_entry)
    if not na or not nh:
        return False
    if na == nh:
        return True
    if nh in na or na in nh:
        return True
    h_tok, a_tok = content_tokens(hierarchy_entry), content_tokens(answer)
    if h_tok and h_tok.issubset(a_tok):
        return True
    return False


def find_match_level(answer: str, hierarchy: list[str]) -> int | None:
    """Most-specific level index `answer` matches, or None."""
    for idx, h in enumerate(hierarchy):
        if match_at_level(answer, h):
            return idx
    return None


@dataclass
class Question:
    qid: int
    question: str
    hierarchy: list[str]   # most specific -> coarsest
    levels: list[float]


def load_split(split: str) -> list[Question]:
    ds = load_dataset(DATASET, split=split)
    qs = []
    for row in ds:
        if row["relation"] != RELATION:
            continue
        hier = list(row["granola_answers"])
        levs = list(row["normalized_levels"])
        qs.append(Question(qid=row["id"], question=row["question"], hierarchy=hier, levels=levs))
    return qs


def max_hierarchy_depth(qs: list[Question]) -> int:
    return max(len(q.hierarchy) for q in qs)


# ---------- prompt (no Level field) ----------

PROMPT_TEMPLATE = (
    "Answer where the following is located. Reply in exactly this format:\n"
    "Location: <place name>\n\n"
    "Give a single place name after 'Location:'. If you are unsure of the "
    "precise place, name a broader region or country you are confident about "
    "instead of guessing.\n\n"
    "Question: {q}"
)


def make_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(q=question)


# Anchor on the "Location:" marker and take the rest of that line. Whatever the
# model writes before or after the line is ignored — same idea as the old
# "Answer:" regex, just without a level field.
LOCATION_RE = re.compile(r"Location:\s*([^\n]+)", re.IGNORECASE)


def extract_answer(completion: str) -> str:
    """Pull the place name after the 'Location:' marker. Garbage around the
    marker is ignored. Returns '' if the marker is absent.
    """
    if not completion:
        return ""
    m = LOCATION_RE.search(completion)
    if not m:
        return ""
    ans = m.group(1).strip()
    # Trim a trailing sentence if the model continued past the name on the
    # same line ("Czech Republic. It is also..."), and drop a leading article.
    ans = re.split(r"\.\s|\.$", ans, maxsplit=1)[0]
    ans = re.sub(r"^(the|a|an)\s+", "", ans, flags=re.IGNORECASE)
    return ans.strip().rstrip(".,").strip()


# ---------- rewards ----------

def reward_correctness(completion: str, hierarchy: list[str]) -> float:
    """1.0 if the answer matches the hierarchy at ANY level (it's a true place
    in the chain). Specificity is the granularity reward's job, not this one.
    """
    ans = extract_answer(completion)
    if not ans:
        return 0.0
    return 1.0 if find_match_level(ans, hierarchy) is not None else 0.0


def reward_granularity(measured: float, target: float) -> float:
    """Closeness of the answer's GranuScore to the target, falling off quadratically."""
    return max(0.0, 1.0 - abs(measured - target) / 100.0) ** 2


def semantic_entropy(answers: list[str]) -> float:
    keys = [normalize(a) if a else "__empty__" for a in answers]
    counts = Counter(keys)
    n = sum(counts.values())
    if n == 0:
        return 0.0
    probs = [c / n for c in counts.values()]
    return -sum(p * math.log(p) for p in probs if p > 0)


def entropy_to_bin(entropy: float, bins: list[float], n_levels: int) -> int:
    for i, b in enumerate(bins):
        if entropy <= b:
            return i
    return n_levels - 1


# ---------- step 1: calibrate ----------

def step_calibrate():
    """Sample the base model on the training set and cache, per question, the entropy
        of its answers and the target GranuScore that the entropy bin selects."""
    print("=== calibrating on full train set (GranuScore targets) ===")
    train = load_split("train")
    n_levels = max_hierarchy_depth(train)
    print(f"train questions: {len(train)}   hierarchy depth: {n_levels}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto")
    model.eval()

    recs = []
    with torch.no_grad():
        for i, q in enumerate(train):
            chat = tok.apply_chat_template(
                [{"role": "user", "content": make_prompt(q.question)}],
                tokenize=False, add_generation_prompt=True)
            inputs = tok(chat, return_tensors="pt").to(model.device)
            out = model.generate(
                **inputs, max_new_tokens=MAX_COMPLETION_LEN,
                do_sample=True, temperature=TEMPERATURE, top_p=0.95,
                num_return_sequences=NUM_ROLLOUTS,
                pad_token_id=tok.pad_token_id)
            gen = out[:, inputs["input_ids"].shape[1]:]
            answers = [extract_answer(s) for s in tok.batch_decode(gen, skip_special_tokens=True)]
            H = semantic_entropy(answers)
            n_hits = sum(1 for a in answers if a and find_match_level(a, q.hierarchy) is not None)
            recs.append({"qid": q.qid, "entropy": H, "n_hits": n_hits})
            if i % 25 == 0:
                print(f"  calib {i+1}/{len(train)}  H={H:.3f}  hits={n_hits}/{NUM_ROLLOUTS}")

    entropies = [r["entropy"] for r in recs]
    pcts = [100 * k / n_levels for k in range(1, n_levels)]
    bins = np.percentile(entropies, pcts).tolist()

    # Per question: entropy bin -> gold level -> GranuScore of that gold answer.
    # Batch all the gold strings we need to score in one GranuScore call.
    qid_to_q = {q.qid: q for q in train}
    target_levels = {}
    for r in recs:
        q = qid_to_q[r["qid"]]
        b = entropy_to_bin(r["entropy"], bins, n_levels)
        lvl = min(b, len(q.hierarchy) - 1)   # bin may exceed this q's depth
        target_levels[r["qid"]] = lvl
    flat_strings, flat_keys = [], []
    for r in recs:
        q = qid_to_q[r["qid"]]
        flat_strings.append(q.hierarchy[target_levels[r["qid"]]])
        flat_keys.append(r["qid"])
    target_scores = gscore(flat_strings)
    qid_to_target = dict(zip(flat_keys, target_scores))

    for r in recs:
        r["target_bin"] = entropy_to_bin(r["entropy"], bins, n_levels)
        r["target_level"] = target_levels[r["qid"]]
        r["target_gscore"] = qid_to_target[r["qid"]]

    kept = sum(1 for r in recs if r["n_hits"] >= MIN_HITS)
    calib = {
        "n_levels": n_levels, "bins": bins, "percentiles": pcts,
        "n_train_total": len(recs), "n_train_kept": kept, "min_hits": MIN_HITS,
        "entropy_min": float(min(entropies)), "entropy_max": float(max(entropies)),
        "entropy_mean": float(np.mean(entropies)),
    }
    with open(CALIB_PATH, "w") as f:
        json.dump(calib, f, indent=2)
    targets = {str(r["qid"]): {"target_gscore": r["target_gscore"],
                               "target_level": r["target_level"],
                               "target_bin": r["target_bin"],
                               "n_hits": r["n_hits"], "entropy": r["entropy"]}
               for r in recs}
    with open(TARGETS_PATH, "w") as f:
        json.dump(targets, f, indent=2)

    print(f"\nn_levels={n_levels}  bins={[round(b,3) for b in bins]}")
    print(f"target_level dist: {dict(sorted(Counter(r['target_level'] for r in recs).items()))}")
    print(f"target_gscore mean={np.mean([r['target_gscore'] for r in recs]):.1f} "
          f"min={min(r['target_gscore'] for r in recs):.1f} "
          f"max={max(r['target_gscore'] for r in recs):.1f}")
    print(f"hit dist: {dict(sorted(Counter(r['n_hits'] for r in recs).items()))}")
    print(f"kept (n_hits>={MIN_HITS}): {kept}/{len(recs)} ({100*kept/len(recs):.0f}%)")
    print(f"saved {CALIB_PATH} and {TARGETS_PATH}")


# ---------- step 2: train ----------

def step_train(smoke_test: bool = False):
    print(f"=== GRPO training {'(SMOKE)' if smoke_test else ''} — GranuScore ===")
    if not CALIB_PATH.exists() or not TARGETS_PATH.exists():
        raise RuntimeError("Run --step calibrate first.")
    with open(CALIB_PATH) as f:
        calib = json.load(f)
    with open(TARGETS_PATH) as f:
        targets = json.load(f)

    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig

    train_qs = load_split("train")
    val_qs = load_split("validation")

    before = len(train_qs)
    train_qs = [q for q in train_qs
                if targets.get(str(q.qid), {}).get("n_hits", 0) >= MIN_HITS]
    print(f"filter: {before} -> {len(train_qs)} (dropped {before-len(train_qs)})")

    if smoke_test:
        train_qs = train_qs[:100]
        val_qs = val_qs[:20]
    print(f"train: {len(train_qs)}  val: {len(val_qs)}")

    def to_records(qs):
        # Val qids weren't calibrated; give a neutral mid-range target (50) —
        # only used for the eval-loss pass we don't act on.
        out = []
        for q in qs:
            t = targets.get(str(q.qid), {})
            out.append({
                "prompt": make_prompt(q.question),
                "hierarchy": q.hierarchy,
                "qid": q.qid,
                "target_gscore": t.get("target_gscore", 50.0),
            })
        return out

    train_ds = Dataset.from_list(to_records(train_qs))
    val_ds = Dataset.from_list(to_records(val_qs))

    call_count = {"n": 0}

    def reward_fn(completions, **kwargs):
        hierarchies = kwargs["hierarchy"]
        target_gscores = kwargs["target_gscore"]
        prompts = kwargs.get("prompts", [None] * len(completions))
        n = len(completions)
        assert n % NUM_ROLLOUTS == 0

        # Score ALL completions' answers in one GranuScore batch call.
        answers = [extract_answer(c) for c in completions]
        measured = gscore(answers)

        if call_count["n"] == 0:
            print("\n--- first reward batch sample ---")
            print(f"prompt[0]: {str(prompts[0])[:160]}")
            print(f"completion[0]: {completions[0]!r}")
            print(f"answer[0]: {answers[0]!r}")
            print(f"measured_gscore[0]: {measured[0]:.1f}  target_gscore[0]: {target_gscores[0]:.1f}")
            print(f"hierarchy[0]: {hierarchies[0]}")
            print("---\n")
        call_count["n"] += 1

        rewards, gran_sum, corr_sum = [], 0.0, 0.0
        for i in range(n):
            r_gran = reward_granularity(measured[i], target_gscores[i])
            r_corr = reward_correctness(completions[i], hierarchies[i])
            rewards.append(W_GRAN * r_gran + W_CORRECT * r_corr)
            gran_sum += r_gran
            corr_sum += r_corr

        if call_count["n"] % 10 == 0:
            print(f"  [rewards, call {call_count['n']}] "
                  f"gran={gran_sum/n:.3f}  corr={corr_sum/n:.3f}")
        return rewards

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")

    grpo_kwargs = dict(
        output_dir=str(CKPT_DIR), learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_generations=NUM_ROLLOUTS,
        max_completion_length=MAX_COMPLETION_LEN,
        temperature=TEMPERATURE, beta=KL_BETA,
        bf16=False, fp16=True, use_vllm=False, report_to="none")
    if smoke_test:
        grpo_kwargs.update(max_steps=150, logging_steps=5, save_steps=50,
                           save_total_limit=2, num_train_epochs=1)
    else:
        grpo_kwargs.update(num_train_epochs=NUM_EPOCHS, logging_steps=5,
                           save_steps=200, save_total_limit=2)

    trainer = GRPOTrainer(
        model=MODEL_NAME, reward_funcs=reward_fn, args=GRPOConfig(**grpo_kwargs),
        train_dataset=train_ds, eval_dataset=val_ds, peft_config=lora_cfg)
    trainer.train()
    save_to = CKPT_DIR / ("smoke" if smoke_test else "final")
    trainer.save_model(str(save_to))
    print(f"saved LoRA to {save_to}")


# ---------- step 3: eval ----------

@dataclass
class EvalResult:
    method: str
    acc_on_hier: float
    selective_acc: float
    specific_acc: float
    avg_norm_level: float | None
    idk_rate: float
    d_prime: float | None
    yfr: float | None
    nfr: float | None


def _phi_inv(p: float) -> float:
    eps = 1e-3
    return NormalDist().inv_cdf(max(eps, min(1 - eps, p)))


def metacognition(pairs: list[tuple[bool, bool]]) -> dict:
    if not pairs:
        return {"d_prime": None, "yfr": None, "nfr": None}
    yes = [p for p in pairs if p[0]]
    no = [p for p in pairs if not p[0]]
    correct = [p for p in pairs if p[1]]
    incorrect = [p for p in pairs if not p[1]]
    if correct and incorrect:
        hit = sum(1 for p in correct if p[0]) / len(correct)
        fa = sum(1 for p in incorrect if p[0]) / len(incorrect)
        d = _phi_inv(hit) - _phi_inv(fa)
    else:
        d = None
    yfr = sum(1 for p in yes if not p[1]) / len(yes) if yes else None
    nfr = sum(1 for p in no if p[1]) / len(no) if no else None
    return {"d_prime": d, "yfr": yfr, "nfr": nfr}


def step_eval():
    print("=== evaluating trained model on P131 test ===")
    final_dir = CKPT_DIR / "final"
    if not final_dir.exists():
        raise RuntimeError(f"No checkpoint at {final_dir}; run train first.")

    from peft import PeftModel
    test_qs = load_split("test")
    print(f"test: {len(test_qs)} questions")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, str(final_dir))
    model.eval()

    rows = []
    with torch.no_grad():
        for q in test_qs:
            chat = tok.apply_chat_template(
                [{"role": "user", "content": make_prompt(q.question)}],
                tokenize=False, add_generation_prompt=True)
            inputs = tok(chat, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=MAX_COMPLETION_LEN,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            gen = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            ans = extract_answer(gen)
            ml = find_match_level(ans, q.hierarchy) if ans else None
            rows.append({"qid": q.qid, "question": q.question, "hierarchy": q.hierarchy,
                         "raw": gen.strip(), "answer": ans, "match_level": ml})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "trained_eval.csv", index=False)

    n = len(df)
    on_hier = int(df["match_level"].notna().sum())
    committed = int((df["answer"].str.len() > 0).sum())
    specific_correct = sum(1 for r in rows if r["match_level"] == 0)
    levels_committed = [
        q.levels[r["match_level"]] for q, r in zip(test_qs, rows)
        if r["match_level"] is not None and r["match_level"] < len(q.levels)]
    avg_level = float(np.mean(levels_committed)) if levels_committed else None

    # metacog pair: said_yes = committed a matched answer; correct = matched.
    pairs = [(r["match_level"] is not None, r["match_level"] is not None) for r in rows]
    meta = metacognition(pairs)

    trained = EvalResult(
        method="trained-granuscore",
        acc_on_hier=on_hier / n,
        selective_acc=on_hier / committed if committed else 0.0,
        specific_acc=specific_correct / n,
        avg_norm_level=avg_level,
        idk_rate=(n - committed) / n,
        d_prime=meta["d_prime"], yfr=meta["yfr"], nfr=meta["nfr"])

    phase1 = Path("phase1_results/summary.json")
    table = []
    if phase1.exists():
        with open(phase1) as f:
            p1 = json.load(f)
        for key in ["base_direct", "meta_question", "drag"]:
            s = p1[key]
            mc = s.get("metacognition", {})
            table.append(EvalResult(
                method=key, acc_on_hier=s.get("acc_on_hierarchy"),
                selective_acc=s.get("selective_acc"),
                specific_acc=s.get("specific_accuracy"),
                avg_norm_level=s.get("avg_normalized_level"),
                idk_rate=s.get("idk", 0) / s.get("n", 1),
                d_prime=mc.get("d_prime"), yfr=mc.get("yfr"), nfr=mc.get("nfr")))
    else:
        print("(phase1 summary not found — trained only)")
    table.append(trained)

    def fmt(v, p=3):
        if v is None: return "—"
        if isinstance(v, float): return f"{v:.{p}f}"
        return str(v)

    headers = ["method", "acc_hier", "sel_acc", "spec_acc", "avg_lvl", "idk", "d′", "YFR", "NFR"]
    print("\n" + " | ".join(f"{h:>11s}" for h in headers))
    print("-" * (12 * len(headers)))
    for r in table:
        print(" | ".join(f"{x:>11s}" for x in [
            r.method, fmt(r.acc_on_hier), fmt(r.selective_acc), fmt(r.specific_acc),
            fmt(r.avg_norm_level), fmt(r.idk_rate), fmt(r.d_prime), fmt(r.yfr), fmt(r.nfr)]))

    with open(OUT_DIR / "comparison.json", "w") as f:
        json.dump({r.method: r.__dict__ for r in table}, f, indent=2)
    print(f"\nresults: {OUT_DIR}/trained_eval.csv, {OUT_DIR}/comparison.json")


# ---------- inspect / diagnose ----------

def step_inspect(adapter="final", n=20):
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, str(CKPT_DIR / adapter))
    model.eval()
    test = load_split("test")[:n]
    answers = []
    with torch.no_grad():
        for q in test:
            chat = tok.apply_chat_template(
                [{"role": "user", "content": make_prompt(q.question)}],
                tokenize=False, add_generation_prompt=True)
            inputs = tok(chat, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=MAX_COMPLETION_LEN,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            gen = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            answers.append(extract_answer(gen))
    scores = gscore(answers)
    for q, a, s in zip(test, answers, scores):
        ml = find_match_level(a, q.hierarchy) if a else None
        print(f"{q.question[:48]:48s} | '{a}' gs={s:5.1f} | match@L{ml} | {q.hierarchy}")


def step_diagnose(adapter="smoke"):
    """Does the GranuScore of the answers vary with the hierarchy level they match?"""
    from peft import PeftModel
    from scipy.stats import spearmanr
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, str(CKPT_DIR / adapter))
    model.eval()
    test = load_split("test")
    answers = []
    with torch.no_grad():
        for q in test:
            chat = tok.apply_chat_template(
                [{"role": "user", "content": make_prompt(q.question)}],
                tokenize=False, add_generation_prompt=True)
            inputs = tok(chat, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=MAX_COMPLETION_LEN,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            gen = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            answers.append(extract_answer(gen))
    scores = gscore(answers)
    gss, mls = [], []
    for q, a, s in zip(test, answers, scores):
        ml = find_match_level(a, q.hierarchy) if a else None
        if ml is not None:
            gss.append(s); mls.append(ml)
    print(f"\n=== granularity diagnostic ({adapter}) ===")
    print(f"matched answers: {len(gss)}/{len(test)}")
    print(f"measured gscore: mean={np.mean(scores):.1f} std={np.std(scores):.1f} "
          f"min={min(scores):.1f} max={max(scores):.1f}")
    if len(set(mls)) > 1 and len(gss) >= 3:
        rho, p = spearmanr(gss, mls)
        print(f"Spearman(gscore, match_level) = {rho:.3f} (p={p:.3f})")
        print("positive rho -> coarser answers land at coarser levels (working)")
    else:
        print("not enough level variance to correlate")


# ---------- main ----------

STEP = "calibrate"   # "calibrate", "smoke", "train", "eval", "inspect", "diagnose"


def main():
    if STEP == "calibrate":
        step_calibrate()
    elif STEP == "smoke":
        step_train(smoke_test=True)
    elif STEP == "train":
        step_train(smoke_test=False)
    elif STEP == "eval":
        step_eval()
    elif STEP == "inspect":
        step_inspect()
    elif STEP == "diagnose":
        step_diagnose()


if __name__ == "__main__":
    main()
