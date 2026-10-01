"""Light multi-task SFT on the capability-curve target.

Three prompt families share one LoRA adapter: the deployment prompt (answer at the
target level, or "I don't know"), a prompt asking for a given level, and a meta
prompt asking which level is reliable. Kept light (1 epoch, low learning rate),
because 3 epochs on ~200 pairs destroyed the model's factual knowledge.
"""

import json
import math
import random
from collections import Counter, defaultdict

from datasets import Dataset

from .config import (MODEL_NAME, MODEL_DTYPE, CKPT_DIR, CAPABILITY_PATH, LORA_R,
                     LORA_ALPHA, BATCH_SIZE, GRAD_ACCUM, SFT_LR, SFT_EPOCHS,
                     SFT_MAX_LEN, SEED, SFT_BALANCE_CAP)
from .prompt import (make_prompt, make_direct_prompt, make_meta_prompt, level_label)
from .text import normalize


def _own_answer(r: dict, level: int, or_finer: bool) -> str | None:
    """The model's own most frequent sampled answer at `level` (or finer with
        `or_finer`), else the greedy answer if it fits, else None (the caller then uses the
        gold string). Training on gold strings the model never produces teaches it to
        hallucinate (Gekhman et al., 2024)."""
    ok = (lambda m: m is not None and m <= level) if or_finer else (lambda m: m == level)
    cands = [a for a, m in zip(r.get("sample_answers", []),
                               r.get("sample_match_levels", [])) if ok(m)]
    if cands:
        counts = Counter(normalize(a) for a in cands)
        best_norm = counts.most_common(1)[0][0]
        for a in cands:
            if normalize(a) == best_norm:
                return a
    if (not r.get("greedy_is_idk", True) and ok(r.get("greedy_match_level"))):
        return r["greedy_answer"]
    return None


def build_sft_records(capability: dict, balance: bool = True, seed: int = SEED,
                      self_distill: bool = False, dep_weight: int = 1) -> list[dict]:
    """Build the SFT pairs from a cached capability curve: one deployment and one meta
        example per question, and direct examples from the target level down to the country.

        balance       oversample rare labels in the meta and direct tasks (up to SFT_BALANCE_CAP)
        self_distill  use the model's own answers, gold strings only as fallback
        dep_weight    repeat the deployment task so it is not a small part of the data"""
    deployment, meta, direct = [], [], []
    n_dep_fallback = n_dep_self = 0
    for qid, r in capability["questions"].items():
        q, hier = r["question"], r["hierarchy"]
        L, is_idk = r["target_level"], r["target_is_idk"]

        if is_idk:
            dep_ans = "I don't know"
        elif self_distill:
            own = _own_answer(r, L, or_finer=True)
            dep_ans = own if own is not None else hier[L]
            n_dep_self += own is not None
            n_dep_fallback += own is None
        else:
            dep_ans = hier[L]
        meta_label = level_label(hier, L)
        deployment.append(_tagged(make_prompt(q), f"Location: {dep_ans}", meta_label))
        meta.append(_tagged(make_meta_prompt(q), f"Reliable level: {meta_label}", meta_label))
        if not is_idk:
            seen_buckets = set()
            for lvl in range(L, len(hier)):
                lab = level_label(hier, lvl)
                if lab in seen_buckets:
                    continue          # A6: no second (contradictory) example per bucket
                seen_buckets.add(lab)
                d_ans = hier[lvl]
                if self_distill:
                    own = _own_answer(r, lvl, or_finer=False)
                    if own is not None:
                        d_ans = own
                direct.append(_tagged(make_direct_prompt(q, lab), f"Location: {d_ans}", lab))

    if self_distill:
        total = max(n_dep_self + n_dep_fallback, 1)
        print(f"  self-distill: own answer {n_dep_self}/{total} "
              f"({100*n_dep_self/total:.0f}%), gold fallback {n_dep_fallback}")
        if n_dep_self == 0:
            raise RuntimeError("self_distill=True but no sample answers in the "
                               "capability cache — rerun step_capability (Jul 7+).")

    if balance:
        meta = _balance_by_label(meta, SFT_BALANCE_CAP, seed)
        direct = _balance_by_label(direct, SFT_BALANCE_CAP, seed)

    records = deployment * dep_weight + meta + direct
    random.Random(seed).shuffle(records)
    print(f"SFT examples: deployment={len(deployment)}x{dep_weight} meta={len(meta)} "
          f"direct={len(direct)}  total={len(records)}")
    print(f"  meta labels: {dict(Counter(r['_label'] for r in meta))}")
    print(f"  direct labels: {dict(Counter(r['_label'] for r in direct))}")
    return [{"prompt": r["prompt"], "completion": r["completion"]} for r in records]


def _balance_by_label(records: list[dict], cap: float, seed: int = SEED) -> list[dict]:
    """Oversample (with replacement) each label group up to the largest group,
    but no more than `cap` x its original size."""
    groups = defaultdict(list)
    for r in records:
        groups[r["_label"]].append(r)
    if not groups:
        return records
    maxc = max(len(g) for g in groups.values())
    rng = random.Random(seed)
    out = []
    for g in groups.values():
        target = min(maxc, math.ceil(len(g) * cap))
        out.extend(g)
        if target > len(g):
            out.extend(rng.choices(g, k=target - len(g)))
    return out


def _tagged(prompt_text: str, completion_text: str, label: str) -> dict:
    """Conversational prompt/completion (+ a label tag used only for balancing;
    stripped before the Dataset is built). TRL applies the chat template and,
    with completion_only_loss, masks the prompt tokens."""
    return {
        "prompt": [{"role": "user", "content": prompt_text}],
        "completion": [{"role": "assistant", "content": completion_text}],
        "_label": label,
    }


def step_sft(smoke_test: bool = False, save_name: str | None = None, balance: bool = True,
             seed: int = SEED, capability_path=CAPABILITY_PATH,
             self_distill: bool = False, dep_weight: int = 1):
    print(f"=== light multi-task SFT {'(SMOKE)' if smoke_test else ''} balance={balance} "
          f"seed={seed} self_distill={self_distill} dep_weight={dep_weight} dtype={MODEL_DTYPE} ===")
    if not capability_path.exists():
        raise RuntimeError(f"capability target {capability_path} missing — run STEP=capability first.")
    with open(capability_path) as f:
        capability = json.load(f)

    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig

    records = build_sft_records(capability, balance=balance, seed=seed,
                                self_distill=self_distill, dep_weight=dep_weight)
    if smoke_test:
        records = records[:120]
        print(f"*** SMOKE: {len(records)} examples ***")
    ds = Dataset.from_list(records)

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")

    sft_kwargs = dict(
        output_dir=str(CKPT_DIR / "sft_run"),
        learning_rate=SFT_LR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=(1 if smoke_test else SFT_EPOCHS),
        max_length=SFT_MAX_LEN,
        completion_only_loss=True,
        packing=False,
        logging_steps=10,
        save_strategy="no",
        bf16=(MODEL_DTYPE == "bfloat16"), fp16=(MODEL_DTYPE != "bfloat16"),
        seed=seed, report_to="none")
    if smoke_test:
        sft_kwargs["max_steps"] = 30

    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=SFTConfig(**sft_kwargs),
        train_dataset=ds,
        peft_config=lora_cfg)
    trainer.train()
    save_to = CKPT_DIR / (save_name if save_name else ("sft_smoke" if smoke_test else "sft"))
    trainer.save_model(str(save_to))
    print(f"saved LoRA to {save_to}")
