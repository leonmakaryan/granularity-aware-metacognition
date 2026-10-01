"""GRPO training with LoRA (8 rollouts, T=0.7, lr 2e-5, KL beta 0.002).

TRL passes the dataset columns to the reward function as flat lists repeated once per
rollout. stop_strings cannot be used, since TRL's generate gets no tokenizer.
"""

import json

from datasets import Dataset

from .config import (MODEL_NAME, CALIB_PATH, TARGETS_PATH, CKPT_DIR, NUM_ROLLOUTS,
                     LEARNING_RATE, KL_BETA, LORA_R, LORA_ALPHA, BATCH_SIZE,
                     GRAD_ACCUM, NUM_EPOCHS, TEMPERATURE, MAX_COMPLETION_LEN, MIN_HITS,
                     REWARD_VARIANT, SEED)
from .data import load_split
from .prompt import make_prompt, extract_answer
from .reward import reward_b2, reward_specificity
from .scoring import gscore


def step_train(smoke_test: bool = False, seed: int | None = None, save_name: str | None = None):
    print(f"=== GRPO training {'(SMOKE)' if smoke_test else ''} — B' gated reward ===")
    if not CALIB_PATH.exists() or not TARGETS_PATH.exists():
        raise RuntimeError("Run STEP=calibrate first.")
    with open(CALIB_PATH) as f:
        calib = json.load(f)
    with open(TARGETS_PATH) as f:
        targets = json.load(f)

    high_entropy_bin = calib["n_levels"] - 1
    reward_fn_sel = {"target": reward_b2, "spec": reward_specificity}[REWARD_VARIANT]
    print(f"reward variant: {REWARD_VARIANT}")

    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig

    train_qs = load_split("train")
    val_qs = load_split("val")

    before = len(train_qs)
    train_qs = [q for q in train_qs
                if targets.get(str(q.qid), {}).get("n_hits", 0) >= MIN_HITS]
    print(f"filter: {before} -> {len(train_qs)} (dropped {before - len(train_qs)})")

    if smoke_test:
        train_qs = train_qs[:100]
        val_qs = val_qs[:20]
    print(f"train: {len(train_qs)}  val: {len(val_qs)}")

    def to_records(qs):
        out = []
        for q in qs:
            t = targets.get(str(q.qid), {})
            out.append({
                "prompt": make_prompt(q.question),
                "hierarchy": q.hierarchy,
                "levels": q.levels,
                "qid": q.qid,
                "target_level": t.get("target_level", high_entropy_bin),
                "target_bin": t.get("target_bin", high_entropy_bin),
                "target_gscore": t.get("target_gscore", 50.0),
            })
        return out

    train_ds = Dataset.from_list(to_records(train_qs))
    val_ds = Dataset.from_list(to_records(val_qs))

    call_count = {"n": 0}

    def reward_fn(completions, **kwargs):
        hierarchies = kwargs["hierarchy"]
        levels_list = kwargs["levels"]
        target_levels = kwargs["target_level"]
        target_bins = kwargs["target_bin"]
        target_gscores = kwargs["target_gscore"]
        prompts = kwargs.get("prompts", [None] * len(completions))
        n = len(completions)

        answers = [extract_answer(c) for c in completions]
        measured = gscore(answers)   # one batched GranuScore call

        rewards = []
        for i in range(n):
            rewards.append(reward_fn_sel(
                answers[i], hierarchies[i], levels_list[i],
                target_levels[i], target_bins[i], high_entropy_bin,
                measured_gs=measured[i], target_gs=target_gscores[i]))

        if call_count["n"] == 0:
            print("\n--- first reward batch sample ---")
            print(f"prompt[0]: {str(prompts[0])[:140]}")
            print(f"completion[0]: {completions[0]!r}")
            print(f"answer[0]: {answers[0]!r}  measured_gs={measured[0]:.1f}")
            print(f"target_level[0]={target_levels[0]} target_bin[0]={target_bins[0]} "
                  f"target_gs[0]={target_gscores[0]:.1f}")
            print(f"reward[0]={rewards[0]:.3f}  hierarchy[0]={hierarchies[0]}")
            print("---\n")
        call_count["n"] += 1
        if call_count["n"] % 10 == 0:
            mn = sum(rewards) / n
            print(f"  [rewards, call {call_count['n']}] mean={mn:.3f} "
                  f"min={min(rewards):.2f} max={max(rewards):.2f}")
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
        temperature=TEMPERATURE, beta=KL_BETA, seed=(seed if seed is not None else SEED),
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
    save_to = CKPT_DIR / (save_name if save_name else (f"{REWARD_VARIANT}_smoke" if smoke_test else REWARD_VARIANT))
    trainer.save_model(str(save_to))
    print(f"saved LoRA to {save_to}")
