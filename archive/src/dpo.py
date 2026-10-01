"""Preference optimization (DPO) on top of the SFT adapter, with pairs from the
capability curve: a reliable specific answer over a needlessly coarse one, and
"I don't know" over a guess when even the country is unreliable. It overshot and
was dropped.
"""

import json
from collections import Counter

from datasets import Dataset

from .config import (CKPT_DIR, CAPABILITY_PATH, BATCH_SIZE, GRAD_ACCUM,
                     DPO_INIT, DPO_LR, DPO_BETA, DPO_EPOCHS, SFT_MAX_LEN, SEED,
                     DPO_COUNTRY_OVER_IDK)
from .prompt import make_prompt
from .inference import load_tokenizer, load_base_model


def _pref(prompt_text: str, chosen: str, rejected: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": prompt_text}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def build_dpo_records(capability: dict) -> list[dict]:
    recs = []
    kinds = Counter()
    for qid, r in capability["questions"].items():
        q, hier = r["question"], r["hierarchy"]
        L, is_idk = r["target_level"], r["target_is_idk"]
        prompt = make_prompt(q)
        last = len(hier) - 1
        country = f"Location: {hier[last]}"
        idk = "Location: I don't know"

        if is_idk:
            recs.append(_pref(prompt, idk, country))      # abstain > unreliable country
            kinds["idk>country"] += 1
        elif L < last:
            chosen = f"Location: {hier[L]}"
            for coarser in range(L + 1, last + 1):
                recs.append(_pref(prompt, chosen, f"Location: {hier[coarser]}"))
                kinds["specific>coarse"] += 1
        elif DPO_COUNTRY_OVER_IDK:
            # Off by default: together with specific>coarse these pairs removed abstention entirely.
            recs.append(_pref(prompt, country, idk))
            kinds["country>idk"] += 1

    print(f"DPO pairs: {dict(kinds)}  total={len(recs)}")
    return recs


def step_dpo(smoke_test: bool = False, save_name: str | None = None):
    print(f"=== DPO {'(SMOKE)' if smoke_test else ''} from {DPO_INIT} ===")
    if not CAPABILITY_PATH.exists():
        raise RuntimeError("Run STEP=capability first.")
    if not (CKPT_DIR / DPO_INIT).exists():
        raise RuntimeError(f"No SFT init at {CKPT_DIR / DPO_INIT}; run STEP=sft first.")
    with open(CAPABILITY_PATH) as f:
        capability = json.load(f)

    from trl import DPOConfig, DPOTrainer
    from peft import PeftModel

    records = build_dpo_records(capability)
    if smoke_test:
        records = records[:120]
        print(f"*** SMOKE: {len(records)} pairs ***")
    ds = Dataset.from_list(records)

    tok = load_tokenizer()
    base = load_base_model()
    # Continue the SFT LoRA (trainable). DPOTrainer builds the reference model by
    # disabling the adapter, so no separate ref copy is needed.
    model = PeftModel.from_pretrained(base, str(CKPT_DIR / DPO_INIT), is_trainable=True)

    dpo_kwargs = dict(
        output_dir=str(CKPT_DIR / "dpo_run"),
        learning_rate=DPO_LR, beta=DPO_BETA, loss_type="sigmoid",
        per_device_train_batch_size=2, gradient_accumulation_steps=GRAD_ACCUM * 2,
        num_train_epochs=(1 if smoke_test else DPO_EPOCHS),
        max_length=SFT_MAX_LEN,
        logging_steps=10, save_strategy="no",
        bf16=False, fp16=True, seed=SEED, report_to="none")
    if smoke_test:
        dpo_kwargs["max_steps"] = 20

    trainer = DPOTrainer(
        model=model, args=DPOConfig(**dpo_kwargs),
        train_dataset=ds, processing_class=tok)
    trainer.train()
    save_to = CKPT_DIR / (save_name if save_name else ("dpo_smoke" if smoke_test else "dpo"))
    trainer.save_model(str(save_to))
    print(f"saved LoRA to {save_to}")
