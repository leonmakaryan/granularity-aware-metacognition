"""KTO on top of the locations SFT, with unpaired labels: correct answers and
fitting abstentions are desirable, wrong answers and needless abstentions are not.
The first run collapsed to abstaining on most questions.

    SMOKE=1 python 3_sft/run_kto.py      python 3_sft/run_kto.py [seed]
"""
import os
os.environ.setdefault("MODEL_DTYPE", "bfloat16")

import json
import sys
import time

import torch
from datasets import Dataset

from src.config import (OUT_DIR, CKPT_DIR, MODEL_NAME, MODEL_DTYPE, GRAD_ACCUM,
                        SFT_MAX_LEN, SEED, LOCATION_RELATIONS, MAX_COMPLETION_LEN)
from src.data import load_split
from src.inference import load_tokenizer, load_base_model, load_trained_model
from src.prompt import make_prompt, extract_answer, is_idk
from src.text import find_match_level, normalize
from src.eval import step_eval

SMOKE = os.environ.get("SMOKE") == "1"
RUN_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
INIT_ADAPTER = "sft_v2_loc"
ADAPTER = "kto_smoke" if SMOKE else f"kto_s{RUN_SEED}"
SAMPLES = OUT_DIR / ("kto_samples_smoke.jsonl" if SMOKE else "kto_samples.jsonl")
TARGETS = OUT_DIR / "targets_loc_v2.json"
K = 8
KTO_LR = 5e-6
KTO_BETA = 0.1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stage1_sample():
    if SAMPLES.exists():
        log(f"stage 1: {SAMPLES} exists, reusing")
        return
    log("stage 1: on-policy sampling from the SFT policy")
    train = load_split("train", LOCATION_RELATIONS)
    if SMOKE:
        train = train[:12]
    model, tok = load_trained_model(INIT_ADAPTER)
    torch.manual_seed(RUN_SEED)
    with open(SAMPLES, "w") as f, torch.no_grad():
        for i, q in enumerate(train):
            chat = tok.apply_chat_template(
                [{"role": "user", "content": make_prompt(q.question)}],
                tokenize=False, add_generation_prompt=True)
            inputs = tok(chat, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=MAX_COMPLETION_LEN,
                                 do_sample=True, temperature=0.7, top_p=0.95,
                                 num_return_sequences=K,
                                 pad_token_id=tok.pad_token_id)
            gen = out[:, inputs["input_ids"].shape[1]:]
            answers = [extract_answer(t) for t in
                       tok.batch_decode(gen, skip_special_tokens=True)]
            f.write(json.dumps({"qid": q.qid, "question": q.question,
                                "hierarchy": q.hierarchy, "answers": answers}) + "\n")
            if i % 100 == 0:
                log(f"  sample {i+1}/{len(train)}")
    del model
    torch.cuda.empty_cache()


def stage2_label():
    log("stage 2: labeling")
    with open(TARGETS) as f:
        targets = {qid: r["target_is_idk"]
                   for qid, r in json.load(f)["questions"].items()}
    records, n_d, n_u = [], 0, 0
    for line in open(SAMPLES):
        r = json.loads(line)
        idk_target = targets.get(str(r["qid"]))
        if idk_target is None:
            continue
        seen = set()
        for a in r["answers"]:
            key = normalize(a) if a else "<empty>"
            if key in seen:
                continue
            seen.add(key)
            if is_idk(a):
                label, completion = idk_target, "Location: I don't know"
            else:
                ml = find_match_level(a, r["hierarchy"])
                label, completion = ml is not None, f"Location: {a}"
            records.append({
                "prompt": [{"role": "user", "content": make_prompt(r["question"])}],
                "completion": [{"role": "assistant", "content": completion}],
                "label": label})
            n_d += label
            n_u += not label
    print(f"  KTO data: {len(records)} unique (desirable {n_d}, undesirable {n_u})")
    return records, n_d, n_u


def stage3_train(records, n_d, n_u):
    log("stage 3: KTO training")
    from trl import KTOConfig, KTOTrainer
    from peft import PeftModel
    tok = load_tokenizer()
    base = load_base_model()
    model = PeftModel.from_pretrained(base, str(CKPT_DIR / INIT_ADAPTER),
                                      is_trainable=True)
    # weights: put w_u*n_u at ~1.33x w_d*n_d (the recommended band's edge that
    # punishes undesirable hardest)
    w_u = max(1.0, 1.33 * n_d / max(n_u, 1))
    print(f"  desirable_weight=1.0 undesirable_weight={w_u:.2f} "
          f"(w_u*n_u/w_d*n_d = {w_u*n_u/max(n_d,1):.2f})")
    kwargs = dict(output_dir=str(CKPT_DIR / "kto_run"), learning_rate=KTO_LR,
                  beta=KTO_BETA, desirable_weight=1.0, undesirable_weight=w_u,
                  per_device_train_batch_size=2,
                  gradient_accumulation_steps=GRAD_ACCUM * 2,
                  num_train_epochs=1, max_length=SFT_MAX_LEN,
                  logging_steps=25, save_strategy="no",
                  bf16=(MODEL_DTYPE == "bfloat16"), seed=RUN_SEED,
                  report_to="none")
    if SMOKE:
        kwargs["max_steps"] = 15
    trainer = KTOTrainer(model=model, args=KTOConfig(**kwargs),
                         train_dataset=Dataset.from_list(records),
                         processing_class=tok)
    trainer.train()
    trainer.save_model(str(CKPT_DIR / ADAPTER))
    log(f"saved {CKPT_DIR / ADAPTER}")
    del trainer, model, base
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def main():
    log(f"KTO from {INIT_ADAPTER}, seed={RUN_SEED}, dtype={MODEL_DTYPE}")
    stage1_sample()
    records, n_d, n_u = stage2_label()
    if SMOKE:
        records = records[:60]
    stage3_train(records, n_d, n_u)
    log("stage 4: eval")
    step_eval(ADAPTER, LOCATION_RELATIONS,
              out_csv=f"{ADAPTER}_eval.csv", cmp_out=f"comparison_{ADAPTER}.json")
    log("DONE")


if __name__ == "__main__":
    main()
