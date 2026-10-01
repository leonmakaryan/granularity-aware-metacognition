"""Model loading and generation, shared by sampling and greedy decoding."""
import hashlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config as C


def load_model(model_name: str, adapter=None, dtype: str = C.DTYPE):
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=getattr(torch, dtype), device_map={"": 0})
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    shipped = model.generation_config.repetition_penalty
    assert abs(shipped - C.REPETITION_PENALTY[model_name]) < 1e-9, \
        f"{model_name} ships repetition_penalty {shipped}, config says {C.REPETITION_PENALTY[model_name]}"
    model.pipeline_repetition_penalty = C.REPETITION_PENALTY[model_name]
    return model, tok


def batch_seed(tag: str, index: int) -> int:
    """Sampling seed for one batch. Fixed question order and batch size make a resumed
    run restart on a batch boundary and reproduce it exactly."""
    return int(hashlib.sha256(f"{C.GLOBAL_SEED}|{tag}|{index}".encode()).hexdigest()[:8], 16)


def generate(model, tok, prompts: list[str], *, sample: bool, batch_size: int,
             tag: str = "", first_batch: int = 0, log=None,
             max_new_tokens: int | None = None) -> list[list[str]]:
    """-> one list per prompt: K_SAMPLES completions when sampling, else one."""
    per = C.K_SAMPLES if sample else 1
    out = []
    for b, start in enumerate(range(0, len(prompts), batch_size)):
        chunk = prompts[start:start + batch_size]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        kw = dict(max_new_tokens=max_new_tokens or C.MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id,
                  repetition_penalty=model.pipeline_repetition_penalty)
        if sample:
            torch.manual_seed(batch_seed(tag, first_batch + b))
            kw.update(do_sample=True, temperature=C.TEMPERATURE, top_p=C.TOP_P,
                      top_k=C.TOP_K, num_return_sequences=per)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.no_grad():
            seqs = model.generate(**enc, **kw)
        dec = tok.batch_decode(seqs[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i in range(len(chunk)):
            out.append([t.strip() for t in dec[i * per:(i + 1) * per]])
        if log and b % 10 == 0:
            log(f"  generated {min(start + batch_size, len(prompts))}/{len(prompts)}")
    return out
