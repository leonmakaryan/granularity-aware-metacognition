"""Shared model loading and greedy generation (eval / controllability / inspect)."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_NAME, MODEL_DTYPE, CKPT_DIR, MAX_COMPLETION_LEN
from .prompt import make_prompt, extract_answer


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=getattr(torch, MODEL_DTYPE), device_map="auto")
    model.eval()
    return model


def load_trained_model(adapter: str = "final"):
    """Base model + LoRA adapter from checkpoint/<adapter>."""
    from peft import PeftModel
    tok = load_tokenizer()
    base = load_base_model()
    model = PeftModel.from_pretrained(base, str(CKPT_DIR / adapter))
    model.eval()
    return model, tok


def greedy_answers(model, tok, questions, prompt_fn=None, raw=False):
    """One greedy answer per question. `prompt_fn` replaces the default prompt; with
        raw=True the whole completion is returned instead of the extracted place."""
    if prompt_fn is None:
        prompt_fn = make_prompt
    answers = []
    with torch.no_grad():
        for q in questions:
            chat = tok.apply_chat_template(
                [{"role": "user", "content": prompt_fn(q.question)}],
                tokenize=False, add_generation_prompt=True)
            inputs = tok(chat, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=MAX_COMPLETION_LEN,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            gen = tok.decode(out[0, inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            answers.append(gen.strip() if raw else extract_answer(gen))
    return answers
