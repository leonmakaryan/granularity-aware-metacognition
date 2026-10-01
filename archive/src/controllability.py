"""Controllability test: ask for a city, a region or a country on
the same questions and check whether the level of the answers actually moves.
"""

import json

import numpy as np
import torch

from .config import MODEL_NAME, CKPT_DIR, OUT_DIR
from .data import load_split
from .prompt import is_idk
from .scoring import gscore
from .inference import load_tokenizer, load_base_model, greedy_answers

CTRL_PROMPTS = {
    "specific": (
        "Answer where the following is located. Reply in exactly this format:\n"
        "Location: <place name>\n\n"
        "Give the most specific place you know (city or district if possible).\n\n"
        "Question: {q}"
    ),
    "country": (
        "Answer where the following is located. Reply in exactly this format:\n"
        "Location: <place name>\n\n"
        "Give only the country, nothing more specific.\n\n"
        "Question: {q}"
    ),
    "vague": (
        "Answer where the following is located. Reply in exactly this format:\n"
        "Location: <place name>\n\n"
        "Give a very broad region (continent or large area). "
        "If you are not sure at all, write: Location: I don't know\n\n"
        "Question: {q}"
    ),
}


def step_controllability(adapter: str = "final", n_questions: int = 30):
    from peft import PeftModel

    print(f"\n=== controllability test ({adapter}, n={n_questions}) ===")
    test_qs = load_split("test")[:n_questions]
    tok = load_tokenizer()

    results = {}
    for variant_name, tpl in CTRL_PROMPTS.items():
        print(f"\n  variant: {variant_name}")
        prompt_fn = lambda q, t=tpl: t.format(q=q)

        base_model = load_base_model()
        base_answers = greedy_answers(base_model, tok, test_qs, prompt_fn=prompt_fn)
        del base_model
        torch.cuda.empty_cache()

        base2 = load_base_model()
        trained = PeftModel.from_pretrained(base2, str(CKPT_DIR / adapter))
        trained.eval()
        trained_answers = greedy_answers(trained, tok, test_qs, prompt_fn=prompt_fn)
        del trained
        torch.cuda.empty_cache()

        results[variant_name] = {
            "base_mean_gs": float(np.mean(gscore(base_answers))),
            "trained_mean_gs": float(np.mean(gscore(trained_answers))),
            "base_idk_rate": sum(1 for a in base_answers if is_idk(a)) / n_questions,
            "trained_idk_rate": sum(1 for a in trained_answers if is_idk(a)) / n_questions,
        }
        r = results[variant_name]
        print(f"    base:    gs={r['base_mean_gs']:.1f}  idk={r['base_idk_rate']:.2f}")
        print(f"    trained: gs={r['trained_mean_gs']:.1f}  idk={r['trained_idk_rate']:.2f}")

    print("\n=== controllability summary ===")
    print(f"{'variant':10s} | {'base gs':>8s} | {'trained gs':>10s} | {'delta':>6s}")
    print("-" * 45)
    for v, r in results.items():
        print(f"{v:10s} | {r['base_mean_gs']:8.1f} | {r['trained_mean_gs']:10.1f} | "
              f"{r['trained_mean_gs'] - r['base_mean_gs']:+6.1f}")

    sp, vg = results.get("specific", {}), results.get("vague", {})
    if sp and vg:
        base_spread = vg["base_mean_gs"] - sp["base_mean_gs"]
        trained_spread = vg["trained_mean_gs"] - sp["trained_mean_gs"]
        print(f"\nspecific->vague spread: base={base_spread:+.1f}  trained={trained_spread:+.1f}")
        print("trained MORE controllable — adaptation confirmed" if trained_spread > base_spread
              else "trained NOT more controllable — model may not have adapted")

    with open(OUT_DIR / "controllability.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {OUT_DIR}/controllability.json")
