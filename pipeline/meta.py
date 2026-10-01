"""ESMA-style meta question (Park et al. 2026): does the model say it knows the answer?

    python -m pipeline.meta --model 7b --adapter base --domain dates --split test

One pass per question on ESMA's meta prompt, copied verbatim from esma/prompt.py
(github.com/cosmoquester/ESMA). Stores the short greedy reply, parsed as ESMA's
metric.py does ("yes" anywhere in the lowercased reply), and log P(Yes) - log P(No) at
the first reply token with the probability mass on those two tokens, for ESMA's
continuous confidence. No reference answer is read. The direct answers it is scored
against come from `pipeline.run eval --mode noidk`; see pipeline/esma.py.
"""
import argparse
import json
from types import SimpleNamespace

import torch

from . import config as C
from .data import load
from .generate import generate, load_model

META_QA_PROMPT = """Do you know the answer to the following question? If you know and are sure about the answer, just return "Yes". If you don't know the answer or are uncertain, just return "No".
Question: {question}
"""
META_MAX_NEW_TOKENS = 4


def main():
    from .run import eval_batch

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7b", choices=list(C.MODELS))
    ap.add_argument("--adapter", default="base")
    ap.add_argument("--domain", required=True, choices=["locations", "dates"])
    ap.add_argument("--split", required=True, choices=["train", "test", "val"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    mdir = (C.RESULTS / "smoke" if args.limit else C.RESULTS) / args.model
    items = load(args.domain, args.split)[: args.limit or None]
    adapter = None if args.adapter == "base" else mdir / "adapters" / args.adapter
    model, tok = load_model(C.MODELS[args.model], adapter)
    yes, no = tok.encode("Yes", add_special_tokens=False), tok.encode("No", add_special_tokens=False)
    assert len(yes) == 1 and len(no) == 1, (yes, no)

    prompts = [META_QA_PROMPT.format(question=it.question) for it in items]
    replies = generate(model, tok, prompts, sample=False,
                       batch_size=eval_batch(SimpleNamespace(limit=args.limit, model=args.model)),
                       max_new_tokens=META_MAX_NEW_TOKENS)
    out = mdir / "meta"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.adapter}__{args.domain}_{args.split}__meta.jsonl"
    with open(path, "w") as f:
        for it, p, r in zip(items, prompts, replies):
            head = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                           add_generation_prompt=True)
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([tok(head)["input_ids"]],
                                                      device=model.device)).logits[0, -1].float()
            lsm = torch.log_softmax(logits, -1)
            f.write(json.dumps({"qid": it.qid, "reply": r[0], "yes": int("yes" in r[0].lower()),
                                "yes_no": float(lsm[yes[0]] - lsm[no[0]]),
                                "mass": float(lsm[yes[0]].exp() + lsm[no[0]].exp())}) + "\n")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
