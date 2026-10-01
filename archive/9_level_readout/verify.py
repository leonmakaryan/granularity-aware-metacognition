"""How strongly the model endorses a proposed answer: log P(Yes) - log P(No) on a
yes/no question about it. (Belonged to pipeline/.)
"""
import argparse
import json

import pandas as pd
import torch

from . import config as C
from .data import load
from .features import candidates
from .generate import load_model
from .score import extract

VERIFY = ("Question: {q}\nProposed answer: {a}\n\n"
          "Is the proposed answer correct? Reply with only Yes or No.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7b", choices=list(C.MODELS))
    ap.add_argument("--adapter", default="base")
    ap.add_argument("--domain", required=True, choices=["locations", "dates"])
    ap.add_argument("--split", required=True, choices=["train", "test", "val"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    mdir = C.RESULTS / args.model
    preds = pd.read_csv(mdir / "preds" / f"{args.adapter}__{args.domain}_{args.split}__deploy.csv",
                        dtype={"qid": str}, keep_default_na=False)
    raw = dict(zip(preds.qid, preds.raw))
    items = load(args.domain, args.split)
    assert set(raw) == {it.qid for it in items}, "predictions do not cover the split exactly"
    items = items[: args.limit or None]
    adapter = None if args.adapter == "base" else mdir / "adapters" / args.adapter
    model, tok = load_model(C.MODELS[args.model], adapter)
    yes, no = tok.encode("Yes", add_special_tokens=False), tok.encode("No", add_special_tokens=False)
    assert len(yes) == 1 and len(no) == 1, (yes, no)

    out = (C.RESULTS / "smoke" if args.limit else C.RESULTS) / args.model / "features"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.adapter}__{args.domain}_{args.split}__verify.jsonl"
    with open(path, "w") as f:
        for i, it in enumerate(items):
            rec = {}
            for name, text in candidates(it, raw[it.qid]):
                kind, answer = extract(it, text)
                if kind != "answered":
                    continue
                msg = VERIFY.format(q=it.question, a=answer)
                head = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                               add_generation_prompt=True)
                with torch.no_grad():
                    logits = model(input_ids=torch.tensor([tok(head)["input_ids"]],
                                                          device=model.device)).logits[0, -1].float()
                lsm = torch.log_softmax(logits, -1)
                rec[name] = {"answer": answer, "yes_no": float(lsm[yes[0]] - lsm[no[0]]),
                             "mass": float(lsm[yes[0]].exp() + lsm[no[0]].exp())}
            f.write(json.dumps({"qid": it.qid, "verify": rec}) + "\n")
            if i % 250 == 0:
                print(f"{i + 1}/{len(items)}", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
