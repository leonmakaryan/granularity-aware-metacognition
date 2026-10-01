"""Read the model on its own greedy answers: hidden states, token log-probabilities,
and the log-probability of each coarser rendering and of abstaining. Never reads a
reference answer. (Belonged to pipeline/.)
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from src.dates import DATE_LADDER

from . import config as C
from .data import load
from .generate import load_model
from .score import extract, ladder_index, parse_date, prompt_for
from .targets import ABSTAIN

LAYERS = (8, 12, 16, 20, 24, 28)
END = "<|im_end|>"


def candidates(item, raw: str) -> list[tuple[str, str]]:
    out = [("greedy", raw)]
    kind, text = extract(item, raw)
    if item.domain == "dates" and kind == "answered":
        v = parse_date(text)
        if v is not None:
            for L in range(ladder_index(v.level), len(DATE_LADDER)):
                try:
                    out.append((DATE_LADDER[L].name,
                                "Date: " + v.to_level(DATE_LADDER[L]).to_text()))
                except (ValueError, KeyError):
                    pass
    out.append(("abstain", ABSTAIN[item.domain]))
    return out


def encode(tok, prompt: str, completion: str) -> tuple[list[int], list[int]]:
    head = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)
    p = tok(head)["input_ids"]
    f = tok(head + completion + END)["input_ids"]
    assert f[:len(p)] == p, "prompt tokens are not a prefix of prompt + completion"
    return p, f


def read(model, pair, hidden: bool) -> dict:
    """One sequence: completion token log-probabilities (with <|im_end|>), optionally the
    hidden states at the three positions."""
    p, f = pair
    n_p, n_f = len(p), len(f)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([f], device=model.device),
                    output_hidden_states=hidden)
    logits = out.logits[0, n_p - 1:n_f - 1].float()
    tgt = torch.tensor(f[n_p:n_f], device=logits.device)
    row = {"lp": torch.log_softmax(logits, -1).gather(1, tgt[:, None])[:, 0].tolist()}
    if hidden:
        hs = torch.stack([out.hidden_states[L][0] for L in LAYERS])     # [layers, T, H]
        last = max(n_f - 2, n_p - 1)          # an empty completion falls back to the prompt
        row["h_prompt"] = hs[:, n_p - 1].float().cpu().numpy()
        row["h_last"] = hs[:, last].float().cpu().numpy()
        row["h_mean"] = hs[:, n_p - 1:last + 1].float().mean(1).cpu().numpy()
    return row


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

    feats = {k: [] for k in ("h_prompt", "h_last", "h_mean")}
    meta = []
    for i, it in enumerate(items):
        row = read(model, encode(tok, prompt_for(it), raw[it.qid]), hidden=True)
        for k in feats:
            feats[k].append(row[k].astype(np.float16))
        cands = []
        for name, text in candidates(it, raw[it.qid]):
            lp = read(model, encode(tok, prompt_for(it), text), hidden=False)["lp"]
            cands.append({"name": name, "text": text, "lp": float(sum(lp)), "n": len(lp)})
        meta.append({"greedy_lp": row["lp"], "candidates": cands})
        if i % 250 == 0:
            print(f"{i + 1}/{len(items)}", flush=True)

    out = (C.RESULTS / "smoke" if args.limit else C.RESULTS) / args.model / "features"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.adapter}__{args.domain}_{args.split}"
    np.savez(out / f"{stem}.npz", qid=np.array([it.qid for it in items]), layers=np.array(LAYERS),
             **{k: np.stack(v) for k, v in feats.items()})
    with open(out / f"{stem}.jsonl", "w") as f:
        for it, m in zip(items, meta):
            f.write(json.dumps({"qid": it.qid, "raw": raw[it.qid], **m}) + "\n")
    print(f"wrote {out / stem}.npz/.jsonl", flush=True)


if __name__ == "__main__":
    main()
