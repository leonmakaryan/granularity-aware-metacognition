"""Ask the location question at country level, to see if the model knows the country
when its town answer is wrong. (Belonged to pipeline/.)
"""
import argparse
import json

from . import config as C
from .data import load
from .generate import generate, load_model
from .score import prompt_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7b", choices=list(C.MODELS))
    ap.add_argument("--domain", default="locations", choices=["locations"])
    ap.add_argument("--split", required=True, choices=["train", "test", "val"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    items = load(args.domain, args.split)[: args.limit or None]
    model, tok = load_model(C.MODELS[args.model])
    prompts = [prompt_for(it, "country") for it in items]
    gate = json.loads((C.RESULTS / args.model / "gates" / "gate4_batching.json").read_text())
    greedy = generate(model, tok, prompts, sample=False, batch_size=gate["canonical_batch"],
                      log=print)
    raws = generate(model, tok, prompts, sample=True, batch_size=C.SAMPLE_BATCH,
                    tag=f"{args.domain}|{args.split}|country", log=print)
    out = (C.RESULTS / "smoke" if args.limit else C.RESULTS) / args.model / "samples"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.domain}_{args.split}__country.jsonl"
    with open(path, "w") as f:
        for it, g, rs in zip(items, greedy, raws):
            f.write(json.dumps({"qid": it.qid, "greedy": g[0], "raws": rs}) + "\n")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
