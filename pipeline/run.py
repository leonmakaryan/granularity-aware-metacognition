"""Command line for every stage. One training per process (TRL constraint).

    python -m pipeline.run sample  --model 7b --domain locations --split train
    python -m pipeline.run train   --model 7b --condition joint --seed 1
    python -m pipeline.run eval    --model 7b --adapter joint_s1 --domain dates --split test
    python -m pipeline.run report  --model 7b --split dev

`--limit N` writes to results/smoke/ and never touches real outputs.

Gates are checks that had to pass before the next stage ran: 1 offline scorer tests
(checks.py), 2 the scorer check against hand labels (scorer_check.py), 3 reproducing
answers stored by the earlier code, 4 the evaluation batch size, 5 the training pilot.
"""
import argparse
import json
import os
import random
import sys
import time

import pandas as pd

from . import config as C


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mdir(args):
    return (C.RESULTS / "smoke" if getattr(args, "limit", None) else C.RESULTS) / args.model


def samples_path(args, domain, split):
    return mdir(args) / "samples" / f"{domain}_{split}.jsonl"


def read_samples(path) -> dict:
    with open(path) as f:
        return {r["qid"]: r["raws"] for r in map(json.loads, f)}


def eval_batch(args) -> int:
    """Evaluation batch size, fixed by Gate 4 for full runs."""
    if args.limit:
        return C.SMOKE_EVAL_BATCH
    gate = C.RESULTS / args.model / "gates" / "gate4_batching.json"
    if not gate.exists():
        sys.exit(f"run Gate 4 first: {gate} is missing")
    return json.loads(gate.read_text())["canonical_batch"]


PILOT_CANDIDATES = ("pilot_locations_s0", "pilot_locations_s0_e1")   # fixed before the second ran


def chosen_epochs(args) -> int:
    """Fixed by the pilot selection for full runs; one recipe at every scale."""
    if args.limit:
        return C.EPOCHS
    sel = C.RESULTS / "7b" / "gates" / "pilot_selection.json"
    if not sel.exists():
        sys.exit(f"run the pilot selection first: {sel} is missing")
    return int(json.loads(sel.read_text())["chosen"]["epochs"])


def adapter_dir(args):
    if args.adapter == "base":
        return None
    path = mdir(args) / "adapters" / args.adapter
    if not (path / "adapter_config.json").exists():
        sys.exit(f"missing adapter {path}")
    return path


def cmd_sample(args):
    from .data import load
    from .generate import generate, load_model
    from .score import prompt_for

    items = load(args.domain, args.split)[: args.limit or None]
    path = samples_path(args, args.domain, args.split)
    path.parent.mkdir(parents=True, exist_ok=True)
    bs = C.SAMPLE_BATCH
    kept = []
    if path.exists():   # resume: keep only whole, parseable batches
        for line in path.read_text().splitlines(keepends=True):
            try:
                json.loads(line)
                if line.endswith("\n"):
                    kept.append(line)
            except json.JSONDecodeError:
                break
        kept = kept[: len(kept) // bs * bs]
        path.write_text("".join(kept))
    start = len(kept) // bs
    n_batches = (len(items) + bs - 1) // bs
    if start >= n_batches:
        log(f"{path.name} already complete")
        return
    model, tok = load_model(C.MODELS[args.model])
    with open(path, "a") as f:
        for b in range(start, n_batches):
            chunk = items[b * bs:(b + 1) * bs]
            outs = generate(model, tok, [prompt_for(it) for it in chunk], sample=True,
                            batch_size=bs, tag=f"{args.domain}|{args.split}", first_batch=b)
            for it, raws in zip(chunk, outs):
                f.write(json.dumps({"qid": it.qid, "raws": raws}) + "\n")
            f.flush()
            if b % 10 == 0 or b == n_batches - 1:
                log(f"sample {args.domain}/{args.split}: {min((b + 1) * bs, len(items))}/{len(items)}")
    log(f"wrote {path}")


def cmd_train(args):
    from .data import holdout, load
    from .targets import build_records
    from .train import train

    if args.pilot and args.condition != "locations":
        sys.exit("the pilot trains on locations only")
    if args.epochs and not args.pilot:
        sys.exit("--epochs is for pilot variants only; full runs use the pilot selection")
    if args.pilot and args.part == "fit":
        sys.exit("the pilot has its own holdout; --part fit is for the confirmation protocol")
    if args.no_julian_credit:
        # Build targets without the Julian calendar credit, as for all reported adapters.
        from . import score
        score.JULIAN_CREDIT = False
    domains = list(C.DOMAINS) if args.condition == "joint" else [args.condition]
    records, stats = [], {}
    for d in domains:
        items = load(d, "train")
        if args.pilot:
            items = holdout(items, C.HOLDOUT_FRAC, C.GLOBAL_SEED)[0]
        if args.part == "fit":     # leave the confirmation slice untouched
            items = holdout(items, C.CONFIRM_FRAC, C.CONFIRM_SEED)[0]
        items = items[: args.limit or None]
        samples = read_samples(samples_path(args, d, "train"))
        missing = [it.qid for it in items if it.qid not in samples]
        if missing:
            sys.exit(f"{d}: {len(missing)} training questions have no samples")
        drafts = None
        if args.draft:      # always the stored base greedy answers, smoke mode included
            p = C.RESULTS / args.model / "preds" / f"base__{d}_train__deploy.csv"
            df = pd.read_csv(p, dtype={"qid": str}, keep_default_na=False)
            drafts = dict(zip(df.qid, df.raw))
        r, s = build_records(d, items, samples, drafts, rule=args.rule)
        records += r
        stats[d] = s
    random.Random(args.seed).shuffle(records)
    if args.pilot:
        epochs = args.epochs or C.EPOCHS
        name = f"pilot_locations_s{args.seed}" + (f"_e{args.epochs}" if args.epochs else "")
    else:
        epochs = chosen_epochs(args)
        name = (f"{args.condition}_s{args.seed}" + ("_draft" if args.draft else "")
                + ("_binary" if args.rule == "binary" else "")
                + ("_fit" if args.part == "fit" else ""))
    out = mdir(args) / "adapters" / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (out / "record_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (out / "train_args.json").write_text(json.dumps(dict(
        model=C.MODELS[args.model], condition=args.condition, seed=args.seed, epochs=epochs,
        pilot=args.pilot, records=len(records)), indent=2) + "\n")
    log(f"{name}: {len(records)} records, {epochs} epochs {stats}")
    history = train(C.MODELS[args.model], records, out, args.seed, epochs,
                    max_steps=3 if args.limit else None)
    (out / "train_log.json").write_text(json.dumps(history, indent=2) + "\n")
    log(f"saved {out}")


def cmd_eval(args):
    from .data import holdout, load
    from .generate import generate, load_model
    from .score import grade, prompt_for
    from .support import category
    from .targets import support_level

    items = load(args.domain, args.split)
    tag = args.split
    if args.holdout:
        items = holdout(items, C.HOLDOUT_FRAC, C.GLOBAL_SEED)[1]
        tag = f"{args.split}-holdout"
    if args.part != "all":
        if args.part == "confirm" and os.environ.get("CONFIRM_SLICE") != "1":
            sys.exit("the confirmation slice is read only after the freeze (CONFIRM_SLICE=1)")
        items = holdout(items, C.CONFIRM_FRAC, C.CONFIRM_SEED)[0 if args.part == "fit" else 1]
        tag = f"{args.split}-{args.part}"
    items = items[: args.limit or None]
    sp = samples_path(args, args.domain, args.split)
    samples = read_samples(sp) if sp.exists() else {}
    model, tok = load_model(C.MODELS[args.model], adapter_dir(args))
    if args.rep_penalty:
        # Sensitivity check only: the draft format has to repeat its own guess on the
        # final line, which the shipped penalty charges for. Any run using this is
        # reported next to the shipped-decoding run, never instead of it.
        model.pipeline_repetition_penalty = args.rep_penalty
        tag = f"{tag}-rp{args.rep_penalty}"
    outs = generate(model, tok, [prompt_for(it, args.mode) for it in items],
                    sample=False, batch_size=eval_batch(args), log=log,
                    max_new_tokens=C.DRAFT_MAX_NEW_TOKENS if args.draft else None)
    rows = []
    for it, o in zip(items, outs):
        g = grade(it, o[0], args.mode)
        has = it.qid in samples
        sup = support_level(it, samples[it.qid]) if has else None
        cat = category(sup, g) if has else ""
        rows.append(dict(qid=it.qid, unit=it.unit, relation=it.relation, question=it.question,
                         raw=o[0], kind=g["kind"], text=g["text"], correct=g["correct"],
                         level=g["level"], score=g["score"], has_support=has, support=sup,
                         category=cat, exact=float(cat == "exact_support_match")))
    out = mdir(args) / "preds" / f"{args.adapter}__{args.domain}_{tag}__{args.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    log(f"{out.name}: n={len(df)} score={df.score.mean():+.4f} correct={df.correct.mean():.3f} "
        f"abstain={(df.kind == 'abstained').mean():.3f} fmt_fail={(df.kind == 'fmt_fail').sum()} "
        f"support_match={df.exact.mean():.3f} (support known for {int(df.has_support.sum())})")


def cmd_rescore(args):
    """Regrade every stored prediction file from its raw outputs with the current scorer.
    The raw outputs are never changed; kind, grade, support and category are recomputed
    exactly as cmd_eval computes them. Files on the confirmation split need CONFIRM=1."""
    from .data import load
    from .score import grade
    from .support import category
    from .targets import support_level

    cache = {}
    for path in sorted((mdir(args) / "preds").glob("*.csv")):
        adapter, ds, mode = path.stem.split("__")
        domain, tag = ds.split("_", 1)
        split = tag.split("-")[0]
        if (domain, split) not in cache:
            sp = samples_path(args, domain, split)
            cache[(domain, split)] = ({str(it.qid): it for it in load(domain, split)},
                                      read_samples(sp) if sp.exists() else {})
        items, samples = cache[(domain, split)]
        old = pd.read_csv(path, dtype={"qid": str, "unit": str}, keep_default_na=False)
        rows = []
        for r in old.to_dict("records"):
            it = items[r["qid"]]
            g = grade(it, r["raw"], mode)
            has = it.qid in samples
            sup = support_level(it, samples[it.qid]) if has else None
            cat = category(sup, g) if has else ""
            rows.append(dict(qid=it.qid, unit=it.unit, relation=it.relation, question=it.question,
                             raw=r["raw"], kind=g["kind"], text=g["text"], correct=g["correct"],
                             level=g["level"], score=g["score"], has_support=has, support=sup,
                             category=cat, exact=float(cat == "exact_support_match")))
        df = pd.DataFrame(rows)
        assert list(df.raw) == list(old.raw), path.name
        changed = int((df.score.round(12).astype(str) != old.score.astype(float).round(12).astype(str)).sum())
        df.to_csv(path, index=False)
        log(f"{path.name}: {changed} of {len(df)} scores changed, "
            f"score {old.score.astype(float).mean():+.4f} -> {df.score.mean():+.4f}")


def cmd_gate_repro(args):
    """Gate 3: the 7B base (batch 1, fp16) must reproduce the answers stored by the earlier code."""
    import csv
    from .data import load
    from .generate import generate, load_model
    from .score import prompt_for

    items = load("locations", "test")
    stored = {r["qid"]: r["raw"] for r in
              csv.DictReader(open(C.ROOT / "archive" / "phase2_results" / "transfer_7b_base_on_locations.csv"))}
    model, tok = load_model(C.MODELS["7b"], dtype="float16")
    outs = generate(model, tok, [prompt_for(it) for it in items], sample=False,
                    batch_size=1, log=log)
    diffs = [dict(qid=it.qid, stored=stored[it.qid], new=o[0])
             for it, o in zip(items, outs) if o[0] != stored[it.qid]]
    result = dict(n=len(items), identical=len(items) - len(diffs), diffs=diffs[:20],
                  passed=not diffs)
    gdir = C.RESULTS / "7b" / "gates"
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "gate3_reproduction.json").write_text(json.dumps(result, indent=2) + "\n")
    log(f"GATE 3: {result['identical']}/{len(items)} identical -> "
        f"{'PASSED' if result['passed'] else 'FAILED'}")
    sys.exit(0 if result["passed"] else 1)


def cmd_gate_batch(args):
    """Gate 4: batch 1 against a larger batch on locations test (rule in config)."""
    from .data import load
    from .generate import generate, load_model
    from .score import grade, prompt_for

    items = load("locations", "test")[: C.GATE4_QUESTIONS]
    model, tok = load_model(C.MODELS[args.model])
    prompts = [prompt_for(it) for it in items]
    one = generate(model, tok, prompts, sample=False, batch_size=1)
    many = generate(model, tok, prompts, sample=False, batch_size=C.SMOKE_EVAL_BATCH)

    def outcome(it, raw):
        g = grade(it, raw)
        return g["kind"], g["correct"], g["level"]
    diffs = [it.qid for it, a, b in zip(items, one, many)
             if outcome(it, a[0]) != outcome(it, b[0])]
    text_diffs = sum(a[0] != b[0] for a, b in zip(one, many))
    batched_ok = len(diffs) <= C.GATE4_MAX_DIFFS
    result = dict(n=len(items), scorer_outcome_diffs=len(diffs), text_diffs=text_diffs,
                  diff_qids=diffs, rule=f"batched canonical if <= {C.GATE4_MAX_DIFFS}",
                  canonical_batch=C.SMOKE_EVAL_BATCH if batched_ok else 1)
    gdir = C.RESULTS / args.model / "gates"
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "gate4_batching.json").write_text(json.dumps(result, indent=2) + "\n")
    log(f"GATE 4: {len(diffs)}/{len(items)} scorer-outcome differences, {text_diffs} text "
        f"differences -> canonical evaluation batch {result['canonical_batch']}")


def cmd_pilot_check(args):
    """Gate 5: does the pilot adapter beat the base on the held-out 10% of locations train?"""
    d = mdir(args) / "preds"
    base = pd.read_csv(d / "base__locations_train-holdout__deploy.csv")
    pilot = pd.read_csv(d / f"{args.adapter}__locations_train-holdout__deploy.csv")
    target_abstain = float(base["support"].isna().mean())
    pilot_abstain = float((pilot.kind == "abstained").mean())
    result = dict(n=len(pilot), base_match=float(base.exact.mean()),
                  pilot_match=float(pilot.exact.mean()), target_abstain_share=target_abstain,
                  pilot_abstain=pilot_abstain, base_abstain=float((base.kind == "abstained").mean()),
                  pilot_score=float(pilot.score.mean()), base_score=float(base.score.mean()))
    result["go"] = (result["pilot_match"] > result["base_match"]
                    and abs(pilot_abstain - target_abstain) <= 0.10)
    (mdir(args) / "gates").mkdir(parents=True, exist_ok=True)
    (mdir(args) / "gates" / f"gate5_{args.adapter}.json").write_text(json.dumps(result, indent=2) + "\n")
    log("GATE 5: " + json.dumps(result))
    log("GO" if result["go"] else "NO-GO")
    sys.exit(0 if result["go"] else 1)


def cmd_pilot_select(args):
    """Choose the training epochs by the rule fixed before the 1-epoch pilot ran: higher
    estimated-support match on the 10% train holdout, ties broken by signed score."""
    rows = []
    for name in PILOT_CANDIDATES:
        df = pd.read_csv(mdir(args) / "preds" / f"{name}__locations_train-holdout__deploy.csv")
        log_ = json.loads((mdir(args) / "adapters" / name / "train_log.json").read_text())
        epochs = int(round(max(e["epoch"] for e in log_ if "epoch" in e)))
        rows.append(dict(adapter=name, epochs=epochs, support_match=float(df.exact.mean()),
                         score=float(df.score.mean()),
                         abstain=float((df.kind == "abstained").mean())))
    chosen = max(rows, key=lambda r: (r["support_match"], r["score"]))
    result = dict(rule="higher support match on the 10% train holdout, ties by signed score; "
                       "fixed before the 1-epoch pilot ran", candidates=rows, chosen=chosen)
    gdir = mdir(args) / "gates"
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "pilot_selection.json").write_text(json.dumps(result, indent=2) + "\n")
    log("PILOT SELECTION: " + json.dumps(result))


def cmd_probe(args):
    from .generate import load_model
    from .probes import run_conflict, run_entity

    model, tok = load_model(C.MODELS[args.model], adapter_dir(args))
    out = mdir(args) / "probes"
    out.mkdir(parents=True, exist_ok=True)
    fn = run_conflict if args.probe == "conflict" else run_entity
    summary = fn(model, tok, out / f"{args.probe}__{args.adapter}.csv", eval_batch(args),
                 limit=args.limit, log=log)
    (out / f"{args.probe}__{args.adapter}.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(json.dumps(summary)[:600])


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, *extra):
        s = sub.add_parser(name)
        s.add_argument("--model", default="7b", choices=list(C.MODELS))
        s.add_argument("--limit", type=int, default=None)
        for e in extra:
            e(s)
        s.set_defaults(fn=fn)

    domain = lambda s: s.add_argument("--domain", required=True, choices=["locations", "dates"])
    split = lambda s: s.add_argument("--split", required=True, choices=["train", "test", "val"])
    add("sample", cmd_sample, domain, split)
    draft = lambda s: s.add_argument("--draft", action="store_true")
    add("train", cmd_train,
        lambda s: s.add_argument("--condition", required=True, choices=list(C.CONDITIONS)),
        lambda s: s.add_argument("--seed", type=int, required=True),
        lambda s: s.add_argument("--pilot", action="store_true"),
        lambda s: s.add_argument("--epochs", type=int, default=None), draft,
        lambda s: s.add_argument("--part", default="all", choices=["all", "fit"]),
        lambda s: s.add_argument("--rule", default="support", choices=["support", "binary"]),
        lambda s: s.add_argument("--no-julian-credit", action="store_true"))
    add("eval", cmd_eval, domain, split,
        lambda s: s.add_argument("--adapter", required=True),
        lambda s: s.add_argument("--mode", default="deploy", choices=["deploy", "noidk", "bare"]),
        lambda s: s.add_argument("--holdout", action="store_true"), draft,
        lambda s: s.add_argument("--part", default="all", choices=["all", "fit", "confirm"]),
        lambda s: s.add_argument("--rep-penalty", type=float, default=None))
    add("rescore", cmd_rescore)
    add("gate-repro", cmd_gate_repro)
    add("gate-batch", cmd_gate_batch)
    add("pilot-check", cmd_pilot_check,
        lambda s: s.add_argument("--adapter", default="pilot_locations_s0"))
    add("pilot-select", cmd_pilot_select)
    add("probe", cmd_probe, lambda s: s.add_argument("--adapter", required=True),
        lambda s: s.add_argument("--probe", required=True, choices=["conflict", "entity"]))
    add("report", lambda a: __import__("pipeline.report", fromlist=["main"]).main(a.model, a.split),
        lambda s: s.add_argument("--split", required=True, choices=["dev", "val"]))
    add("scorer-check-build", lambda a: __import__("pipeline.scorer_check", fromlist=["build"]).build())
    add("scorer-check-score", lambda a: __import__("pipeline.scorer_check", fromlist=["score"]).score())
    add("scorer-check-rekey", lambda a: __import__("pipeline.scorer_check", fromlist=["rekey"]).rekey())
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
