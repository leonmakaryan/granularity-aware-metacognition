"""One consistency check for everything the thesis reports from results/.

    python analysis/check_consistency.py

1. TRAINING: every adapter's stored records.jsonl equals a rebuild from the saved samples
   under ONE training rule: the frozen scorer with JULIAN_CREDIT off (how every adapter
   was built, at every size). Support-rule adapters and answer-or-abstain adapters alike.
2. EVALUATION: every stored prediction file regrades with 0 differences under ONE
   evaluation rule: the frozen scorer as it is (JULIAN_CREDIT on).
3. PROBES: every conflict and entity probe file regrades with 0 differences under the
   same evaluation rule (conflict answers through src/conflict.classify, which uses the
   frozen scorer's date parsing and grading).
Prints PASS or the exact failures. Needs CONFIRM=1 for the locations val files.
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

from pipeline import config as C
from pipeline import score
from pipeline.data import load
from pipeline.support import category
from pipeline.targets import build_records, support_level

FAILS = []
items_cache = {}


def items(domain, split):
    if (domain, split) not in items_cache:
        items_cache[(domain, split)] = load(domain, split)
    return items_cache[(domain, split)]


def samples(model, domain, split):
    p = C.RESULTS / model / "samples" / f"{domain}_{split}.jsonl"
    if not p.exists():
        return {}
    raw = {str(r["qid"]): r["raws"] for r in map(json.loads, open(p))}
    return {it.qid: raw[str(it.qid)] for it in items(domain, split) if str(it.qid) in raw}


# 1. training records
score.JULIAN_CREDIT = False
n_train = 0
for m in ("7b", "1p5b", "3b"):
    rebuilt = {}
    for rule in ("support", "binary"):
        for d in ("locations", "dates"):
            recs = build_records(d, items(d, "train"), samples(m, d, "train"), rule=rule)[0]
            rebuilt[(rule, d)] = {str(r["qid"]): r["completion"][0]["content"] for r in recs}
    for ad in sorted((C.RESULTS / m / "adapters").iterdir()):
        name = ad.name
        if any(t in name for t in ("pilot", "fit", "draft")):   # development adapters, not reported
            continue
        rule = "binary" if name.endswith("_binary") else "support"
        stored = [json.loads(l) for l in open(ad / "records.jsonl")]
        for d in {r["domain"] for r in stored}:
            st = {str(r["qid"]): r["completion"][0]["content"] for r in stored if r["domain"] == d}
            if st != rebuilt[(rule, d)]:
                FAILS.append(f"train {m}/{name}/{d}: records differ from the one training rule")
            n_train += 1
score.JULIAN_CREDIT = True

# 2. predictions
nz = lambda x: None if (x == "" or (isinstance(x, float) and math.isnan(x))) else float(x)
n_pred = 0
for m in ("7b", "1p5b", "3b"):
    for path in sorted((C.RESULTS / m / "preds").glob("*.csv")):
        adapter, ds, mode = path.stem.split("__")
        domain, tag = ds.split("_", 1)
        split = tag.split("-")[0]
        its = {str(it.qid): it for it in items(domain, split)}
        smp = samples(m, domain, split)
        bad = 0
        for r in pd.read_csv(path, dtype={"qid": str}, keep_default_na=False).to_dict("records"):
            it = its[r["qid"]]
            g = score.grade(it, r["raw"], mode)
            sup = support_level(it, smp[it.qid]) if it.qid in smp else None
            cat = category(sup, g) if it.qid in smp else ""
            bad += (g["kind"] != r["kind"] or str(g["correct"]) != str(r["correct"])
                    or abs(g["score"] - float(r["score"])) > 1e-9
                    or (sup if sup is None else float(sup)) != nz(r["support"]) or cat != r["category"])
        if bad:
            FAILS.append(f"eval {m}/{path.name}: {bad} rows differ from the one evaluation rule")
        n_pred += 1

# 3. probes: conflict answers regrade with the one scorer (via src/conflict.classify,
#    which reads and grades dates with pipeline/score.py); entity answers with grade().
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rescore_probes import COLS, regrade  # noqa: E402
from src.conflict import build_cases  # noqa: E402
from src.dates import load_dates_split  # noqa: E402

cases = build_cases(load_dates_split("test", historical_bce=True), seed=C.GLOBAL_SEED)
ent = {str(it.qid): it for it in items("entity", "test")}
n_probe = 0
for m in ("7b", "1p5b"):
    for path in sorted((C.RESULTS / m / "probes").glob("conflict__*.csv")):
        df = pd.read_csv(path, keep_default_na=False)
        bad = 0
        for case, r in zip(cases, df.to_dict("records")):
            new = regrade(case, r["raw"])
            bad += (case["qid"] != r["qid"]
                    or any(str(new[k]) != str(r[k]) for k in COLS if k != "signed_strict")
                    or abs(float(new["signed_strict"]) - float(r["signed_strict"])) > 1e-9)
        if bad or len(df) != len(cases):
            FAILS.append(f"probe {m}/{path.name}: {bad} rows differ from the one evaluation rule")
        n_probe += 1
    for path in sorted((C.RESULTS / m / "probes").glob("entity__*.csv")):
        df = pd.read_csv(path, dtype={"qid": str}, keep_default_na=False)
        bad = 0
        for r in df.to_dict("records"):
            g = score.grade(ent[r["qid"]], r["raw"])
            bad += (g["kind"] != r["kind"] or str(g["correct"]) != str(r["correct"])
                    or abs(g["score"] - float(r["score"])) > 1e-9)
        if bad:
            FAILS.append(f"probe {m}/{path.name}: {bad} rows differ from the one evaluation rule")
        n_probe += 1

print(f"checked {n_train} adapter training sets, {n_pred} prediction files and "
      f"{n_probe} probe files")
print("PASS" if not FAILS else "FAIL\n" + "\n".join(FAILS))
sys.exit(1 if FAILS else 0)
