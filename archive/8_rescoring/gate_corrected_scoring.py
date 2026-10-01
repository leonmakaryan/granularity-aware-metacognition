"""Check the two scoring fixes in both directions: with CORRECTED_SCORING=1 the stored
answers must give the corrected numbers, with 0 the original ones. Offline, from
the saved answers.
"""

import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TOL = 1e-10


def rows(name):
    with (ROOT / "phase2_results" / name).open() as f:
        return list(csv.DictReader(f))


def boolean(x):
    assert x in ("True", "False"), x
    return x == "True"


def summarize(records):
    out = {k: statistics.mean(r[k] for r in records)
           for k in ("correct", "idk", "wrong", "strict", "lenient")}
    assert abs(out["correct"] + out["idk"] + out["wrong"] - 1) < 1e-9
    out["n"] = len(records)
    return out


def score_all():
    """Re-score every stored cell through the CURRENT src/ default path."""
    from src.config import LOCATION_RELATIONS, CORRECTED_SCORING
    from src.data import load_split
    from src.text import match_level
    from src.prompt import finest_element
    from src.eval import build_level_pool, signed_informativeness
    import src.dates as D

    test = {q.qid: q for q in load_split("test", LOCATION_RELATIONS)}
    assert len(test) == 303, len(test)
    pool = build_level_pool()
    golds = {q.qid: q.gold for q in D.load_dates_split("test")}
    assert len(golds) == 503, len(golds)

    def location(name):
        rs = rows(name)
        assert len(rs) == 303 and {int(r["qid"]) for r in rs} == set(test)
        recs = []
        for r in rs:
            q, a, abstain = test[int(r["qid"])], r["answer"], boolean(r["is_idk"])
            ml = None if abstain else match_level(a, q.hierarchy)
            sl = None if abstain else match_level(finest_element(a), q.hierarchy)
            recs.append(dict(correct=sl is not None, idk=abstain,
                             wrong=not abstain and sl is None,
                             strict=signed_informativeness(q, sl, abstain, a, pool),
                             lenient=signed_informativeness(q, ml, abstain, a, pool)))
        return summarize(recs)

    def dates(name):
        rs = rows(name)
        assert len(rs) == 503 and {r["qid"] for r in rs} == set(golds)
        recs = []
        for r in rs:
            a, abstain, g = r["answer"], boolean(r["is_idk"]), golds[r["qid"]]
            v = None if abstain else D.parse_date_text(a)
            correct = not abstain and D.credited_level(v, g, True) is not None
            recs.append(dict(correct=correct, idk=abstain,
                             wrong=not abstain and not correct,
                             strict=D.signed_date_info(v, abstain, g, True),
                             lenient=D.signed_date_info(v, abstain, g, False)))
        return summarize(recs)

    out = {"corrected": CORRECTED_SCORING, "locations": {}, "dates": {},
           "baselines": {}, "date_specialists": {}}
    corrections = json.loads((ROOT / "phase2_results" / "corrections_aug12.json").read_text())
    for domain, fn in (("locations", location), ("dates", dates)):
        for cell in corrections[domain]:
            out[domain][cell] = fn(f"transfer_7b_{cell}_on_{domain}.csv")
    for label, fname in {"plain": "eval_base_plain_7b.csv", "noidk": "eval_base_noidk_7b.csv",
                         "idk": "eval_7b_base.csv", "drag": "eval_drag_7b.csv"}.items():
        out["baselines"][label] = location(fname)
    for seed in range(4):
        out["date_specialists"][str(seed)] = dates(f"dates_sft_eval_7b_s{seed}w2.csv")
    return out


def compare(label, got, want, keys=("correct", "idk", "wrong", "strict", "lenient")):
    bad = []
    for k in keys:
        if abs(got[k] - want[k]) > TOL:
            bad.append(f"    {k}: got {got[k]:.12f} want {want[k]:.12f} "
                       f"(d={got[k]-want[k]:+.2e})")
    print(f"  {'OK  ' if not bad else 'FAIL'} {label}")
    for line in bad:
        print(line)
    return not bad


def main():
    if os.environ.get("_GATE_CHILD"):
        json.dump(score_all(), sys.stdout)
        return 0

    env = dict(os.environ, _GATE_CHILD="1")
    def run(flag):
        p = subprocess.run([sys.executable, __file__], capture_output=True, text=True,
                           env=dict(env, CORRECTED_SCORING=flag), cwd=ROOT)
        if p.returncode:
            print(p.stdout[-3000:]); print(p.stderr[-3000:]); sys.exit(1)
        return json.loads(p.stdout[p.stdout.index("{"):])

    new, old = run("1"), run("0")
    assert new["corrected"] and not old["corrected"], "flag did not take effect"

    audit = json.loads((ROOT / "writing" / "audit" / "results.json").read_text())
    corrections = json.loads((ROOT / "phase2_results" / "corrections_aug12.json").read_text())
    baselines = json.loads((ROOT / "phase2_results" / "baselines_combined_7b.json").read_text())
    ok = True

    print("\nSIDE A  CORRECTED_SCORING=1  must equal the thesis numbers "
          "(writing/audit/results.json)")
    for domain in ("locations", "dates"):
        print(f" {domain}:")
        for cell, want in audit[domain].items():
            if cell not in new[domain]:
                continue          # gate_aug12_base is an alias row, not a transfer cell
            ok &= compare(cell, new[domain][cell], want)
    print(" baselines:")
    for label, want in audit["baselines"].items():
        ok &= compare(label, new["baselines"][label], want)
    print(" date specialists:")
    for seed, want in audit["date_specialists"].items():
        ok &= compare(f"seed {seed}", new["date_specialists"][seed], want)

    print("\nSIDE B  CORRECTED_SCORING=0  must equal the ORIGINAL as-run numbers")
    for domain in ("locations", "dates"):
        print(f" {domain}:")
        for cell, exp in corrections[domain].items():
            got, want = old[domain][cell]["strict"], exp["stored"]
            good = abs(got - want) <= TOL
            ok &= good
            print(f"  {'OK  ' if good else 'FAIL'} {cell} strict "
                  f"{got:.12f} vs stored {want:.12f}")
    print(" baselines:")
    for label in ("plain", "noidk", "idk", "drag"):
        for key, col in (("info_signed_strict", "strict"), ("info_signed", "lenient")):
            got, want = old["baselines"][label][col], baselines[label][key]
            good = abs(got - want) <= TOL
            ok &= good
            print(f"  {'OK  ' if good else 'FAIL'} {label} {col} "
                  f"{got:.12f} vs stored {want:.12f}")

    print("\nGATE " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
