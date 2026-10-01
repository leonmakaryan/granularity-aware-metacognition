"""Rescore every 7B evaluation under the two scoring fixes, from the saved answers:
an exact match beats a looser one at a finer level ("Austria" is not credited at
"Lower Austria"), and a bare year before 1000 now parses ("912").
"""
import glob
import json
import re

import pandas as pd

from src.config import LOCATION_RELATIONS, OUT_DIR
from src.data import load_split
from src.eval import build_level_pool, signed_informativeness
from src.prompt import finest_element
from src.text import find_match_level_v3
import src.dates as D
from ugc.parser.temporal_value import TemporalValue
from ugc.parser.temporal_enums import TemporalLevel

BARE_YEAR = re.compile(r"^\s*(\d{1,3})\s*(AD|CE)?\s*$", re.I)


def rescore_locations(csv_path, test_by_qid, pool):
    d = pd.read_csv(csv_path)
    if "qid" not in d.columns or "answer" not in d.columns:
        return None
    tot_len = tot_str = 0.0
    n = changed = 0
    for _, r in d.iterrows():
        q = test_by_qid.get(r["qid"])
        if q is None:
            continue
        n += 1
        ans = "" if pd.isna(r["answer"]) else str(r["answer"])
        idk = bool(r["is_idk"])
        ml = None if (idk or not ans) else find_match_level_v3(ans, q.hierarchy)
        sl = None if (idk or not ans) else find_match_level_v3(finest_element(ans), q.hierarchy)
        old_ml = None if pd.isna(r.get("match_level")) else int(r["match_level"])
        if ml != old_ml:
            changed += 1
        tot_len += signed_informativeness(q, ml, idk, ans, pool)
        tot_str += signed_informativeness(q, sl, idk, ans, pool)
    return None if not n else {"n": n, "changed": changed,
                               "signed": tot_len / n, "signed_strict": tot_str / n}


def rescore_dates(csv_path, golds):
    d = pd.read_csv(csv_path)
    if "qid" not in d.columns or "answer" not in d.columns:
        return None
    tot = 0.0
    n = fixed = 0
    for _, r in d.iterrows():
        g = golds.get(r["qid"])
        if g is None:
            continue
        n += 1
        ans = "" if pd.isna(r["answer"]) else str(r["answer"])
        idk = bool(r["is_idk"])
        if idk:
            continue
        v = D._PARSER.first_most_specific(ans) if ans else None
        if v is None and ans:
            # The parser ignores a bare number below 1000 without context; give it some.
            v = D._PARSER.first_most_specific("in " + ans)
            if v is not None:
                fixed += 1
        tot += D.signed_date_info(v, False, g, strict=True)
    return None if not n else {"n": n, "fixed": fixed, "signed_strict": tot / n}


def main():
    test_by_qid = {q.qid: q for q in load_split("test", LOCATION_RELATIONS)}
    pool = build_level_pool()
    golds = {q.qid: q.gold for q in D.load_dates_split("test")}

    out = {"locations": {}, "dates": {}}

    print("=== LOCATIONS: matcher v3 (strict signed) ===")
    print(f"{'cell':44s} {'stored':>8s} {'fixed':>8s} {'delta':>7s} {'chg':>4s}")
    for f in sorted(glob.glob("phase2_results/transfer_7b_*_on_locations.csv")):
        jf = f.replace(".csv", ".json")
        try:
            stored = json.load(open(jf))["metrics"]
        except Exception:
            continue
        r = rescore_locations(f, test_by_qid, pool)
        if not r:
            continue
        name = f.split("/")[-1][len("transfer_7b_"):-len("_on_locations.csv")]
        old = stored["info_signed_strict"]
        out["locations"][name] = {"stored": old, "fixed": r["signed_strict"],
                                  "delta": r["signed_strict"] - old, "changed": r["changed"]}
        print(f"{name:44s} {old:+8.4f} {r['signed_strict']:+8.4f} "
              f"{r['signed_strict']-old:+7.4f} {r['changed']:4d}")

    print("\n=== DATES: bare-short-year fallback (strict signed) ===")
    print(f"{'cell':44s} {'stored':>8s} {'fixed':>8s} {'delta':>7s} {'fix':>4s}")
    for f in sorted(glob.glob("phase2_results/transfer_7b_*_on_dates.csv")):
        jf = f.replace(".csv", ".json")
        try:
            stored = json.load(open(jf))["metrics"]
        except Exception:
            continue
        r = rescore_dates(f, golds)
        if not r:
            continue
        name = f.split("/")[-1][len("transfer_7b_"):-len("_on_dates.csv")]
        old = stored["info_signed_strict"]
        out["dates"][name] = {"stored": old, "fixed": r["signed_strict"],
                              "delta": r["signed_strict"] - old, "n_fixed": r["fixed"]}
        print(f"{name:44s} {old:+8.4f} {r['signed_strict']:+8.4f} "
              f"{r['signed_strict']-old:+7.4f} {r['fixed']:4d}")

    with open(OUT_DIR / "corrections_aug12.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT_DIR / 'corrections_aug12.json'}")


if __name__ == "__main__":
    main()
