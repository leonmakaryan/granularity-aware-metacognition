"""Agreement and Cohen's kappa between the matcher and the human labels of
build_matcher_study.py, overall and per tier, with every disagreement listed.
"""
import sys
from collections import Counter

import pandas as pd

from src.config import OUT_DIR

SHEET = OUT_DIR / "matcher_study_sheet.csv"


def _lab(v) -> str:
    """Normalise a level cell to a comparable string label."""
    s = str(v).strip().lower()
    if s in ("", "nan", "none", "no match", "no-match"):
        return "none"
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def main():
    if not SHEET.exists():
        sys.exit(f"{SHEET} not found -- run build_matcher_study.py first")
    df = pd.read_csv(SHEET)

    filled = df[df["human_level"].astype(str).str.strip().ne("")
                & df["human_level"].notna()].copy()
    if filled.empty:
        print(f"{SHEET} has no human_level entries filled in yet.")
        print(f"{len(df)} rows are waiting for annotation.")
        return

    filled["m"] = filled["matcher_level"].map(_lab)
    filled["h"] = filled["human_level"].map(_lab)
    agree = (filled["m"] == filled["h"])

    print(f"scored {len(filled)}/{len(df)} rows")
    print(f"raw agreement : {agree.mean():.3f}")
    print(f"Cohen's kappa : {cohen_kappa(list(filled['m']), list(filled['h'])):.3f}")

    print("\nby tier:")
    for tier, sub in filled.groupby("tier"):
        a = (sub["m"] == sub["h"])
        print(f"  {tier:14s} n={len(sub):3d}  agreement={a.mean():.3f}")

    dis = filled[~agree]
    if dis.empty:
        print("\nno disagreements")
        return
    print(f"\n{len(dis)} disagreement(s):")
    for _, r in dis.iterrows():
        print(f"\n  [{r['tier']}] {r['question']}")
        print(f"    gold     : {r['gold_hierarchy']}")
        print(f"    answer   : {r['model_answer']}")
        print(f"    matcher  : {r['matcher_says']}")
        print(f"    human    : {r['human_level']}")
        if str(r.get("note") or "").strip() not in ("", "nan"):
            print(f"    note     : {r['note']}")


if __name__ == "__main__":
    main()
