"""Sheet for a 50-answer human check of the place matcher.

Stratified towards the cases where the matcher is most likely wrong: the looser
matching tiers, every matched level and unmatched answers, several models.
Scored by score_matcher_study.py. Superseded by the 162-answer scorer check of the
final pipeline.
"""
import random
from pathlib import Path

import pandas as pd

from src.config import OUT_DIR, LOCATION_RELATIONS
from src.data import load_split
from src.text import normalize, match_at_level, find_match_level, content_tokens

SEED = 0
N_TOTAL = 50
SHEET = OUT_DIR / "matcher_study_sheet.csv"

# Per-question eval CSVs to draw from (answer style differs by model/adapter).
SOURCES = [
    "transfer_7b_base_on_locations.csv",
    "transfer_7b_sft_loc_7b_v2_on_locations.csv",
    "transfer_7b_dates_sft_7b_s0w2_on_locations.csv",
    "transfer_1p5b_base_on_locations.csv",
    "transfer_1p5b_sft_v2_loc_on_locations.csv",
]


def tier_of(answer: str, entry: str) -> str:
    """Which matching tier accepted the pair, in the same order as src.text.match_at_level."""
    na, nh = normalize(answer), normalize(entry)
    if not na or not nh:
        return "none"
    if na == nh:
        return "exact"
    if f" {nh} " in f" {na} " or f" {na} " in f" {nh} ":
        return "substring"
    h_tok, a_tok = content_tokens(entry), content_tokens(answer)
    if h_tok and h_tok.issubset(a_tok):
        return "content-word"
    return "none"


def main():
    rng = random.Random(SEED)
    golds = {q.qid: q for q in load_split("test", LOCATION_RELATIONS)}

    rows = []
    for src in SOURCES:
        p = OUT_DIR / src
        if not p.exists():
            print(f"  skip missing {src}")
            continue
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            q = golds.get(r["qid"])
            if q is None or bool(r.get("is_idk")):
                continue
            ans = str(r.get("answer") or "").strip()
            if not ans:
                continue
            ml = find_match_level(ans, q.hierarchy)
            tier = tier_of(ans, q.hierarchy[ml]) if ml is not None else "none"
            rows.append({
                "source": src.replace("transfer_", "").replace("_on_locations.csv", ""),
                "qid": r["qid"],
                "question": q.question,
                "gold_hierarchy": " > ".join(q.hierarchy),
                "model_answer": ans,
                "matcher_level": "" if ml is None else ml,
                "matcher_says": "no match" if ml is None else f"{ml}: {q.hierarchy[ml]}",
                "tier": tier,
            })

    if not rows:
        print("no rows; are the eval CSVs present in phase2_results/?")
        return
    pool = pd.DataFrame(rows).drop_duplicates(subset=["qid", "model_answer"])

    # Oversample the looser tiers. The content-word tier never fired on real answers
    # (exact 54%, substring 18%, no match 28%), but stays in the quota at zero.
    quota = {"substring": 20, "none": 15, "exact": 15, "content-word": 0}
    picked = []
    for tier, want in quota.items():
        sub = pool[pool["tier"] == tier]
        if len(sub) == 0:
            print(f"  tier '{tier}': none available")
            continue
        take = min(want, len(sub))
        picked.append(sub.sample(n=take, random_state=SEED))
        print(f"  tier '{tier}': {take} sampled (pool {len(sub)})")
    sheet = pd.concat(picked).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Top up to N_TOTAL from whatever is left if a tier was short.
    if len(sheet) < N_TOTAL:
        rest = pool[~pool.index.isin(sheet.index)]
        extra = rest.sample(n=min(N_TOTAL - len(sheet), len(rest)), random_state=SEED)
        sheet = pd.concat([sheet, extra]).reset_index(drop=True)
    sheet = sheet.head(N_TOTAL)

    # Blank columns for the annotator. Deliberately last, and deliberately empty.
    sheet["human_level"] = ""      # 0..n-1, or "none" if the answer matches no level
    sheet["human_agrees"] = ""     # y / n
    sheet["note"] = ""

    sheet.to_csv(SHEET, index=False)
    print(f"\nwrote {SHEET} ({len(sheet)} rows)")
    print("\nHOW TO FILL IT IN")
    print("  For each row read the question, the gold hierarchy and the model's")
    print("  answer, then write in human_level the most specific hierarchy level")
    print("  the answer genuinely supports (0 = finest), or 'none'. Fill")
    print("  human_agrees with y if that equals matcher_level, n otherwise.")
    print("  Do not look at matcher_says first; it is there for scoring, and")
    print("  reading it before judging biases the study.")
    print("\nThen: conda run -n thesis312 python score_matcher_study.py")


if __name__ == "__main__":
    main()
