"""Were the adapters' training records built with the corrected scorer? Rebuild them
from the saved samples and compare byte for byte, then count how many targets the
old rules would have changed.
"""
import json
from pipeline import score
from pipeline.data import load
from pipeline.targets import build_records, support_level
from src.text import find_match_level
score.JULIAN_CREDIT = False
new_match, new_parse = score.find_match_level_v3, score.parse_date
def rd(p): return [json.loads(l) for l in open(p)]
for M in ("7b", "1p5b", "3b"):
    for dom in ("locations", "dates"):
        items = load(dom, "train")
        raw = {str(json.loads(l)["qid"]): json.loads(l)["raws"] for l in open(f"results/sep11/{M}/samples/{dom}_train.jsonl")}
        S = {it.qid: raw[str(it.qid)] for it in items}
        rebuilt = {str(r["qid"]): r["completion"][0]["content"] for r in build_records(dom, items, S)[0]}
        for ad in (f"{dom}_s0", f"{dom}_s1", f"{dom}_s2", "joint_s0"):
            st = {str(r["qid"]): r["completion"][0]["content"] for r in rd(f"results/sep11/{M}/adapters/{ad}/records.jsonl") if r["domain"] == dom}
            same = sum(st.get(q) == c for q, c in rebuilt.items()) if len(st) == len(rebuilt) else -1
            print(f"{M:4s} {dom:9s} {ad:12s} stored {len(st)} records, identical to rebuild: {same}/{len(rebuilt)}")
        cur = {it.qid: support_level(it, S[it.qid]) for it in items}
        score.find_match_level_v3 = lambda text, h: find_match_level(text, h)
        score.parse_date = lambda t: score._PARSER.first_most_specific(t)
        old_items = load(dom, "train", historical_bce=False) if dom == "dates" else items
        old = {it.qid: support_level(it, S[it.qid]) for it in old_items}
        score.find_match_level_v3, score.parse_date = new_match, new_parse
        print(f"     -> targets the OLD matcher/parser/BCE rules would have changed: {sum(cur[q] != old[q] for q in cur)}/{len(cur)}\n")
