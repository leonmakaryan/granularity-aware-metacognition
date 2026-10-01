"""Recompute the targets of the cached base samples under the two scoring fixes,
offline. Writes <name>_fixed.json next to each cache and leaves the originals,
since the adapters were trained on those.
"""

import copy
import json
import sys
from collections import Counter
from pathlib import Path

OUT = Path("phase2_results")

# (file, scale, expected change count or None when the audit did not check it)
LOCATION_CACHES = [
    ("targets_loc_7b_t2.json", "7B", 109),    # trained the current 7B loc + joint adapters
    ("capability_loc_7b.json", "7B", None),
    ("targets_loc_v2.json", "1.5B", None),
    ("capability_loc_v2.json", "1.5B", None),
]
DATE_CACHES = [
    ("dates_7b_capability.json", "7B", 93),
    ("dates_capability.json", "1.5B", None),
]


def load(name):
    return json.loads((OUT / name).read_text())


def summarize(changes, dist_before, dist_after, n):
    kinds = Counter()
    for c in changes:
        old, new = str(c["old"]), str(c["new"])
        kinds["IDK->level" if old == "None" else
              "level->IDK" if new == "None" else "level->level"] += 1
    print(f"    changed {len(changes)}/{n} ({100*len(changes)/n:.1f}%)  "
          + "  ".join(f"{k} {v}" for k, v in kinds.most_common()))
    idk_b, idk_a = dist_before.get("None", 0), dist_after.get("None", 0)
    print(f"    abstain targets {idk_b} -> {idk_a} "
          f"({100*idk_b/n:.1f}% -> {100*idk_a/n:.1f}%)")
    top = Counter((str(c["old"]), str(c["new"])) for c in changes).most_common(5)
    for (o, nw), v in top:
        print(f"      {o:8} -> {nw:8}  {v}")
    return dict(n=n, changed=len(changes), kinds=dict(kinds),
                idk_before=idk_b, idk_after=idk_a,
                dist_before=dist_before, dist_after=dist_after,
                changes=changes)


def repair_locations(name, expected):
    """Recompute a location cache under its own target rule (v1 for capability_*.json,
        v2 for targets_*.json), so only the matcher changes."""
    from src.capability import pick_target_v2, pick_target, reliability_curve
    from src.text import match_level, find_match_level
    from src.prompt import is_idk

    cap = load(name)
    qs = cap["questions"]
    is_v2 = cap.get("target_rule", {}).get("rule") == "v2"
    print(f"    rule: {'v2 (retargeted)' if is_v2 else 'v1 (raw capability)'}")
    dist_before = Counter(str(r["target_level"]) for r in qs.values())
    fixed = copy.deepcopy(cap)
    changes = []
    for qid, r in fixed["questions"].items():
        hier, depth = r["hierarchy"], len(r["hierarchy"])

        def derive(mls, gm):
            if is_v2:
                return pick_target_v2(r["sample_answers"], mls, r["greedy_is_idk"],
                                      gm, depth, .5)
            return pick_target(reliability_curve(mls, depth), cap.get("tau", .5))

        # Sanity: the stored targets must be reproducible under the OLD matcher,
        # otherwise this cache was not built by the production path.
        old_mls = [None if is_idk(a) else find_match_level(a, hier)
                   for a in r["sample_answers"]]
        old_gm = (None if r["greedy_is_idk"]
                  else find_match_level(r["greedy_answer"], hier))
        assert derive(old_mls, old_gm) == r["target_level"], f"{name} {qid}"

        mls = [None if is_idk(a) else match_level(a, hier) for a in r["sample_answers"]]
        gm = None if r["greedy_is_idk"] else match_level(r["greedy_answer"], hier)
        target = derive(mls, gm)
        r["curve"] = reliability_curve(mls, depth)
        if target != r["target_level"]:
            changes.append(dict(qid=qid, old=r["target_level"], new=target))
        r["sample_match_levels"] = mls
        r["greedy_match_level"] = gm
        r["target_level"] = target
        r["target_is_idk"] = target is None
        r["target_norm_level"] = (r["levels"][target] if target is not None
                                  and target < len(r["levels"]) else None)
    if expected is not None and len(changes) != expected:
        sys.exit(f"GATE FAIL {name}: {len(changes)} changes, audit says {expected}")
    dist_after = Counter(str(r["target_level"]) for r in fixed["questions"].values())
    fixed["scoring"] = "corrected_sep09"
    return fixed, summarize(changes, dict(dist_before), dict(dist_after), len(qs))


def repair_dates(name, expected):
    import src.dates as D

    cap = load(name)
    split = "train"
    golds = {q.qid: q.gold for q in D.load_dates_split(split)}
    D.derive_dates_targets(cap)                      # stored rule, original parse
    dist_before = Counter(str(r["target_level"]) for r in cap["questions"].values())
    fixed = copy.deepcopy(cap)
    for qid, r in fixed["questions"].items():
        for s in r["samples"] + [r["greedy"]]:
            if s["idk"]:
                continue
            v = D.parse_date_text(s["text"])
            level = D.credited_level(v, golds[qid], False)
            s["lenient_level"] = level.name if level is not None else None
            s["parsed_level"] = v.level.name if v is not None else None
    D.derive_dates_targets(fixed)
    changes = [dict(qid=k, old=r["target_level"],
                    new=fixed["questions"][k]["target_level"])
               for k, r in cap["questions"].items()
               if r["target_level"] != fixed["questions"][k]["target_level"]]
    if expected is not None and len(changes) != expected:
        sys.exit(f"GATE FAIL {name}: {len(changes)} changes, audit says {expected}")
    dist_after = Counter(str(r["target_level"]) for r in fixed["questions"].values())
    fixed["scoring"] = "corrected_sep09"
    return fixed, summarize(changes, dict(dist_before), dict(dist_after),
                            len(cap["questions"]))


def main():
    from src.config import CORRECTED_SCORING
    assert CORRECTED_SCORING, "run with CORRECTED_SCORING=1 (the default)"
    report = {}
    for group, fn in ((LOCATION_CACHES, repair_locations), (DATE_CACHES, repair_dates)):
        for name, scale, expected in group:
            if not (OUT / name).exists():
                print(f"  skip {name} (missing)")
                continue
            print(f"\n{name}  [{scale}]" + (f"  gate: audit says {expected}" if expected else ""))
            fixed, stats = fn(name, expected)
            out = OUT / name.replace(".json", "_fixed.json")
            out.write_text(json.dumps(fixed, indent=2) + "\n")
            print(f"    wrote {out}")
            report[name] = dict(scale=scale, **stats)
    path = OUT / "cache_repair_sep09.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {path}")
    print("originals untouched; every stored adapter still traces to its original targets")


if __name__ == "__main__":
    main()
