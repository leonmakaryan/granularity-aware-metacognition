"""Export both datasets with checksums and the dataset revision to
phase2_results/provenance/, and check that the normalized levels follow from the
hierarchy depth on both splits.
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("phase2_results/provenance")
AUDIT_SHA = "afa18652857e319172b0f8f49d5ca0cf99bf74cc"   # revision recorded at the first download


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolved_revision():
    """The commit the local datasets cache actually resolved for this dataset."""
    try:
        from huggingface_hub import HfApi
        from src.config import DATASET
        return HfApi().dataset_info(DATASET).sha
    except Exception as exc:                       # offline / auth / API change
        return f"unavailable: {type(exc).__name__}: {exc}"


def freeze_locations():
    from src.config import DATASET, LOCATION_RELATIONS
    from src.data import load_split

    rows, per_rel, patterns, violations = [], Counter(), {}, []
    for split in ("train", "test"):
        for q in load_split(split, LOCATION_RELATIONS):
            rows.append(dict(qid=q.qid, relation=q.relation, split=split,
                             question=q.question, hierarchy=q.hierarchy,
                             normalized_levels=q.levels))
            per_rel[f"{q.relation}/{split}"] += 1
            # Re-test the audit's reconstruction assumption on the real column.
            depth = len(q.hierarchy)
            if depth in patterns and patterns[depth] != q.levels:
                violations.append(dict(qid=q.qid, depth=depth, levels=q.levels,
                                       expected=patterns[depth]))
            patterns.setdefault(depth, q.levels)

    path = OUT / "locations.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    splits = Counter(r["split"] for r in rows)
    assert splits["test"] == 303, splits
    ids = [r["qid"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate location qid"
    by_split = defaultdict(set)
    for r in rows:
        by_split[r["split"]].add(r["qid"])
    assert not (by_split["train"] & by_split["test"]), "train/test leak"

    return dict(
        dataset=DATASET, relations=LOCATION_RELATIONS,
        resolved_revision=resolved_revision(), audit_revision=AUDIT_SHA,
        n=len(rows), splits=dict(splits), per_relation=dict(sorted(per_rel.items())),
        level_patterns_by_depth={str(k): v for k, v in sorted(patterns.items())},
        depth_pattern_violations=violations,
        depth_reconstruction_valid=not violations,
        export=str(path), export_sha256=sha256(path))


def freeze_dates():
    import src.dates as D

    rows = []
    for split in ("train", "test"):
        for q in D.load_dates_split(split):
            rows.append(dict(qid=q.qid, split=split, person=q.person, kind=q.kind,
                             question=q.question, gold=str(q.gold),
                             gold_level=q.gold.level.name))
    path = OUT / "dates.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    splits = Counter(r["split"] for r in rows)
    assert splits["test"] == 503 and splits["train"] == 2724, splits
    ids = [r["qid"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate date qid"
    # The split is per PERSON, so no person may appear on both sides.
    people = defaultdict(set)
    for r in rows:
        people[r["split"]].add(r["person"])
    leaked = people["train"] & people["test"]
    assert not leaked, f"person-level leak: {sorted(leaked)[:5]}"

    return dict(
        source=str(D.DATES_PATH), source_sha256=sha256(D.DATES_PATH),
        split_seed=D.DATES_SPLIT_SEED, test_frac=D.DATES_TEST_FRAC,
        n=len(rows), splits=dict(splits),
        gold_levels=dict(Counter(r["gold_level"] for r in rows)),
        person_level_split_clean=True,
        export=str(path), export_sha256=sha256(path))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from src.config import MODEL_NAME, CORRECTED_SCORING

    loc, dat = freeze_locations(), freeze_dates()
    inputs = ["src/data.py", "src/dates.py", "src/text.py", "src/eval.py",
              "src/config.py", "src/capability.py", "src/prompt.py"]
    manifest = dict(
        created="2026-09-09", corrected_scoring=CORRECTED_SCORING,
        default_model=MODEL_NAME, locations=loc, dates=dat,
        code_sha256={p: sha256(p) for p in inputs})
    path = OUT / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"locations  {loc['n']} rows  {loc['splits']}")
    print(f"           revision resolved: {loc['resolved_revision']}")
    match = loc["resolved_revision"] == AUDIT_SHA
    print(f"           matches the revision the audit recorded: {match}")
    print(f"           depth->levels reconstruction valid: "
          f"{loc['depth_reconstruction_valid']} "
          f"({len(loc['depth_pattern_violations'])} violations)")
    print(f"           patterns: {loc['level_patterns_by_depth']}")
    print(f"dates      {dat['n']} rows  {dat['splits']}  "
          f"person-level split clean: {dat['person_level_split_clean']}")
    print(f"           gold levels: {dat['gold_levels']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
