"""Regrade the saved conflict-probe answers with the frozen scorer.

    python analysis/rescore_probes.py

The raw answers stay unchanged. Every other column is recomputed by
src/conflict.classify, which reads and grades dates with pipeline/score.py (bare-year
retry, Julian credit for pre-1582 exact dates), so the probe follows the same scoring
rules as every other answer. The per-file summaries (.json) are rewritten from the
regraded rows. The entity probe already uses pipeline.score.grade and is not touched.
Prints, per file, how many rows changed.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import config as C
from src.conflict import build_cases, classify, summarize
from src.dates import TemporalLevel, level_info, load_dates_split

COLS = ["answer", "parsed", "answer_level", "idk", "behaviour", "signed_strict", "correct_strict"]


def regrade(case, raw):
    if not str(raw).strip():
        # Same rule as pipeline/probes.py: an empty completion is a format failure.
        row = classify(case, "x")
        row.update(answer="", parsed="", answer_level="", idk=False, behaviour="fmt_fail",
                   correct_strict=False, signed_strict=-level_info(TemporalLevel.YEAR))
        return row
    return classify(case, raw)


def main():
    cases = build_cases(load_dates_split("test", historical_bce=True), seed=C.GLOBAL_SEED)
    for m in ("7b", "1p5b"):
        for path in sorted((C.RESULTS / m / "probes").glob("conflict__*.csv")):
            df = pd.read_csv(path, keep_default_na=False)
            assert len(df) == len(cases) and list(df.qid) == [c["qid"] for c in cases]
            rows, changed = [], 0
            for case, r in zip(cases, df.to_dict("records")):
                new = regrade(case, r["raw"])
                new["raw"] = r["raw"]
                changed += any(str(new[k]) != str(r[k]) for k in COLS
                               if k not in ("signed_strict",)) or \
                    abs(float(new["signed_strict"]) - float(r["signed_strict"])) > 1e-9
                rows.append({k: new[k] for k in df.columns})
            pd.DataFrame(rows, columns=df.columns).to_csv(path, index=False)
            js = path.with_suffix(".json")
            if js.exists():
                old = json.loads(js.read_text())
                old["summary"] = summarize(rows)
                js.write_text(json.dumps(old, indent=1) + "\n")
            print(f"{m} {path.name}: {changed} of {len(rows)} rows changed")


if __name__ == "__main__":
    main()
