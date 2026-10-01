import csv
from collections import Counter
from src.dates import extract_date_text, parse_date_text as probe_parse
from pipeline.score import parse_date as pipeline_parse

P = "results/sep11/7b/probes/"
for adapter in ("base", "locations_s0", "dates_s0", "joint_s0"):
    rows = list(csv.DictReader(open(P + f"conflict__{adapter}.csv")))
    month = [r for r in rows if r["condition"] == "conflict" and r["agreement_level"] == "MONTH"]
    levels = Counter(r["answer_level"] or r["behaviour"] for r in month)
    month_answers = sum(1 for r in rows if r["answer_level"] == "MONTH")
    differ = []
    for r in rows:
        text = extract_date_text(r["raw"])
        if not text:
            continue
        a, b = probe_parse(text), pipeline_parse(text)
        if (a is None) != (b is None) or (a is not None and b is not None and a.level != b.level):
            differ.append(text)
    print(f"{adapter}: month-agreement answer levels {dict(levels.most_common())}; "
          f"month-level answers anywhere in the probe {month_answers}/{len(rows)}; "
          f"answers the two parsers read differently {len(differ)}/{len(rows)} {sorted(set(differ))[:6]}")
