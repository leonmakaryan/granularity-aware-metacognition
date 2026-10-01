# Scorer check (Gate 2)

Fill in `sheet.csv` without opening `key.csv`.

- `human_outcome`: `correct`, `wrong`, `abstain`, or `format_failure` (the reply does not give an answer in the requested `Location:` / `Date:` form and does not decline).
- Judge only the primary claim: for places the text before the first comma, for dates the date as stated at its own precision.
- `human_level`, only when correct: for places the number of the reference entry the answer is correct at; for dates `day`, `month`, `year`, `decade` or `century` (a season counts as a year, an early/mid/late decade as a decade). Credit cannot be finer than the reference precision.

Then run `python -m pipeline.run scorer-check-score`.
