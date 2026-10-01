# How sheet.csv was labelled

An LLM (Claude) labelled every answer from the question, the reference and the model
output alone: the scorer was not run on these rows, and `key.csv` was not opened until
every row was labelled. The author then checked every label (11 September 2026) and
changed none. This is weaker than labelling by an independent human annotator.

Rules (from README.md): judge only the primary claim (places: the text before the first
comma; dates: the date as stated, at its own precision); credit is never finer than the
reference; `human_level` only for correct answers.

Conventions for cases the rules do not settle:

1. BCE answers are judged against the historical year. Wikidata counts BCE years
   astronomically, so a reference shown as "43 BCE" for Julius Caesar's death means 44 BC.
2. Where a GRANOLA-EQ hierarchy itself looks wrong (item 97, the Battle of Sinop listed
   in Brazil), the answer is judged against the reference as given, with a note.
3. Detail the reference cannot verify (a day against a year-only reference) is credited
   at the reference precision; a note says when the detail is known to be wrong.
4. A true place finer than the finest reference entry is credited at the finest entry
   that contains it (item 45, King Cove in Alaska).
5. Dates without a year ("24 March") and year ranges ("353-406") have no level and are
   marked `wrong`; notes say whether the partial claim is true.
6. A doubled marker ("Location: Location: Spain") counts as a well-formed answer.

## Result

After the BCE year fix in the scorer: outcome agreement 160/162 (Cohen's kappa 0.980)
and level agreement 77/77, so Gate 2 passed. `result.json` holds the details.
