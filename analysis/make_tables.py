"""Every table of the Results chapter, from the saved results only.

    CONFIRM=1 PYTHONPATH=. python analysis/make_tables.py

Writes analysis/output/tables/*.tex, the figures, and results/tables.json (every number shown,
for checking the text). Numbers are recomputed from the saved answers with the pipeline's
own functions and checked against the stored reports. The script stops if two compared
conditions have different questions, or if one number would be shown with two values.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import config as C
from pipeline.report import load_preds, paired_bootstrap, summarize
from pipeline.support import CATEGORIES

OUT = C.ROOT / "analysis" / "output" / "tables"
SIZES = ("7b", "3b", "1p5b")
NAME = {"7b": "7B", "3b": "3B", "1p5b": "1.5B"}
OTHER = {"locations": "dates", "dates": "locations"}
COND = ("in-domain", "transfer", "joint")
LABEL = {"base": "Base", "in-domain": "In-domain", "transfer": "Transfer", "joint": "Joint"}
SETNAME = {("locations", "test"): "Locations test", ("dates", "test"): "Dates test",
           ("locations", "val"): "Locations validation"}

# ---------------------------------------------------------------- registry
VALUES = {}


def reg(key, value):
    """Record a shown number; the same key must never get two values."""
    if key in VALUES and not np.isclose(VALUES[key], value, atol=1e-12, rtol=0):
        raise AssertionError(f"{key} shown twice with different values: {VALUES[key]} vs {value}")
    VALUES[key] = float(value)
    return value


# ---------------------------------------------------------------- formatting
def signed(x, d=3):
    t = f"{x:+.{d}f}"
    return "+" + t[1:] if t.startswith("-") and float(t) == 0 else t   # no "-0.000"


def sgn(x, d=3):
    return f"${signed(x, d)}$"


def num(x, d=3):
    return f"{x:.{d}f}"


def pct(x):
    return f"{100 * x:.1f}"


def interval(ci, d=3):
    return f"${signed(ci['delta'], d)}\\;[{signed(ci['lo'], d)}, {signed(ci['hi'], d)}]$"


def two(a, b):
    """A header cell on two lines."""
    return f"\\begin{{tabular}}[b]{{@{{}}c@{{}}}}{a}\\\\{b}\\end{{tabular}}"


PCT3 = " & ".join(two(x, "(\\%)") for x in ("Correct", "Abstain", "Wrong"))


def points(ci):
    """A difference of two shares, in percentage points, with its interval."""
    f = lambda x: signed(100 * x, 1)
    return f"${f(ci['delta'])}\\;[{f(ci['lo'])}, {f(ci['hi'])}]$"


def sd(x):
    if x is None:
        return "--"
    return "$<0.001$" if x < 0.0005 else f"{x:.3f}"


def short_caption(caption):
    """First sentence of a long caption, used as its entry in the List of Tables."""
    m = re.search(r"(?<!e\.g)(?<!i\.e)(?<!al)\.\s", caption)
    if not m or m.end() >= len(caption) - 1:
        return None
    short = caption[:m.start() + 1]
    return short if short.count("{") == short.count("}") else None


def table(name, caption, label, colspec, header, rows, size="\\small", note=None, sep=None):
    lines = ["\\begin{table}[!tb]", "\\centering" + size + (f"\\setlength{{\\tabcolsep}}{{{sep}}}" if sep else ""),
             (f"\\caption[{short_caption(caption)}]{{{caption}}}" if short_caption(caption) else f"\\caption{{{caption}}}"), f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(r if r.startswith("\\") else r + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append(f"\\par\\smallskip{{\\footnotesize {note}\\par}}")
    lines.append("\\end{table}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- data
def adapters(cond, domain):
    """Adapter names of a training condition, as evaluated on `domain`."""
    trained = {"in-domain": domain, "transfer": OTHER[domain], "joint": "joint"}[cond]
    return [f"{trained}_s{s}" for s in C.SEEDS]


def preds(m, adapter, domain, split, mode="deploy"):
    return load_preds(C.RESULTS / m, adapter, domain, split, mode)[1]


def same_questions(ref, dfs, what):
    for d in dfs:
        if list(d.qid) != list(ref.qid):
            raise AssertionError(f"question sets differ: {what}")


def condition_block(m, domain, split):
    """Base and the three training conditions on one set, checked against the report."""
    base = preds(m, "base", domain, split)
    rep = json.load(open(C.RESULTS / m / f"report_{'val' if split == 'val' else 'dev'}.json"))
    rep = rep[f"{domain}_{split}"]
    out = {"base": dict(summarize([base]), n=len(base), dfs=[base])}
    for cond in COND:
        dfs = [preds(m, a, domain, split) for a in adapters(cond, domain)]
        same_questions(base, dfs, f"{m} {domain} {split} {cond}")
        s = summarize(dfs)
        s["d_score"] = paired_bootstrap(base, dfs, "score")
        s["d_match"] = paired_bootstrap(base, dfs, "exact")
        s["dfs"] = dfs
        out[cond] = s
        for k in ("score", "exact", "correct", "abstain", "wrong"):
            assert np.isclose(s[k], rep[cond][k], atol=1e-12), (m, domain, split, cond, k)
        for k, j in (("d_score", "delta_score"), ("d_match", "delta_exact")):
            for f in ("delta", "lo", "hi"):
                assert np.isclose(s[k][f], rep[cond][j][f], atol=1e-12), (m, domain, split, cond, k, f)
    for k in ("score", "exact", "correct", "abstain", "wrong"):
        assert np.isclose(out["base"][k], rep["base"][k], atol=1e-12)
    # share of questions whose estimated support is abstention (the same for every condition)
    out["base"]["no_support"] = float(base.support.isna().mean())
    for cond in COND:
        for d in out[cond]["dfs"]:
            assert list(d.support.isna()) == list(base.support.isna())
    reg(f"{m}.{domain}_{split}.no_support", out["base"]["no_support"])
    for cond, s in out.items():
        key = f"{m}.{domain}_{split}.{cond}"
        for k in ("score", "exact", "correct", "abstain", "wrong"):
            reg(f"{key}.{k}", s[k])
        reg(f"{key}.answered", 1 - s["abstain"])
        for c in CATEGORIES:
            reg(f"{key}.group.{c}", s["categories"][c])
        if cond != "base":
            for k in ("d_score", "d_match"):
                for f in ("delta", "lo", "hi", "seed_sd"):
                    reg(f"{key}.{k}.{f}", s[k][f])
    return out


# ---------------------------------------------------------------- 4.1 scores
def score_rows(block, set_name, n):
    rows = [f"\\multicolumn{{7}}{{l}}{{\\textit{{{set_name}, {n} questions}}}}"
            + " \\\\"]
    for cond in ("base",) + COND:
        s = block[cond]
        diff = "" if cond == "base" else interval(s["d_score"])
        spread = "" if cond == "base" else sd(s["d_score"]["seed_sd"])
        rows.append(f"{LABEL[cond]} & {sgn(s['score'])} & {diff} & {spread} & "
                    f"{pct(s['correct'])} & {pct(s['abstain'])} & {pct(s['wrong'])}")
    return rows


def main_tables(blocks):
    head = "Model & Score & " + two("Difference", "to base") + " & SD & " + PCT3
    spec = "lrlrrrr"
    rows = []
    for i, (domain, split) in enumerate((("locations", "test"), ("locations", "val"), ("dates", "test"))):
        if i:
            rows.append("\\midrule")
        b = blocks[("7b", domain, split)]
        rows += score_rows(b, SETNAME[(domain, split)], b["base"]["n"])
    table("scores_7b",
          "Scores of the 7B model on the location test and validation sets and on the date "
          "test set. The in-domain adapters raise the score significantly on dates and on the "
          "validation set, and the joint adapters on both test sets. The transfer adapters do "
          "not change the score significantly. Base: Qwen2.5-7B-Instruct with the deployment "
          "prompt. In-domain, transfer, and joint as in Section~\\ref{sec:eval-design}; each "
          "of these rows is the mean of three adapters. Differences are paired, computed "
          "before rounding, with 95\\% intervals in brackets. SD is the standard deviation of "
          "the difference over the three training runs.",
          "tab:scores-7b", spec, head, rows)

    rows = []
    for i, (domain, split) in enumerate((("locations", "test"), ("dates", "test"))):
        for j, m in enumerate(("3b", "1p5b")):
            if i or j:
                rows.append("\\midrule")
            b = blocks[(m, domain, split)]
            rows += score_rows(b, f"{NAME[m]}, {SETNAME[(domain, split)].lower()}", b["base"]["n"])
    table("scores_small",
          "Scores of the 3B and 1.5B models on the two test sets. The in-domain and joint "
          "adapters raise the score significantly on dates at both sizes and on locations at "
          "3B. At 1.5B, the transfer adapters also raise the date score significantly. Base: "
          "Qwen2.5-3B-Instruct and Qwen2.5-1.5B-Instruct with the deployment prompt. Columns "
          "as in Table~\\ref{tab:scores-7b}.",
          "tab:scores-small", spec, head, rows, size="\\footnotesize")


# ---------------------------------------------------------------- 4.2 support match
def support_tables(blocks):
    sets = (("locations", "test"), ("locations", "val"), ("dates", "test"))
    head = "Model & " + " & ".join(SETNAME[s] for s in sets)
    rows = []
    for i, m in enumerate(SIZES):
        if i:
            rows.append("\\midrule")
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{NAME[m]}}}}} \\\\")
        for cond in ("base",) + COND:
            cells = []
            for s in sets:
                b = blocks.get((m,) + s)
                if b is None:
                    cells.append("--")
                elif cond == "base":
                    cells.append(num(b["base"]["exact"]))
                else:
                    cells.append(interval(b[cond]["d_match"]))
            rows.append(f"{LABEL[cond]} & " + " & ".join(cells))
    table("support_match",
          "Support match on the test sets (303 location and 503 date questions) and the 7B "
          "validation set (285 questions). The base rows give the "
          "support match of the base model; the other rows give the paired difference to it "
          "with its 95\\% interval. Support match rises significantly only for the 3B "
          "in-domain and joint adapters on locations. Each adapter row is the mean of three "
          "adapters. The validation set is used only with the 7B model.",
          "tab:support-match", "llll", head, rows, size="\\footnotesize")

    groups = GROUPS
    sets2 = (("locations", "test"), ("dates", "test"))
    head = ("Group & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{SETNAME[s]}}}" for s in sets2)
            + " \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n & "
            + " & ".join(["Base", "In-dom.", "Transf.", "Joint"] * 2))
    rows = []
    for c, lab in groups:
        cells = [pct(blocks[("7b",) + s][cond]["categories"][c]) for s in sets2 for cond in ("base",) + COND]
        rows.append(f"{lab} & " + " & ".join(cells))
    table("support_groups_7b",
          "Answers of the 7B model by their agreement with the estimated support, in percent "
          "of the test questions (303 on locations, 503 on dates). Supported questions have "
          "a level as their estimated support; for the others it is abstention. On "
          "locations, the in-domain and joint adapters give about half as many wrong answers "
          "to questions without support. On dates, they give many correct answers that are "
          "coarser than the support.",
          "tab:support-groups-7b", "l" + "r" * 8, head, rows, size="\\footnotesize", sep="4pt")

    # all sizes and sets, for the appendix
    rows = []
    order = [("7b", "locations", "val"), ("3b", "locations", "test"), ("3b", "dates", "test"),
             ("1p5b", "locations", "test"), ("1p5b", "dates", "test")]
    head = "Group & Base & In-domain & Transfer & Joint"
    for i, key in enumerate(order):
        if i:
            rows.append("\\midrule")
        b = blocks[key]
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{NAME[key[0]]}, "
                    f"{SETNAME[key[1:]].lower()}, {b['base']['n']} questions}}}} \\\\")
        for c, lab in groups:
            rows.append(f"{lab} & " + " & ".join(pct(b[cond]["categories"][c]) for cond in ("base",) + COND))
    table("support_groups_other",
          "Answers by their agreement with the estimated support, in percent of the "
          "questions, for the 7B validation set and the 3B and 1.5B test sets. On the "
          "validation set and on 3B locations, the in-domain and joint adapters give fewer wrong "
          "answers to questions without support; at 1.5B they do not. On dates, the 1.5B "
          "in-domain and joint adapters give more correct answers that are coarser than the "
          "support, and the 3B ones more correct answers to questions without support. Groups "
          "as in Table~\\ref{tab:support-groups-7b}.",
          "tab:support-groups-other", "lrrrr", head, rows, size="\\scriptsize")


GROUPS = [("exact_support_match", "Match"),
          ("correct_coarser_than_support", "Supported: correct but coarser"),
          ("correct_finer_than_support", "Supported: correct but finer"),
          ("wrong_despite_support", "Supported: wrong"),
          ("abstained_despite_support", "Supported: abstains"),
          ("answered_without_support_correct", "Not supported: correct"),
          ("answered_without_support_wrong", "Not supported: wrong")]


def diagnostic_3b(blocks):
    """3B base on dates with the other penalty and without the abstention sentence."""
    base = blocks[("3b", "dates", "test")]["base"]
    runs = [("Deployment prompt, penalty 1.05 (default)", base["dfs"][0]),
            ("Deployment prompt, penalty 1.1", preds("3b", "base", "dates", "test-rp1.1")),
            ("Without abstention sentence, penalty 1.05", preds("3b", "base", "dates", "test", "noidk"))]
    rows = []
    for i, (lab, df) in enumerate(runs):
        same_questions(base["dfs"][0], [df], f"3b diagnostic {lab}")
        s = summarize([df])
        key = f"3b.dates_test.base_diag{i}"
        for k in ("score", "correct", "abstain", "wrong"):
            reg(f"{key}.{k}", s[k])
        rows.append(f"{lab} & {sgn(s['score'])} & {pct(s['correct'])} & {pct(s['abstain'])} & {pct(s['wrong'])}")
    table("diagnostic_3b",
          "The 3B base model on the date test set (503 questions) under three settings. "
          "With a repetition penalty of 1.1, it abstains far less often and gives more "
          "correct answers. Without the abstention sentence, it never abstains, and more "
          "than half of its answers are wrong. The first row is the base row of "
          "Table~\\ref{tab:scores-small}.",
          "tab:diagnostic-3b", "lrrrr",
          "Setting & Score & " + PCT3, rows)


# ---------------------------------------------------------------- 4.3 baselines and controls
def seed_mean(dfs):
    """One frame whose score and match columns are the per-question means over seeds."""
    out = dfs[0].copy()
    for col in ("score", "exact"):
        out[col] = sum(d.set_index("qid").loc[out.qid, col].to_numpy() for d in dfs) / len(dfs)
    return out


def prompt_baselines(blocks):
    head = "Prompt & Score & " + two("Support", "match") + " & " + PCT3
    rows = []
    for i, m in enumerate(("7b", "1p5b")):
        for j, domain in enumerate(("locations", "dates")):
            if i or j:
                rows.append("\\midrule")
            dep = blocks[(m, domain, "test")]["base"]
            direct = preds(m, "base", domain, "test", "noidk")
            same_questions(dep["dfs"][0], [direct], f"{m} {domain} direct")
            s = summarize([direct])
            ds = paired_bootstrap(dep["dfs"][0], [direct], "score")
            dm = paired_bootstrap(dep["dfs"][0], [direct], "exact")
            key = f"{m}.{domain}_test.base_direct"
            for k in ("score", "exact", "correct", "abstain", "wrong"):
                reg(f"{key}.{k}", s[k])
            reg(f"{key}.fmt_fail", s["fmt_fail"])
            for k, ci in (("d_score", ds), ("d_match", dm)):
                for f in ("delta", "lo", "hi"):
                    reg(f"{key}.{k}.{f}", ci[f])
            rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{NAME[m]}, {domain} test, "
                        f"{len(direct)} questions}}}} \\\\")
            rows.append(f"Deployment & {sgn(dep['score'])} & {num(dep['exact'])} & "
                        f"{pct(dep['correct'])} & {pct(dep['abstain'])} & {pct(dep['wrong'])}")
            rows.append(f"Direct & {sgn(s['score'])} & {num(s['exact'])} & "
                        f"{pct(s['correct'])} & {pct(s['abstain'])} & {pct(s['wrong'])}")
            rows.append(f"Difference & {interval(ds)} & {interval(dm)} & & &")
    table("prompt_baselines",
          "The 1.5B and 7B base models with the deployment prompt and with the prompt "
          "without the abstention sentence (direct answers). Without the sentence, the models "
          "almost never abstain, and support match falls in every case. Differences are "
          "paired, direct minus deployment, with 95\\% intervals.",
          "tab:baselines", "lllrrr", head, rows, size="\\footnotesize")


def year_cap(m):
    """The base model's date answers with every answer finer than a year cut to its year."""
    from pipeline import score
    from pipeline.data import load
    from pipeline.support import category
    from src.dates import TemporalLevel
    from ugc.parser.temporal_constants import PRECISION
    items = {it.qid: it for it in load("dates", "test")}
    base = preds(m, "base", "dates", "test")
    cap, changed = base.copy(), 0
    for i, r in base.iterrows():
        if r.kind != "answered":
            continue
        v = score.parse_date(r.text)
        if v is None or PRECISION[v.level] <= PRECISION[TemporalLevel.YEAR]:
            continue
        text = f"Date: {v.year}" if v.year >= 1 else f"Date: {-v.year} BC"
        g = score.grade(items[r.qid], text)
        assert g["kind"] == "answered" and g["text"] and score.parse_date(g["text"]).level == TemporalLevel.YEAR
        sup = None if pd.isna(r.support) else int(float(r.support))
        cat = category(sup, g)
        cap.at[i, "correct"] = float(g["correct"])
        cap.at[i, "score"] = g["score"]
        cap.at[i, "category"] = cat
        cap.at[i, "exact"] = float(cat == "exact_support_match")
        changed += 1
    cap["wrong"] = ((cap.kind != "abstained") & (cap.correct == 0)).astype(float)
    return cap, changed


def date_controls(blocks):
    head = ("Answers & Score & " + two("Support", "match") + " & " + PCT3 + " & "
            + two("In-domain minus", "row, score"))
    rows = []
    for i, m in enumerate(("7b", "1p5b")):
        if i:
            rows.append("\\midrule")
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{NAME[m]}}}}} \\\\")
        b = blocks[(m, "dates", "test")]
        base, ours = b["base"]["dfs"][0], b["in-domain"]["dfs"]
        cap, changed = year_cap(m)
        binary = [preds(m, f"dates_s{s}_binary", "dates", "test") for s in C.SEEDS]
        same_questions(base, [cap] + binary, f"{m} date controls")
        key = f"{m}.dates_test"
        reg(f"{key}.year_cap.changed", changed)
        conds = [("Base model", "base", [base], base),
                 ("Answer-or-abstain adapters", "binary", binary, seed_mean(binary)),
                 ("Fixed year cap", "year_cap", [cap], cap),
                 ("In-domain adapters", "in-domain", ours, None)]
        for j, (lab, k, dfs, ref) in enumerate(conds):
            s = summarize(dfs)
            for f in ("score", "exact", "correct", "abstain", "wrong"):
                reg(f"{key}.{k}.{f}", s[f])
            gap = ""
            if ref is not None:
                ci_s = paired_bootstrap(ref, ours, "score")
                ci_m = paired_bootstrap(ref, ours, "exact")
                for kk, ci in (("score", ci_s), ("match", ci_m)):
                    for f in ("delta", "lo", "hi"):
                        reg(f"{key}.in-domain_minus_{k}.{kk}.{f}", ci[f])
                gap = interval(ci_s)
            if k in ("year_cap", "binary"):
                for kk in ("score", "exact"):
                    ci = paired_bootstrap(base, dfs, kk)
                    for f in ("delta", "lo", "hi", "seed_sd"):
                        if ci[f] is not None:
                            reg(f"{key}.{k}_minus_base.{kk}.{f}", ci[f])
            rows.append(f"{lab} & {sgn(s['score'])} & {num(s['exact'])} & {pct(s['correct'])} & "
                        f"{pct(s['abstain'])} & {pct(s['wrong'])} & {gap}")
    table("date_controls",
          "Two controls on the date test set (503 questions): answer-or-abstain training, "
          "and the base answers cut to their year. Both raise the score, and at 7B the "
          "in-domain adapters score significantly higher than either. Base: Qwen2.5-7B-Instruct and "
          "Qwen2.5-1.5B-Instruct with the deployment prompt. Answer-or-abstain and in-domain "
          "rows are means over three adapters. The last column is the paired score difference "
          "of the in-domain adapters to the row, with its 95\\% interval.",
          "tab:date-controls", "lrrrrrl", head, rows, size="\\footnotesize", sep="4pt")
    # the 3B base never answers finer than a year, so its cap is the base model itself
    reg("3b.dates_test.year_cap.changed", year_cap("3b")[1])


LEVELS = ("DAY", "MONTH", "YEAR", "DECADE", "CENTURY")


def stated_levels(df):
    """Counts of correct and wrong date answers by the level the answer states."""
    from pipeline.score import parse_date
    out = {}
    for kind, text, corr in zip(df.kind, df.text, df.correct):
        if kind == "abstained":
            lab = "abstain"
        elif kind == "fmt_fail":
            lab = "fmt_fail"
        else:
            v = parse_date(text)
            lab = "unreadable" if v is None else (v.level.name if v.level.name in LEVELS else "other")
        c, w = out.get(lab, (0, 0))
        if lab == "abstain":
            out[lab] = (c + 1, 0)          # abstentions: the count, no correct/wrong split
        else:
            out[lab] = (c + bool(corr), w + (not corr))
    return out


def mean_counts(dfs, fn):
    per = [fn(d) for d in dfs]
    keys = set().union(*per)
    return {k: tuple(np.mean([p.get(k, (0, 0))[i] for p in per]) for i in (0, 1)) for k in keys}


def cnt(x):
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


def answer_levels(blocks):
    names = [("DAY", "Exact date"), ("MONTH", "Month"), ("YEAR", "Year"), ("DECADE", "Decade"),
             ("CENTURY", "Century"), ("unreadable", "Unreadable"),
             ("abstain", "Abstention")]
    for m in SIZES:
        b = blocks[(m, "dates", "test")]
        counts = {cond: mean_counts(b[cond]["dfs"], stated_levels) for cond in ("base",) + COND}
        for cond, c in counts.items():
            assert "fmt_fail" not in c, (m, cond)
            assert "other" not in c, (m, cond)       # no answer states a season or another period
            for lab, (ok, bad) in c.items():
                reg(f"{m}.dates_test.{cond}.stated.{lab}.correct", ok)
                reg(f"{m}.dates_test.{cond}.stated.{lab}.wrong", bad)
                reg(f"{m}.dates_test.{cond}.stated.{lab}.total", ok + bad)
            assert np.isclose(sum(ok + bad for lab, (ok, bad) in c.items() if lab != "abstain")
                              + c.get("abstain", (0, 0))[0], 503)
        head = ("Stated level & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{LABEL[c]}}}" for c in ("base",) + COND)
                + " \\\\\n" + "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(4))
                + "\n & " + " & ".join(["Correct", "Wrong"] * 4))
        rows = []
        for lab, name in names:
            cells = []
            for cond in ("base",) + COND:
                ok, bad = counts[cond].get(lab, (0, 0))
                if lab == "abstain":
                    cells.append(f"\\multicolumn{{2}}{{c}}{{{cnt(ok)}}}")
                else:
                    cells += [cnt(ok), cnt(bad)]
            rows.append(f"{name} & " + " & ".join(cells))
        fix = {"7b": ("tab:date-levels-7b", "Date answers of the 7B model on the test set (503 "
                      "questions) by the level they state. The in-domain and joint adapters give far "
                      "fewer exact dates than the base model and more years. Adapter counts are "
                      "means over three adapters. Abstentions are not split into correct and wrong."),
               "3b": ("tab:date-levels-3b", "Date answers of the 3B model on the test set (503 "
                      "questions) by the level they state. The base model abstains on 322 questions; the "
                      "in-domain and joint adapters abstain on about 220 and give about twice as many "
                      "centuries. Adapter counts are means over three adapters."),
               "1p5b": ("tab:date-levels-1p5b", "Date answers of the 1.5B model on the test set "
                        "(503 questions) by the level they state. The in-domain and joint adapters give "
                        "almost no exact dates, more years, decades, and centuries, and almost never "
                        "abstain. The transfer adapters give 80 unreadable dates. Adapter counts are "
                        "means over three adapters.")}[m]
        table(f"date_levels_{m}", fix[1], fix[0], "l" + "r" * 8, head, rows, size="\\footnotesize")

    # correct location answers by credited position in the hierarchy
    from pipeline.data import load
    for split in ("test",):
        items = {it.qid: it for it in load("locations", split)}

        def positions(df):
            out = {}
            for q, corr, lvl in zip(df.qid, df.correct, df.level):
                if not corr:
                    continue
                n = len(items[q].hierarchy)
                lvl = int(float(lvl))
                lab = "finest" if lvl == 0 else ("coarsest" if lvl == n - 1 else "between")
                out[lab] = (out.get(lab, (0, 0))[0] + 1, 0)
            return out
        head = "Credited entry & " + " & ".join(LABEL[c] for c in ("base",) + COND)
        rows_all = {}
        for m in SIZES:
            b = blocks[(m, "locations", split)]
            counts = {cond: mean_counts(b[cond]["dfs"], positions) for cond in ("base",) + COND}
            for cond, c in counts.items():
                for lab in ("finest", "between", "coarsest"):
                    reg(f"{m}.locations_{split}.{cond}.credited.{lab}", c.get(lab, (0, 0))[0])
                total = sum(v[0] for v in c.values())
                assert np.isclose(total, b[cond]["correct"] * len(b["base"]["dfs"][0]))
            rows_all[m] = counts
        rows = []
        for i, m in enumerate(SIZES):
            if i:
                rows.append("\\midrule")
            rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{NAME[m]}}}}} \\\\")
            for lab, name in (("finest", "Finest"), ("between", "Between"), ("coarsest", "Coarsest")):
                rows.append(f"{name} & " + " & ".join(cnt(rows_all[m][c].get(lab, (0, 0))[0]) for c in ("base",) + COND))
        table("location_levels",
              "Correct answers on the location test set (303 questions) by the hierarchy entry "
              "they match: the finest entry, an entry between the finest and the coarsest, or "
              "the coarsest entry. Wrong answers match no entry. For every base model and every "
              "adapter, most correct answers match the coarsest entry. Adapter counts are means "
              "over three adapters.",
              "tab:location-output-levels", "lrrrr", head, rows, size="\\footnotesize")


def paired_date_changes(blocks):
    """How each adapter answer differs from the base answer to the same date question."""
    from pipeline.score import parse_date
    from ugc.parser.temporal_constants import PRECISION

    def key(v):
        return (v.level, v.year, v.month, v.day, v.decade, v.century, v.season)

    def classify(b, a):
        if b.kind == "abstained" and a.kind == "abstained":
            return "both_abstain"
        if b.kind == "abstained":
            return "base_abstains"
        if a.kind == "abstained":
            return "adapter_abstains"
        vb, va = parse_date(b.text), parse_date(a.text)
        if vb is None or va is None:
            return "unreadable"
        if key(vb) == key(va):
            return "same"
        pb, pa = PRECISION[vb.level], PRECISION[va.level]
        if pa < pb:
            return "coarser"
        if pa > pb:
            return "finer"
        return "same_level_other"

    cats = [("same", "Same answer"), ("coarser", "Coarser level"), ("finer", "Finer level"),
            ("same_level_other", "Same level, other date"), ("unreadable", "Base or adapter answer unreadable"),
            ("base_abstains", "Only the base model abstains"),
            ("adapter_abstains", "Only the adapter abstains"), ("both_abstain", "Both abstain")]
    for m in SIZES:
        b = blocks[(m, "dates", "test")]
        base = b["base"]["dfs"][0].set_index("qid")
        res = {}
        for cond in ("in-domain", "joint"):
            per = []
            for d in b[cond]["dfs"]:
                c = {}
                for _, a in d.iterrows():
                    r = base.loc[a.qid]
                    k = classify(r, a)
                    n, bc, ac = c.get(k, (0, 0, 0))
                    c[k] = (n + 1, bc + bool(r.correct), ac + bool(a.correct))
                per.append(c)
            res[cond] = {k: tuple(np.mean([p.get(k, (0, 0, 0))[i] for p in per]) for i in range(3))
                         for k, _ in cats}
            assert np.isclose(sum(v[0] for v in res[cond].values()), 503)
            for k, v in res[cond].items():
                for i, f in enumerate(("n", "base_correct", "adapter_correct")):
                    reg(f"{m}.dates_test.{cond}.paired.{k}.{f}", v[i])
        head = ("Change & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{LABEL[c]}}}" for c in ("in-domain", "joint"))
                + " \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n & "
                + " & ".join(["Questions", two("Base", "correct"), two("Adapter", "correct")] * 2))
        rows = [f"{name} & " + " & ".join(cnt(x) for cond in ("in-domain", "joint") for x in res[cond][k])
                for k, name in cats]
        who = ("Date answers of the {} in-domain and joint adapters, the adapters trained on "
               "dates, compared with the base answer to the same question on the date test set "
               "(503 questions). ")
        cap = {"7b": who.format("7B") + "Most changes give a coarser level, and they turn many wrong "
                     "base answers into correct ones. Counts are means over three adapters.",
               "3b": who.format("3B") + "Most changes answer questions on which the base model "
                     "abstains, and most of these answers are correct. Counts are means over three "
                     "adapters.",
               "1p5b": who.format("1.5B") + "Most changes give a coarser level, and the adapters "
                       "answer almost all questions on which the base model abstains. Counts are "
                       "means over three adapters."}[m]
        table(f"date_changes_{m}", cap, f"tab:date-changes-{m}", "lrrrrrr", head, rows, size="\\footnotesize",
              sep="4pt")


# ---------------------------------------------------------------- 4.4 sensitivity
def fixed_cost(df, c):
    d = df.copy()
    d.loc[d.wrong.astype(bool), "score"] = -c
    return d


def log_credit(df):
    """Location credit on a log scale (eq:log-location), for credits and penalties alike."""
    d = df.copy()
    s = d.score.astype(float)
    ell = 5 - 4 * s.abs()
    d["score"] = np.where(s != 0, np.sign(s) * (1 - 0.75 * np.log(ell) / np.log(4)), 0.0)
    return d


VARIANTS = [("original", "Score", lambda d: d), ("log", "Log credit", log_credit),
            ("cost0.25", "Cost 0.25", lambda d: fixed_cost(d, 0.25)),
            ("cost0.5", "Cost 0.5", lambda d: fixed_cost(d, 0.5)),
            ("cost1", "Cost 1", lambda d: fixed_cost(d, 1.0))]
JSON_NAME = {"original": "original", "log": "logarithmic", "cost0.25": "fixed_wrong_cost_0.25",
             "cost0.5": "fixed_wrong_cost_0.5", "cost1": "fixed_wrong_cost_1.0"}


def bold_if(ci):
    x = signed(ci["delta"])
    return f"$\\mathbf{{{x}}}$" if ci["lo"] > 0 or ci["hi"] < 0 else f"${x}$"


def sensitivity(blocks):
    sens = json.load(open(C.RESULTS / "sensitivity.json"))["results"]
    head = "Model and set & Adapters & " + " & ".join(v[1] for v in VARIANTS)
    rows = []
    order = [("7b", "locations", "test"), ("7b", "locations", "val"), ("7b", "dates", "test"),
             ("3b", "locations", "test"), ("3b", "dates", "test"),
             ("1p5b", "locations", "test"), ("1p5b", "dates", "test")]
    for i, (m, domain, split) in enumerate(order):
        if i:
            rows.append("\\midrule")
        b = blocks[(m, domain, split)]
        base = b["base"]["dfs"][0]
        for j, cond in enumerate(COND):
            cells = []
            for vk, _, f in VARIANTS:
                if vk == "log" and domain == "dates":
                    cells.append("--")
                    continue
                dfs = [f(d) for d in b[cond]["dfs"]]
                ci = paired_bootstrap(f(base), dfs, "score")
                key = f"{m}.{domain}_{split}.{cond}.sens.{vk}"
                for ff in ("delta", "lo", "hi"):
                    reg(f"{key}.{ff}", ci[ff])
                if split == "test":   # the stored sensitivity file covers the test sets
                    ref = sens[f"{m}_{domain}"]
                    got = np.mean([d.score.mean() for d in dfs]) - f(base).score.mean()
                    want = ref[cond][JSON_NAME[vk]] - ref["base"][JSON_NAME[vk]]
                    assert np.isclose(got, want, atol=1e-12), (m, domain, cond, vk)
                cells.append(bold_if(ci))
            what = f"{NAME[m]}, {SETNAME[(domain, split)].lower()}" if j == 0 else ""
            rows.append(f"{what} & {LABEL[cond]} & " + " & ".join(cells))
    reg("max_log_credit_change", max(
        abs(VALUES[f"{m}.locations_{sp}.{c}.sens.log.delta"] - VALUES[f"{m}.locations_{sp}.{c}.sens.original.delta"])
        for m, _, sp in [o for o in order if o[1] == "locations"] for c in COND))
    reg("max_log_credit_change", max(
        abs(VALUES[f"{m}.locations_{sp}.{c}.sens.log.delta"] - VALUES[f"{m}.locations_{sp}.{c}.sens.original.delta"])
        for m, _, sp in [o for o in order if o[1] == "locations"] for c in COND))
    table("sensitivity",
          "Score differences to the base model under other credits and penalties. Log credit "
          "gives location answers the logarithmic credit of Equation~\\ref{eq:log-location}. "
          "Dates already use a logarithmic credit, so this column applies to locations only. "
          "Cost $c$ gives every wrong answer the score $-c$. Bold differences are significant. "
          "The date gains of the in-domain and joint adapters stay significant under every "
          "choice; the location gains of the 7B model are significant only under some of them. "
          "Test sets have 303 location and 503 date questions; the 7B validation set has 285. "
          "Adapter rows are means over three adapters.",
          "tab:weight-sensitivity", "llrrrrr", head, rows, size="\\footnotesize")


def robustness(blocks):
    """Main comparisons on two subsets: exact-date references, locations without the US."""
    from pipeline.data import load
    from src.dates import TemporalLevel
    from src.text import normalize
    us = {"united states", "united states of america", "usa"}
    keep = {"dates": {it.qid for it in load("dates", "test") if it.gold.level == TemporalLevel.DAY},
            "locations": {it.qid for it in load("locations", "test")
                          if not any(normalize(h) in us for h in it.hierarchy)}}
    assert len(keep["dates"]) == 415 and len(keep["locations"]) == 256
    head = "Model & Answers & Score & Difference & Support match & Difference"
    rows = []
    for i, domain in enumerate(("dates", "locations")):
        if i:
            rows.append("\\midrule")
        title = ("Dates test, exact reference dates only, 415 questions" if domain == "dates"
                 else "Locations test without the United States, 256 questions")
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for m in SIZES:
            b = blocks[(m, domain, "test")]
            sub = lambda d: d[d.qid.isin(keep[domain])].reset_index(drop=True)
            base = sub(b["base"]["dfs"][0])
            conds = [(c, [sub(d) for d in b[c]["dfs"]]) for c in COND]
            if domain == "dates" and m in ("7b", "1p5b"):
                conds.append(("binary", [sub(preds(m, f"dates_s{s}_binary", "dates", "test")) for s in C.SEEDS]))
            s0 = summarize([base])
            key = f"{m}.{domain}_test.{'exact_refs' if domain == 'dates' else 'no_us'}"
            reg(f"{key}.base.score", s0["score"])
            reg(f"{key}.base.exact", s0["exact"])
            rows.append(f"{NAME[m]} & Base & {sgn(s0['score'])} & & {num(s0['exact'])} &")
            for c, dfs in conds:
                s = summarize(dfs)
                ds, dm = paired_bootstrap(base, dfs, "score"), paired_bootstrap(base, dfs, "exact")
                for k, ci in (("d_score", ds), ("d_match", dm)):
                    for f in ("delta", "lo", "hi"):
                        reg(f"{key}.{c}.{k}.{f}", ci[f])
                reg(f"{key}.{c}.score", s["score"])
                reg(f"{key}.{c}.exact", s["exact"])
                lab = "Answer-or-abstain" if c == "binary" else LABEL[c]
                rows.append(f" & {lab} & {sgn(s['score'])} & {interval(ds)} & {num(s['exact'])} & {interval(dm)}")
    table("robustness",
          "The main comparisons of the 7B, 3B, and 1.5B models on two subsets of the test sets. "
          "On dates, the gains of the in-domain and joint adapters remain significant at every "
          "size when only exact reference dates are kept. On locations, no score difference "
          "changes its significance without the questions about the United States. Differences "
          "are paired, to the base model, with 95\\% intervals. Adapter rows are means over "
          "three adapters.",
          "tab:robustness", "llrlrl", head, rows, size="\\scriptsize")


# ---------------------------------------------------------------- 4.5 self-knowledge
ESMA_FAM = [("base", "Base"), ("locations", "Location"), ("dates", "Date"),
            ("joint", "Joint"), ("dates_binary", "Answer-or-abstain")]
FAM_ADAPTERS = {"base": ["base"], **{c: [f"{c}_s{s}" for s in C.SEEDS] for c in ("locations", "dates", "joint")},
                "dates_binary": [f"dates_s{s}_binary" for s in C.SEEDS]}


def esma_signals(m, domain, fam):
    from pipeline.esma import load_signals
    qids = list(preds(m, "base", domain, "test").qid)
    return [load_signals(C.RESULTS / m, a, domain, qids) for a in FAM_ADAPTERS[fam]]


def esma_boot(m, domain):
    """Replicates of the ESMA bootstrap: the same resampled units (people for dates) as
    pipeline/esma.py, as index arrays over the questions."""
    base = pd.read_csv(C.RESULTS / m / "preds" / f"base__{domain}_test__deploy.csv",
                       dtype={"qid": str, "unit": str}, keep_default_na=False)
    groups = {}
    for i, u in enumerate(base.unit):
        groups.setdefault(u, []).append(i)
    groups = [np.array(g) for g in groups.values()]
    member = np.zeros((len(groups), len(base)))
    for g, idx in enumerate(groups):
        member[g, idx] = 1
    boot = np.random.default_rng(C.GLOBAL_SEED).integers(0, len(groups), size=(C.BOOTSTRAP_B, len(groups)))
    return groups, member, boot


def auroc_difference(m, domain, fam):
    """AUROC of the explicit signal, adapters minus base, recomputed in each bootstrap sample."""
    from pipeline.esma import auroc
    groups, _, boot = esma_boot(m, domain)
    base = esma_signals(m, domain, "base")[0]
    sigs = esma_signals(m, domain, fam)
    reps = np.zeros(len(boot))
    for b, draw in enumerate(boot):
        idx = np.concatenate([groups[g] for g in draw])
        a0 = auroc(base["yes_no"][idx], base["c"][idx])
        reps[b] = np.mean([auroc(s["yes_no"][idx], s["c"][idx]) for s in sigs]) - a0
    full = np.mean([auroc(s["yes_no"], s["c"]) for s in sigs]) - auroc(base["yes_no"], base["c"])
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return dict(delta=float(full), lo=float(lo), hi=float(hi))


def cross_difference(m, domain, sig_a, sig_b, corr):
    """Behavioral d'_2 of signal family sig_a minus sig_b, both scored against the direct-answer
    correctness of family corr. Pairs as in the cross-scoring table: matched runs when the
    signal and the correctness come from the same family, all run pairs otherwise."""
    from pipeline.esma import dprime
    _, member, boot = esma_boot(m, domain)
    S = {f: esma_signals(m, domain, f) for f in {sig_a, sig_b, corr}}

    def value(sig):
        pairs = list(zip(S[sig], S[corr])) if sig == corr else [(a, c) for a in S[sig] for c in S[corr]]
        full, reps = [], []
        for a, c in pairs:
            y, cc = a["behavioural"], c["c"]
            cnt = member @ np.stack([y & cc, cc, y & ~cc, ~cc], 1).astype(float)
            full.append(dprime(*cnt.sum(0)))
            r = cnt[boot].sum(1)
            reps.append(dprime(r[:, 0], r[:, 1], r[:, 2], r[:, 3]))
        return np.mean(full), np.mean(reps, 0)
    fa, ra = value(sig_a)
    fb, rb = value(sig_b)
    lo, hi = np.percentile(ra - rb, [2.5, 97.5])
    return dict(delta=float(fa - fb), lo=float(lo), hi=float(hi))


def self_knowledge():
    from pipeline.esma import auroc, dprime
    rows_b, rows_e = [], []
    for i, m in enumerate(("7b", "1p5b")):
        rep = json.load(open(C.RESULTS / m / "report_esma.json"))
        for j, domain in enumerate(("locations", "dates")):
            if i or j:
                rows_b.append("\\midrule")
                rows_e.append("\\midrule")
            n = 303 if domain == "locations" else 503
            text = f"\\textit{{{NAME[m]}, {domain} test, {n} questions}}"
            rows_b.append(f"\\multicolumn{{6}}{{l}}{{{text}}} \\\\")
            rows_e.append(f"\\multicolumn{{8}}{{l}}{{{text}}} \\\\")
            for fam, lab in ESMA_FAM:
                if fam not in rep[domain]["behavioural"]:
                    continue
                sigs = esma_signals(m, domain, fam)
                row_b, row_e = rep[domain]["behavioural"][fam], rep[domain]["explicit"][fam]
                # check the stored full-sample values against the signals
                for key, row in (("behavioural", row_b), ("explicit", row_e)):
                    d = np.mean([dprime((s[key] & s["c"]).sum(), s["c"].sum(), (s[key] & ~s["c"]).sum(),
                                        (~s["c"]).sum()) for s in sigs])
                    assert np.isclose(d, row["dprime"], atol=1e-9), (m, domain, fam, key)
                    hr = np.mean([s[key][s["c"]].mean() for s in sigs])
                    assert np.isclose(hr, row["hit_rate"], atol=1e-12)
                a = np.mean([auroc(s["yes_no"], s["c"]) for s in sigs])
                assert np.isclose(a, row_e["auroc"], atol=1e-12)
                acc = np.mean([s["c"].mean() for s in sigs])
                key = f"{m}.{domain}_test.esma.{fam}"
                reg(f"{key}.direct_acc", acc)
                for sk, row in (("behavioral", row_b), ("explicit", row_e)):
                    for f in ("dprime", "hit_rate", "false_alarm", "yes_rate"):
                        reg(f"{key}.{sk}.{f}", row[f])
                    if "delta" in row:
                        for f in ("delta", "lo", "hi"):
                            reg(f"{key}.{sk}.d.{f}", row["delta"][f])
                reg(f"{key}.explicit.auroc", row_e["auroc"])

                def dp(row):
                    x = f"{row['dprime']:.2f}"
                    return f"${x}" + ("^\\dagger$" if row["clipped"] else "$")

                base_b, base_e = rep[domain]["behavioural"]["base"], rep[domain]["explicit"]["base"]

                def dd(row, base_row):
                    if "delta" not in row:
                        return ""
                    mark = "^\\dagger" if row["clipped"] or base_row["clipped"] else ""
                    return interval(row["delta"], 2)[:-1] + mark + "$"

                if fam == "base":
                    ad = ""
                else:
                    ci = auroc_difference(m, domain, fam)
                    for f in ("delta", "lo", "hi"):
                        reg(f"{key}.explicit.auroc_d.{f}", ci[f])
                    ad = interval(ci)
                rows_b.append(f"{lab} & {pct(acc)} & {num(row_b['hit_rate'])} & {num(row_b['false_alarm'])} & "
                              f"{dp(row_b)} & {dd(row_b, base_b)}")
                rows_e.append(f"{lab} & {pct(row_e['yes_rate'])} & {num(row_e['hit_rate'])} & "
                              f"{num(row_e['false_alarm'])} & {dp(row_e)} & {dd(row_e, base_e)} & "
                              f"{num(row_e['auroc'])} & {ad}")
    common = ("Differences are paired, to the base model, with 95\\% intervals. $^\\dagger$: a hit or "
              "false-alarm rate of this row or of the base model is 0 or 1, so the value depends on "
              "the limit of $10^{-4}$ on the rates (Section~\\ref{sec:method-esma}). Adapter rows "
              "are means over three adapters.")
    table("self_knowledge_behavioral",
          "Behavioral signal of the 1.5B and 7B models: the decision to answer with the deployment "
          "prompt, scored against the correctness of the direct answer. On dates, the 7B date and "
          "joint adapters separate correct from wrong direct answers significantly better than the "
          "base model. " + common,
          "tab:sk-behavioral", "lrrrrl",
          "Answers & Direct correct (\\%) & Hit rate & False-alarm rate & $d'_2$ & Difference", rows_b,
          size="\\footnotesize")
    table("self_knowledge_explicit",
          "Explicit signal of the 1.5B and 7B models: the reply to the self-knowledge question, "
          "scored against the correctness of the direct answer. The AUROC uses the confidence "
          "value $z_q$. On locations, the AUROC barely changes. On dates, it rises significantly "
          "at 7B for the date, joint, and answer-or-abstain adapters, and at 1.5B only for the "
          "answer-or-abstain adapters. " + common,
          "tab:sk-explicit", "lrrrrlrl",
          "Answers & Yes (\\%) & Hit rate & " + two("False-alarm", "rate") + " & $d'_2$ & Difference & "
          "AUROC & " + two("AUROC", "difference"), rows_e,
          size="\\scriptsize", sep="2.5pt")

    # cross-scoring differences stated in the text
    for m, domain, a, b, c in (("7b", "dates", "base", "dates", "dates"),
                               ("7b", "dates", "dates", "base", "base"),
                               ("7b", "locations", "locations", "base", "base"),
                               ("7b", "locations", "joint", "base", "base"),
                               ("1p5b", "locations", "locations", "base", "base"),
                               ("1p5b", "locations", "joint", "base", "base")):
        ci = cross_difference(m, domain, a, b, c)
        for f in ("delta", "lo", "hi"):
            reg(f"{m}.{domain}_test.cross_d.{a}_minus_{b}.on_{c}.{f}", ci[f])

    # cross-scoring: rows = whose signal, columns = whose correctness
    fams = [f for f, _ in ESMA_FAM if f != "dates_binary"]
    short = {"base": "Base", "locations": "Location", "dates": "Date", "joint": "Joint"}

    def cross(m, domain, what):
        sig = {f: esma_signals(m, domain, f) for f in fams}
        out = {}
        for r in fams:
            for c in fams:
                if r == c:
                    pairs = list(zip(sig[r], sig[c]))
                else:
                    pairs = [(a, b) for a in sig[r] for b in sig[c]]
                vals = []
                for a, b in pairs:
                    cc = b["c"]
                    if what == "auroc":
                        vals.append(auroc(a["yes_no"], cc))
                    else:
                        y = a["behavioural"]
                        vals.append(dprime((y & cc).sum(), cc.sum(), (y & ~cc).sum(), (~cc).sum()))
                out[(r, c)] = float(np.mean(vals))
                reg(f"{m}.{domain}_test.cross.{what}.{r}.{c}", out[(r, c)])
        return out

    panels = [("7b", "dates"), ("7b", "locations"), ("1p5b", "dates"), ("1p5b", "locations")]
    res = {(m, d, w): cross(m, d, w) for m, d in panels for w in ("auroc", "behavioral")}
    # the diagonal must equal the main table
    for (m, d, w), out in res.items():
        rep = json.load(open(C.RESULTS / m / "report_esma.json"))[d]
        for f in fams:
            want = rep["explicit"][f]["auroc"] if w == "auroc" else rep["behavioural"][f]["dprime"]
            assert np.isclose(out[(f, f)], want, atol=1e-9), (m, d, w, f)

    def panel_rows(keys, dec):
        head = ("Signal from & " + " & ".join(
            f"\\multicolumn{{4}}{{c}}{{{t}}}" for t in [k[3] for k in keys])
            + " \\\\\n" + "".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(keys)))
            + "\n & " + " & ".join([short[f] for f in fams] * len(keys)))
        rows = []
        for r in fams:
            cells = [f"{res[(m, d, w)][(r, c)]:.{dec[w]}f}" for m, d, w, _ in keys for c in fams]
            rows.append(f"{short[r]} & " + " & ".join(cells))
        return head, rows

    dec = {"auroc": 3, "behavioral": 2}
    for tag, (m, d), cap in (
            ("cross_7b_dates", ("7b", "dates"),
             "Cross-scoring on the 7B date test set (503 questions). Each cell scores the signal "
             "of the row model against the direct-answer correctness of the column model. In "
             "each column, all signals reach about the same AUROC: it depends on whose answers "
             "are scored, not on whose signal is used. The diagonal repeats "
             "Tables~\\ref{tab:sk-behavioral} and~\\ref{tab:sk-explicit}."),):
        head, rows = panel_rows([(m, d, "auroc", "Explicit signal, AUROC"),
                                 (m, d, "behavioral", "Behavioral signal, $d'_2$")], dec)
        table(tag, cap, "tab:cross-7b-dates", "l" + "r" * 8, head, rows, size="\\footnotesize")
    head, rows = [], []
    blocks_rows = []
    for m, d in (("7b", "locations"), ("1p5b", "dates"), ("1p5b", "locations")):
        h, r = panel_rows([(m, d, "auroc", "Explicit signal, AUROC"),
                           (m, d, "behavioral", "Behavioral signal, $d'_2$")], dec)
        blocks_rows.append((f"{NAME[m]}, {d} test", h, r))
    rows = []
    for i, (title, h, r) in enumerate(blocks_rows):
        if i:
            rows.append("\\midrule")
        rows.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{title}}}}} \\\\")
        rows += r
    table("cross_other",
          "Cross-scoring for the 7B location test set (303 questions) and the 1.5B location and "
          "date test sets (303 and 503 questions). Cells as in "
          "Table~\\ref{tab:cross-7b-dates}. On 1.5B dates, the behavioral values in the Date "
          "and Joint rows are set by the limit of Section~\\ref{sec:method-esma}, because these "
          "adapters almost never abstain.",
          "tab:cross-other", "l" + "r" * 8, blocks_rows[0][1], rows, size="\\footnotesize")


# ---------------------------------------------------------------- 4.6 conflicting sources
PROBE_FAM = [("base", "Base"), ("locations", "Location"), ("dates", "Date"), ("joint", "Joint")]
CONFLICT_CELLS = [("conflict", "MONTH", "Month"), ("conflict", "YEAR", "Year"),
                  ("conflict", "DECADE", "Decade"), ("agree", "DAY", "Agreement control"),
                  ("none", "", "No-source control")]
BEHAVIOURS = [("at_level", "At level"), ("at_level_wrong", "At level, wrong"), ("over_commit", "Finer"),
              ("too_coarse", "Coarser"), ("idk", "Abstain"), ("fmt_fail", "Format failure")]


def probe_frames(m, kind, fam):
    ads = ["base"] if fam == "base" else [f"{fam}_s{s}" for s in C.SEEDS]
    out = []
    for a in ads:
        d = pd.read_csv(C.RESULTS / m / "probes" / f"{kind}__{a}.csv", dtype={"qid": str}, keep_default_na=False)
        out.append(d)
    return out


def conflict_cell(d, cond, lvl):
    x = d[(d.condition == cond) & (d.agreement_level == lvl)].copy().reset_index(drop=True)
    x["unit"] = x.qid.str.split("_").str[0]
    x["score"] = x.signed_strict.astype(float)
    x["correct"] = x.correct_strict.astype(str).eq("True").astype(float)
    for b, _ in BEHAVIOURS:
        x[b] = x.behaviour.eq(b).astype(float)
    x["finer_correct"] = (x.behaviour.eq("over_commit") & x.correct.astype(bool)).astype(float)
    return x


def conflict_probe():
    stats = {}
    for m in ("7b", "1p5b"):
        frames = {f: probe_frames(m, "conflict", f) for f, _ in PROBE_FAM}
        keys = None
        for f, fr in frames.items():
            for d in fr:
                assert len(d) == 1227
                k = list(zip(d.qid, d.condition, d.agreement_level))
                keys = keys or k
                assert k == keys, (m, f)
        for cond, lvl, _ in CONFLICT_CELLS:
            base = conflict_cell(frames["base"][0], cond, lvl)
            n = len(base)
            assert n == {"MONTH": 137, "YEAR": 136, "DECADE": 136, "DAY": 409, "": 409}[lvl]
            for f, _ in PROBE_FAM:
                cells = [conflict_cell(d, cond, lvl) for d in frames[f]]
                key = f"{m}.conflict.{cond}_{lvl or 'none'}.{f}"
                s = {c: float(np.mean([x[c].mean() for x in cells]))
                     for c in ["score", "correct", "finer_correct"] + [b for b, _ in BEHAVIOURS]}
                s["answered"] = float(np.mean([x.behaviour.eq("answered").mean() for x in cells]))
                s["n"] = n
                for c, v in s.items():
                    reg(f"{key}.{c}", v)
                if f != "base":
                    for col in ("score", "at_level"):
                        ci = paired_bootstrap(base, cells, col)
                        for ff in ("delta", "lo", "hi"):
                            reg(f"{key}.d_{col}.{ff}", ci[ff])
                        s[f"d_{col}"] = ci
                stats[(m, cond, lvl, f)] = s
    head = ("Model & " + " & ".join(two(*n.split(", ")) if ", " in n else n for _, n in BEHAVIOURS)
            + " & Correct & Score")
    for m in ("7b", "1p5b"):
        rows = []
        for i, (cond, lvl, name) in enumerate(CONFLICT_CELLS):
            if i:
                rows.append("\\midrule")
            title = (f"Sources agree on the {name.lower()}" if cond == "conflict" else name)
            n = stats[(m, cond, lvl, "base")]["n"]
            rows.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{title}, {n} questions}}}} \\\\")
            for j, (f, lab) in enumerate(PROBE_FAM):
                s = stats[(m, cond, lvl, f)]
                if cond == "none":
                    shares = ["--"] * 4 + [pct(s["idk"]), pct(s["fmt_fail"])]
                else:
                    shares = [pct(s[b]) for b, _ in BEHAVIOURS]
                rows.append(f"{lab} & " + " & ".join(shares) + f" & {pct(s['correct'])} & {sgn(s['score'])}")
        cap = {"7b": "Answers of the 7B model with two conflicting date sources in the prompt, in "
                     "percent of the questions of each row group. "
                     "The level of each answer is compared with the level at which the two sources "
                     "agree. Correct and Score grade the answer against the reference. When the "
                     "sources agree on the month or the year, the date and joint adapters answer at "
                     "that level or coarser more often than the base model, and their scores rise. "
                     "Adapter rows are means over three adapters.",
               "1p5b": "Answers of the 1.5B model with two conflicting date sources in the prompt. "
                       "When the sources agree on the month, the date and joint adapters answer at "
                       "the month on most questions, and their scores rise. When the sources agree on "
                       "the year or the decade, they often give a finer date, and their scores fall. "
                       "Columns as in Table~\\ref{tab:conflict-7b}. Adapter rows are means over three "
                       "adapters."}[m]
        table(f"conflict_{m}", cap, f"tab:conflict-{m}", "lrrrrrrrr", head, rows, size="\\footnotesize")
    return stats


def conflict_figure(stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.linewidth": 0.6})
    groups = [("At the shared level", ["at_level"], "#2a78d6"), ("Finer", ["over_commit"], "#eb6834"),
              ("Coarser", ["too_coarse"], "#1baf7a"), ("Abstain", ["idk"], "#eda100"),
              ("Other", ["at_level_wrong", "fmt_fail"], "#b8b7b2")]
    cells = CONFLICT_CELLS[:4]
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.3), sharey=True)
    for r, m in enumerate(("7b", "1p5b")):
        for c, (cond, lvl, name) in enumerate(cells):
            ax = axes[r, c]
            for y, (f, lab) in enumerate(PROBE_FAM):
                s = stats[(m, cond, lvl, f)]
                left = 0.0
                for g, parts, col in groups:
                    w = sum(s[p] for p in parts)
                    ax.barh(y, w, left=left, height=0.62, color=col, edgecolor="white", linewidth=1.0)
                    if w >= 0.12:
                        ax.text(left + w / 2, y, f"{100 * w:.0f}", ha="center", va="center",
                                fontsize=6.5, color="white" if g in ("At the shared level",) else "#0b0b0b")
                    left += w
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.6, 3.6)
            ax.invert_yaxis()
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["0", "50", "100"], color="#52514e")
            ax.tick_params(length=0, pad=2)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color("#b8b7b2")
            if r == 0:
                ax.set_title(("Sources agree\non the " + name.lower()) if cond == "conflict" else
                             "Agreement\ncontrol", fontsize=8)
            if c == 0:
                ax.set_yticks(range(4))
                ax.set_yticklabels([lab for _, lab in PROBE_FAM])
                ax.set_ylabel(NAME[m], rotation=0, ha="right", va="center", fontsize=8, fontweight="bold",
                              labelpad=28)
    for ax in axes[1]:
        ax.set_xlabel("% of questions", color="#52514e")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in groups]
    fig.legend(handles, [g for g, _, _ in groups], loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.55, 1.02), handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.6, h_pad=0.8)
    outdir = C.ROOT / "analysis" / "output" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "conflict_levels.pdf")
    fig.savefig(outdir / "conflict_levels.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- 4.7 other relations
RELATIONS = [("P127", "Owned by"), ("P176", "Manufacturer"), ("P50", "Author"), ("P264", "Record label"),
             ("P112", "Founded by"), ("P170", "Creator"), ("P175", "Performer"), ("P69", "Educated at")]
ENTITY_FAM = [("base", "Base"), ("locations", "Location adapters"), ("dates", "Date adapters"),
              ("joint", "Joint adapters")]


def entity_frames(m, fam):
    out = []
    for d in probe_frames(m, "entity", fam):
        d = d.copy()
        d["unit"] = d.qid
        d["correct"] = d.correct.astype(str).eq("True").astype(float)
        d["abstain"] = d.kind.eq("abstained").astype(float)
        d["wrong"] = ((d.kind != "abstained") & (d.correct == 0)).astype(float)
        d["score"] = d.score.astype(float)
        assert (d.kind != "fmt_fail").all()
        out.append(d)
    return out


def entity_probe():
    sens = json.load(open(C.RESULTS / "sensitivity.json"))["results"]
    head = ("Answers & Score & " + two("Score", "difference") + " & " + PCT3 + " & "
            + two("Wrong difference", "(points)"))
    rows, rows_cost, per_rel = [], [], {}
    head_cost = "Model & Adapters & " + " & ".join(v[1] for v in VARIANTS if v[0] != "log")
    for i, m in enumerate(("7b", "1p5b")):
        if i:
            rows.append("\\midrule")
            rows_cost.append("\\midrule")
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{NAME[m]}, 648 questions}}}} \\\\")
        fr = {f: entity_frames(m, f) for f, _ in ENTITY_FAM}
        base = fr["base"][0]
        assert len(base) == 648
        for f, lab in ENTITY_FAM:
            same_questions(base, fr[f], f"{m} entity {f}")
            s = {k: float(np.mean([d[k].mean() for d in fr[f]])) for k in ("score", "correct", "abstain", "wrong")}
            key = f"{m}.entity.{f}"
            for k, v in s.items():
                reg(f"{key}.{k}", v)
            diff, wdiff = "", ""
            if f != "base":
                ds = paired_bootstrap(base, fr[f], "score")
                dw = paired_bootstrap(base, fr[f], "wrong")
                for k, ci in (("d_score", ds), ("d_wrong", dw)):
                    for ff in ("delta", "lo", "hi", "seed_sd"):
                        reg(f"{key}.{k}.{ff}", ci[ff])
                diff, wdiff = interval(ds), points(dw)
            rows.append(f"{lab.replace(' adapters', '')} & {sgn(s['score'])} & {diff} & {pct(s['correct'])} & "
                        f"{pct(s['abstain'])} & {pct(s['wrong'])} & {wdiff}")
            # fixed wrong-answer costs
            if f != "base":
                cells = []
                for vk, _, fn in VARIANTS:
                    if vk == "log":
                        continue
                    dfs = [fn(d) for d in fr[f]]
                    ci = paired_bootstrap(fn(base), dfs, "score")
                    for ff in ("delta", "lo", "hi"):
                        reg(f"{key}.sens.{vk}.{ff}", ci[ff])
                    got = np.mean([d.score.mean() for d in dfs]) - fn(base).score.mean()
                    ref = sens[f"{m}_entity"]
                    want = ref[f][JSON_NAME[vk]] - ref["base"][JSON_NAME[vk]]
                    assert np.isclose(got, want, atol=1e-12), (m, f, vk)
                    cells.append(bold_if(ci))
                rows_cost.append(f"{NAME[m] if f == 'locations' else ''} & {lab} & " + " & ".join(cells))
            # per relation
            for rel, _ in RELATIONS:
                sub = [d[d.relation == rel] for d in fr[f]]
                n = len(sub[0])
                r = {k: float(np.mean([d[k].mean() for d in sub])) for k in ("score", "correct", "abstain", "wrong")}
                r["n"] = n
                for k, v in r.items():
                    reg(f"{key}.rel.{rel}.{k}", v)
                per_rel[(m, f, rel)] = r
    table("entity",
          "Answers to the questions of the eight entity relations (648 test questions), which no "
          "adapter was trained on. At 7B, the location and joint adapters give fewer wrong "
          "answers at about the same score; at 1.5B, they also raise the score. Base: "
          "Qwen2.5-7B-Instruct and Qwen2.5-1.5B-Instruct with the entity prompt. Adapter rows are "
          "means over three adapters. Differences are paired, to the base model, with 95\\% "
          "intervals.",
          "tab:entity", "lrlrrrl", head, rows, size="\\footnotesize", sep="4pt")
    table("entity_costs",
          "Score differences of the 7B and 1.5B adapters to their base models on the 648 entity "
          "questions under fixed wrong-answer costs (Section~\\ref{sec:weight-design}). Bold "
          "differences are significant. The differences of the location and joint adapters grow "
          "with the cost. Adapter rows are means over three adapters.",
          "tab:entity-costs", "llrrrr", head_cost, rows_cost, size="\\footnotesize")
    # per-relation table for the appendix
    head = ("Relation (questions) & Model & " + " & ".join(
        f"\\multicolumn{{4}}{{c}}{{{NAME[m]}}}" for m in ("7b", "1p5b")) + " \\\\\n"
        "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\n & & " + " & ".join(["Correct", "Abstain", "Wrong", "Score"] * 2))
    rows = []
    for i, (rel, name) in enumerate(RELATIONS):
        if i:
            rows.append("\\midrule")
        for j, (f, lab) in enumerate(ENTITY_FAM):
            cells = []
            for m in ("7b", "1p5b"):
                r = per_rel[(m, f, rel)]
                cells += [pct(r["correct"]), pct(r["abstain"]), pct(r["wrong"]), sgn(r["score"])]
            what = f"{name} ({per_rel[('7b', 'base', rel)]['n']})" if j == 0 else ""
            rows.append(f"{what} & {lab.replace(' adapters', '')} & " + " & ".join(cells))
    table("entity_relations",
          "Answers of the 7B and 1.5B models to the entity questions by relation, in percent of "
          "the questions of each relation. At both sizes, the location and joint adapters give "
          "fewer wrong answers than the base model on every relation. Number of test questions "
          "in brackets. Adapter rows are means over three adapters.",
          "tab:entity-relations", "ll" + "r" * 8, head, rows, size="\\scriptsize", sep="4pt")
    return per_rel


def entity_figure(per_rel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.linewidth": 0.6})
    parts = [("Correct", "correct", "#2a78d6"), ("Abstain", "abstain", "#eda100"), ("Wrong", "wrong", "#eb6834")]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 4.8), sharey=True)
    ticks, labels = [], []
    for c, m in enumerate(("7b", "1p5b")):
        ax = axes[c]
        y = 0.0
        for rel, name in RELATIONS:
            if c == 0:
                ax.text(-0.02, y, f"{name} ({per_rel[(m, 'base', rel)]['n']})", ha="right", va="center",
                        fontsize=7.5, fontweight="bold", transform=ax.get_yaxis_transform())
            y += 1.0
            for f, lab in ENTITY_FAM:
                r = per_rel[(m, f, rel)]
                left = 0.0
                for pname, k, col in parts:
                    ax.barh(y, r[k], left=left, height=0.8, color=col, edgecolor="white", linewidth=0.8)
                    left += r[k]
                if c == 0:
                    ticks.append(y)
                    labels.append(lab.replace(" adapters", ""))
                y += 1
            y += 0.4
        ax.set_xlim(0, 1)
        ax.set_ylim(y - 0.2, -0.8)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "50", "100"], color="#52514e")
        ax.set_xlabel("% of questions", color="#52514e")
        ax.set_title(NAME[m], fontsize=8.5, fontweight="bold")
        ax.tick_params(length=0, pad=2)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#b8b7b2")
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(labels, fontsize=6.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in parts]
    fig.legend(handles, [p for p, _, _ in parts], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.6, 1.0), handlelength=1.2)
    fig.tight_layout(rect=(0.12, 0, 1, 0.96), w_pad=1.0)
    outdir = C.ROOT / "analysis" / "output" / "figures"
    fig.savefig(outdir / "entity_relations.pdf")
    fig.savefig(outdir / "entity_relations.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- summary figure
def differences_figure():
    """Score and support-match differences to the base model, every size and set, with
    95% intervals. Values come from the registry, so they equal the tables."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.linewidth": 0.6})
    colors = {"in-domain": "#2a78d6", "transfer": "#eb6834", "joint": "#1baf7a"}
    rows = {"locations": [("7b", "test", "7B, test"), ("7b", "val", "7B, validation"),
                          ("3b", "test", "3B, test"), ("1p5b", "test", "1.5B, test")],
            "dates": [("7b", "test", "7B, test"), ("3b", "test", "3B, test"), ("1p5b", "test", "1.5B, test")]}
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6), sharey="row",
                             gridspec_kw={"height_ratios": [4, 3], "hspace": 0.55, "wspace": 0.08})
    dodge = {"in-domain": -0.22, "transfer": 0.0, "joint": 0.22}
    for r, domain in enumerate(("locations", "dates")):
        for c, (col, title) in enumerate((("d_score", "Score"), ("d_match", "Support match"))):
            ax = axes[r, c]
            ax.axvline(0, color="#8a8984", lw=0.8, ls=(0, (3, 2)), zorder=1)
            for y, (m, split, _) in enumerate(rows[domain]):
                for cond in COND:
                    k = f"{m}.{domain}_{split}.{cond}.{col}"
                    d, lo, hi = VALUES[k + ".delta"], VALUES[k + ".lo"], VALUES[k + ".hi"]
                    yy = y + dodge[cond]
                    ax.plot([lo, hi], [yy, yy], color=colors[cond], lw=1.5, solid_capstyle="round", zorder=2)
                    ax.plot(d, yy, "o", ms=4.5, color=colors[cond], mec="white", mew=0.8, zorder=3)
            ax.set_ylim(len(rows[domain]) - 0.5, -0.5)
            ax.set_yticks(range(len(rows[domain])))
            ax.set_yticklabels([lab for *_, lab in rows[domain]])
            ax.tick_params(length=0, pad=3, labelcolor="#3a3935")
            ax.tick_params(axis="x", labelcolor="#52514e")
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color("#b8b7b2")
            ax.grid(axis="x", color="#ecebe8", lw=0.6, zorder=0)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            ax.set_title(f"{'Locations' if domain == 'locations' else 'Dates'}: {title.lower()} difference",
                         fontsize=8, loc="left", color="#0b0b0b")
    handles = [plt.Line2D([0], [0], color=colors[c], marker="o", ms=4.5, mec="white", lw=1.5) for c in COND]
    fig.legend(handles, [LABEL[c] for c in COND], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.55, 1.0), handlelength=1.8, columnspacing=1.6)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.88, bottom=0.07)
    outdir = C.ROOT / "analysis" / "output" / "figures"
    fig.savefig(outdir / "differences.pdf")
    fig.savefig(outdir / "differences.png", dpi=200)
    plt.close(fig)


def main():
    blocks = {}
    for m in SIZES:
        for domain, split in (("locations", "test"), ("dates", "test")):
            blocks[(m, domain, split)] = condition_block(m, domain, split)
    blocks[("7b", "locations", "val")] = condition_block("7b", "locations", "val")
    # summary numbers the text states
    gaps = [abs(b["joint"]["score"] - b["in-domain"]["score"]) for b in blocks.values()]
    reg("max_joint_vs_in_domain_score_gap", max(gaps))
    reg("max_score_seed_sd", max(b[c]["d_score"]["seed_sd"] for b in blocks.values() for c in COND))
    reg("max_match_seed_sd", max(b[c]["d_match"]["seed_sd"] for b in blocks.values() for c in COND))
    reg("max_match_seed_sd", max(b[c]["d_match"]["seed_sd"] for b in blocks.values() for c in COND))
    main_tables(blocks)
    diagnostic_3b(blocks)
    support_tables(blocks)
    prompt_baselines(blocks)
    date_controls(blocks)
    answer_levels(blocks)
    paired_date_changes(blocks)
    sensitivity(blocks)
    robustness(blocks)
    self_knowledge()
    conflict_figure(conflict_probe())
    entity_figure(entity_probe())
    differences_figure()
    json.dump(VALUES, open(C.RESULTS / "tables.json", "w"), indent=1, sort_keys=True)
    print(f"wrote {len(list(OUT.glob('*.tex')))} tables, {len(VALUES)} numbers")


if __name__ == "__main__":
    main()
