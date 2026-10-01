"""Synthetic tests of the analysis code: report and paired bootstrap, the pilot go/no-go,
and the scorer check. No GPU, no real predictions; writes only under the given directory.

    python -m pipeline.analysis_tests /tmp/some_dir

Includes exact cases (a constant +0.1 shift must give delta and interval exactly 0.1)
and the Gate 2 cases that exposed the first pattern detector: three same-domain errors
spread over three strata must block, two must not."""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent; SCR = Path(sys.argv[1]) / "analysis_test"
sys.path.insert(0, str(REPO))
from pipeline import config as C
shutil.rmtree(SCR, ignore_errors=True)
C.ROOT = SCR; C.RESULTS = SCR / "results"
(SCR).mkdir(parents=True); (SCR / "CODE_COMMIT").write_text("test\n")
from pipeline import report, run, scorer_check
from pipeline.support import CATEGORIES
fails = []
def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok: fails.append(msg)

rng = np.random.default_rng(1)
def fake(n, domain, p_correct):
    qids = [f"q{i:04d}" for i in range(n)]
    units = [f"p{i // 2}" for i in range(n)] if domain == "dates" else qids
    kind = rng.choice(["answered", "abstained", "fmt_fail"], p=[.7, .28, .02], size=n)
    correct = (kind == "answered") & (rng.random(n) < p_correct)
    score = np.where(kind == "abstained", 0.0, np.where(correct, rng.uniform(.25, 1, n), -rng.uniform(.25, 1, n)))
    cats = rng.choice(CATEGORIES, size=n)
    return pd.DataFrame(dict(qid=qids, unit=units, relation="", question="", raw="", kind=kind, text="",
                             correct=correct, level=0, score=score, has_support=True, support=0.0,
                             category=cats, exact=(cats == "exact_support_match").astype(float)))

print("bootstrap exactness")
b = fake(303, "locations", .6)
s = b.copy(); s["score"] = s["score"] + 0.1
r = report.paired_bootstrap(b, [s, s, s], "score")
check(abs(r["delta"] - .1) < 1e-12 and abs(r["lo"] - .1) < 1e-9 and abs(r["hi"] - .1) < 1e-9 and r["seed_sd"] == 0,
      f"constant +0.1 shift -> delta {r['delta']:.6f} CI [{r['lo']:.6f}, {r['hi']:.6f}] sd {r['seed_sd']}")
r0 = report.paired_bootstrap(b, [b.copy()] * 3, "score")
check(r0["delta"] == 0 and r0["lo"] == 0 and r0["hi"] == 0, "identical seeds -> delta 0, CI [0, 0]")
s1, s2 = b.copy(), b.copy(); s1["score"] += .05; s2["score"] += .15
r2 = report.paired_bootstrap(b, [s1, s2], "score")
check(abs(r2["delta"] - .1) < 1e-12 and abs(r2["seed_sd"] - .05) < 1e-12, f"seed deltas .05/.15 -> mean {r2['delta']:.3f}, sd {r2['seed_sd']:.3f}")
bd = fake(503, "dates", .6); sd_ = fake(503, "dates", .75)
rp = report.paired_bootstrap(bd, [sd_], "score")
bq = bd.copy(); bq["unit"] = bq["qid"]; sq = sd_.copy(); sq["unit"] = sq["qid"]
rq = report.paired_bootstrap(bq, [sq], "score")
check(rp["delta"] == rq["delta"], "resampling unit changes the interval, not the point estimate")

print("full dev report on synthetic predictions")
pdir = C.RESULTS / "7b" / "preds"; pdir.mkdir(parents=True)
for domain, n in (("locations", 303), ("dates", 503)):
    base = fake(n, domain, .55); base.to_csv(pdir / f"base__{domain}_test__deploy.csv", index=False)
    for cond in C.CONDITIONS:
        for seed in C.SEEDS:
            df = base.copy(); flip = rng.random(n) < .15
            df.loc[flip, "score"] = rng.uniform(-.5, 1, flip.sum())
            df.loc[flip, "category"] = "exact_support_match"; df["exact"] = (df.category == "exact_support_match").astype(float)
            df.to_csv(pdir / f"{cond}_s{seed}__{domain}_test__deploy.csv", index=False)
try:
    report.main("7b", "dev")
    rep = json.loads((C.RESULTS / "7b" / "report_dev.json").read_text())
    check(set(rep) == {"locations_test", "dates_test", "manifest", "gates"},
          "report has both dev sets, a manifest and the gate records")
    check(set(rep["locations_test"]) == {"base", "in-domain", "transfer", "joint"}, "rows: base, in-domain, transfer, joint")
    check(len(rep["manifest"]["files"]) == 20 and rep["manifest"]["commit"].endswith("(CODE_COMMIT)"),
          f"manifest hashes {len(rep['manifest']['files'])} prediction files, commit {rep['manifest']['commit'][:20]}")
    check("frozen_commit" in rep["manifest"], "manifest records the freeze marker separately")
    g1 = C.RESULTS / "1p5b" / "gates"; g1.mkdir(parents=True)
    g7 = C.RESULTS / "7b" / "gates"; g7.mkdir(parents=True, exist_ok=True)
    (g7 / "gate4_batching.json").write_text('{"canonical_batch": 32}')
    (g7 / "pilot_selection.json").write_text('{"chosen": 2}')
    (g1 / "gate4_batching.json").write_text('{"canonical_batch": 1}')
    shutil.copytree(pdir, C.RESULTS / "1p5b" / "preds")
    report.main("1p5b", "dev")
    g = json.loads((C.RESULTS / "1p5b" / "report_dev.json").read_text())["gates"]
    check(g.get("gate4_batching.json") == {"canonical_batch": 1}
          and "7b/gate4_batching.json" not in g
          and g.get("7b/pilot_selection.json") == {"chosen": 2},
          f"a second model keeps its own Gate 4 and marks inherited gates: {sorted(g)}")
    t = rep["locations_test"]["in-domain"]
    check(t["delta_exact"]["lo"] <= t["delta_exact"]["delta"] <= t["delta_exact"]["hi"], "point estimate inside its interval")
except Exception as e:
    check(False, f"report crashed: {type(e).__name__}: {e}")

print("pilot go/no-go")
hb = fake(290, "locations", .55); hb["support"] = np.where(rng.random(290) < .45, np.nan, 1.0)
hp = hb.copy(); hp["exact"] = np.minimum(1, hp["exact"] + (rng.random(290) < .3)); hp["kind"] = np.where(rng.random(290) < .45, "abstained", "answered")
hb.to_csv(pdir / "base__locations_train-holdout__deploy.csv", index=False)
hp.to_csv(pdir / "pilot_locations_s0__locations_train-holdout__deploy.csv", index=False)
try:
    run.cmd_pilot_check(argparse.Namespace(model="7b", limit=None, adapter="pilot_locations_s0"))
    code = 0
except SystemExit as e:
    code = e.code
g = json.loads((C.RESULTS / "7b" / "gates" / "gate5_pilot_locations_s0.json").read_text())
expect = g["pilot_match"] > g["base_match"] and abs(g["pilot_abstain"] - g["target_abstain_share"]) <= .10
check((code == 0) == expect and g["go"] == expect, f"go={g['go']} exit={code} (match {g['base_match']:.3f}->{g['pilot_match']:.3f}, abstain {g['pilot_abstain']:.3f} vs target {g['target_abstain_share']:.3f})")

print("pilot selection")
adir = C.RESULTS / "7b" / "adapters"
for name, ep in (("pilot_locations_s0", 2), ("pilot_locations_s0_e1", 1)):
    (adir / name).mkdir(parents=True, exist_ok=True)
    (adir / name / "train_log.json").write_text(json.dumps([{"loss": .1, "epoch": ep / 2}, {"train_loss": .1, "epoch": float(ep)}]))
def pilot_preds(name, exact, score):
    df = hb.copy(); df["exact"] = exact; df["score"] = score
    df.to_csv(pdir / f"{name}__locations_train-holdout__deploy.csv", index=False)
pilot_preds("pilot_locations_s0", 0.0, 0.2); pilot_preds("pilot_locations_s0_e1", 1.0, -0.5)
run.cmd_pilot_select(argparse.Namespace(model="7b", limit=None))
sel = json.loads((C.RESULTS / "7b" / "gates" / "pilot_selection.json").read_text())["chosen"]
check(sel["adapter"] == "pilot_locations_s0_e1" and sel["epochs"] == 1, "higher support match wins despite lower score")
pilot_preds("pilot_locations_s0", 0.5, 0.2); pilot_preds("pilot_locations_s0_e1", 0.5, 0.1)
run.cmd_pilot_select(argparse.Namespace(model="7b", limit=None))
sel = json.loads((C.RESULTS / "7b" / "gates" / "pilot_selection.json").read_text())["chosen"]
check(sel["adapter"] == "pilot_locations_s0" and sel["epochs"] == 2, "a tie on support match is broken by signed score")

print("scorer check")
scorer_check.CHECK = C.RESULTS / "scorer_check"; scorer_check.CHECK.mkdir(parents=True)
sheet = pd.read_csv(REPO / "results/scorer_check/sheet.csv", dtype=str, keep_default_na=False)
key = pd.read_csv(REPO / "results/scorer_check/key.csv", dtype=str, keep_default_na=False)
perfect = sheet.copy(); perfect["human_outcome"] = key["scorer_outcome"]; perfect["human_level"] = key["scorer_level"]
perfect.to_csv(scorer_check.CHECK / "sheet.csv", index=False); key.to_csv(scorer_check.CHECK / "key.csv", index=False)
scorer_check.score(); res = json.loads((scorer_check.CHECK / "result.json").read_text())
check(res["passed"] and res["outcome_agreement"] == 1 and abs(res["outcome_kappa"] - 1) < 1e-12, "perfect annotation passes, kappa 1")
loc_correct = key[(sheet.domain == "locations") & (key.scorer_outcome == "correct")]
spread = loc_correct.groupby("stratum").head(1).index[:3]
bad = perfect.copy(); bad.loc[spread, "human_outcome"] = "wrong"
bad.to_csv(scorer_check.CHECK / "sheet.csv", index=False)
scorer_check.score(); res = json.loads((scorer_check.CHECK / "result.json").read_text())
check(not res["passed"] and "locations/correct->wrong" in res["systematic_patterns"],
      f"3 same-domain errors across strata {sorted(key.loc[spread, 'stratum'])} block: {res['systematic_patterns']}")
two = perfect.copy(); two.loc[spread[:2], "human_outcome"] = "wrong"
two.to_csv(scorer_check.CHECK / "sheet.csv", index=False)
scorer_check.score(); res = json.loads((scorer_check.CHECK / "result.json").read_text())
check(res["passed"], f"2 errors of one pattern do not block: agreement {res['outcome_agreement']:.3f}")
lvl = perfect.copy(); rows = loc_correct.index[:3]
lvl.loc[rows, "human_level"] = (key.loc[rows, "scorer_level"].astype(int) + 1).astype(str)
lvl.to_csv(scorer_check.CHECK / "sheet.csv", index=False)
scorer_check.score(); res = json.loads((scorer_check.CHECK / "result.json").read_text())
check(not res["passed"] and "locations/level:scorer_finer" in res["systematic_patterns"],
      f"3 level inflations block regardless of which pair: {res['systematic_patterns']}")
print("\nANALYSIS TESTS " + ("PASSED" if not fails else f"FAILED: {fails}"))
