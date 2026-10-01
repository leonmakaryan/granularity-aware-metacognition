import json
from collections import Counter
import pandas as pd
from pipeline.data import load, holdout
from pipeline.score import evidence, extract, valid_levels, level_name
from pipeline import config as C


def stats(it, raws):
    return [evidence(it, r) for r in raws], [extract(it, r)[0] for r in raws]


def read_samples(dom):
    out = {}
    for line in open(f"results/sep11/7b/samples/{dom}_train.jsonl"):
        r = json.loads(line)
        out[r["qid"]] = r["raws"]
    return out


# A. On the pilot holdout, how strong was the support behind each outcome?
d = "results/sep11/7b/preds/"
b = pd.read_csv(d + "base__locations_train-holdout__deploy.csv", dtype={"qid": str})
p = pd.read_csv(d + "pilot_locations_s0__locations_train-holdout__deploy.csv", dtype={"qid": str})
loc_samples = read_samples("locations")
held = {it.qid: it for it in holdout(load("locations", "train"), C.HOLDOUT_FRAC, C.GLOBAL_SEED)[1]}
rows = []
for bq, pq in zip(b.itertuples(), p.itertuples()):
    if pd.isna(bq.support):
        continue
    ev, kinds = stats(held[bq.qid], loc_samples[bq.qid])
    level = int(bq.support)
    rows.append(dict(outcome=pq.category,
                     correct_of_8=sum(1 for e in ev if e is not None and e <= level),
                     declined_of_8=sum(1 for k in kinds if k == "abstained")))
df = pd.DataFrame(rows)
print("A. supported holdout questions: correct samples at the supported level (of 8), by pilot outcome")
summary = df.groupby("outcome").agg(n=("correct_of_8", "size"), mean_correct=("correct_of_8", "mean"),
                                    min_correct=("correct_of_8", "min"), mean_declined=("declined_of_8", "mean"))
print(summary.round(2).to_string())
wrongly = df[df.outcome == "abstained_despite_support"]
print("   abstained-despite-support, correct samples:", sorted(wrongly.correct_of_8.tolist()))
print("   abstained-despite-support, declining samples:", sorted(wrongly.declined_of_8.tolist()))


# B. Training labels under the current rule vs judging only the samples that answered.
def rule_now(it, ev, kinds):
    for L in valid_levels(it):
        if sum(1 for e in ev if e is not None and e <= L) / len(ev) >= 0.5:
            return L
    return None


def rule_answered_only(it, ev, kinds):
    answered = [e for e, k in zip(ev, kinds) if k == "answered"]
    if len(answered) < len(ev) / 2:
        return None
    for L in valid_levels(it):
        if sum(1 for e in answered if e is not None and e <= L) / len(answered) >= 0.5:
            return L
    return None


print()
print("B. training labels: current rule vs judging only samples that answered (needs >= 4 of 8 to answer)")
for dom in ("locations", "dates"):
    smp = read_samples(dom)
    now, alt, moved = Counter(), Counter(), Counter()
    for it in load(dom, "train"):
        ev, kinds = stats(it, smp[it.qid])
        a, c = rule_now(it, ev, kinds), rule_answered_only(it, ev, kinds)
        now[level_name(it, a)] += 1
        alt[level_name(it, c)] += 1
        if a != c:
            if a is None:
                moved["abstain -> answer"] += 1
            elif c is None:
                moved["answer -> abstain"] += 1
            else:
                moved["level change"] += 1
    n = sum(now.values())
    a_now, a_alt = now["abstain"], alt["abstain"]
    print(f"   {dom}: abstain labels {a_now} ({a_now / n:.1%}) -> {a_alt} ({a_alt / n:.1%}); changes {dict(moved)}")
