"""How stable are the training targets if the eight samples were drawn again?

    python analysis/target_stability.py

Offline stand-in for a second sampling run: for every training question, resample its
eight saved answers with replacement (B times) and recompute the target with the
50%-support rule. Reports, per model and domain, the mean share of resamples whose
target equals the original target. Scoring as in training (JULIAN_CREDIT off).
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import config as C
from pipeline import score
from pipeline.data import load
from pipeline.score import evidence, valid_levels

B = 200
score.JULIAN_CREDIT = False


def target(item, ev):
    for L in valid_levels(item):
        if sum(e is not None and e <= L for e in ev) / len(ev) >= C.SUPPORT_THRESHOLD:
            return L
    return None


def main():
    rng = random.Random(C.GLOBAL_SEED)
    for m in ("7b", "3b", "1p5b"):
        for d in ("locations", "dates"):
            items = load(d, "train")
            raw = {str(r["qid"]): r["raws"]
                   for r in map(json.loads, open(C.RESULTS / m / "samples" / f"{d}_train.jsonl"))}
            agree = []
            for it in items:
                ev = [evidence(it, r) for r in raw[it.qid]]
                t = target(it, ev)
                same = sum(target(it, [rng.choice(ev) for _ in ev]) == t for _ in range(B))
                agree.append(same / B)
            print(f"{m} {d}: target unchanged in {sum(agree) / len(agree):.1%} of resamples "
                  f"({len(items)} questions, B={B})")


if __name__ == "__main__":
    main()
