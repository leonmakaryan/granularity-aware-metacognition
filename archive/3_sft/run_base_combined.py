"""The untrained base model on the 303 location test questions of all six relations,
scored like the trained models, with the answers saved per question.
"""
from collections import defaultdict

import pandas as pd

from src.config import LOCATION_RELATIONS, OUT_DIR
from src.data import load_split
from src.inference import load_tokenizer, load_base_model, greedy_answers
from src.prompt import is_idk
from src.text import match_level
from src.eval import (info_weighted_acc, metacognition, build_level_pool,
                      signed_informativeness, output_level, level_mean_std)

test = load_split("test", LOCATION_RELATIONS)
tok = load_tokenizer()
model = load_base_model()
answers = greedy_answers(model, tok, test)

pool = build_level_pool()
rows = []
for q, a in zip(test, answers):
    idk = is_idk(a)
    ml = match_level(a, q.hierarchy) if (a and not idk) else None
    rows.append((q, idk, ml, a))

pd.DataFrame([{"qid": q.qid, "question": q.question, "hierarchy": q.hierarchy,
               "answer": a, "is_idk": idk, "match_level": ml}
              for q, idk, ml, a in rows]).to_csv(OUT_DIR / "base_combined_eval.csv", index=False)


def info_levels(sub):
    return [q.levels[ml] if (ml is not None and ml < len(q.levels)) else None
            for q, _, ml, _ in sub]


def line(name, sub):
    n = len(sub)
    oh = sum(1 for _, _, ml, _ in sub if ml is not None)
    idkc = sum(1 for _, idk, _, _ in sub if idk)
    mc = metacognition([(not idk, ml is not None) for _, idk, ml, _ in sub])
    signed = sum(signed_informativeness(q, ml, idk, a, pool)
                 for q, idk, ml, a in sub) / n
    lm, ls = level_mean_std([output_level(q, ml, idk, a, pool) for q, idk, ml, a in sub])
    lvl = f"{lm:.2f}±{ls:.2f}" if lm is not None else "  —  "
    d = mc["d_prime"]
    ds = f"{d:.3f}" if d is not None else "  —  "
    ns = f"{mc['nfr']:.3f}" if mc["nfr"] is not None else "  —  "
    print(f"{name:>16s}: n={n:3d}  acc_hier={oh/n:.3f}  info={info_weighted_acc(info_levels(sub)):.3f}  "
          f"info_signed={signed:+.3f}  lvl={lvl}  idk={idkc/n:.3f}  d'={ds}  nfr={ns}")


print("\n=== base_direct on COMBINED test (same prompt as trained eval) ===")
line("ALL", rows)
byrel = defaultdict(list)
for r in rows:
    byrel[r[0].relation].append(r)
print("per-relation:")
for rel in sorted(byrel):
    line(rel, byrel[rel])
