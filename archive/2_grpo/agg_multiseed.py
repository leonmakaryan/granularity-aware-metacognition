"""Mean, spread, min and max over the seeds in a results file.

    python 2_grpo/agg_multiseed.py [results.jsonl]
"""
import sys, json, numpy as np

fname = sys.argv[1] if len(sys.argv) > 1 else "phase2_multiseed_results.jsonl"
rows = [json.loads(l) for l in open(fname) if l.strip()]
rows.sort(key=lambda r: r["seed"])
has_info = any(r.get("info") is not None for r in rows)

print(f"\n================ MULTI-SEED ({fname}, n={len(rows)}) ================")
for r in rows:
    d = r["d_prime"] if r["d_prime"] is not None else float("nan")
    info_s = f"info={r['info']:.3f}  " if r.get("info") is not None else ""
    print(f"seed {r['seed']}: acc_hier={r['acc_hier']:.3f}  sel={r['sel_acc']:.3f}  "
          f"spec={r['spec_acc']:.3f}  {info_s}idk={r['idk']:.3f}  d'={d:.3f}  "
          f"NFR={r['nfr']}  avg_lvl={r['avg_lvl']}")

keys = ["acc_hier", "sel_acc", "spec_acc"] + (["info"] if has_info else []) + ["idk", "d_prime"]
print("\nstat       mean    std     min     max")
for k in keys:
    v = [r[k] for r in rows if r.get(k) is not None]
    if v:
        print(f"{k:9s} {np.mean(v):.3f}  {np.std(v):.3f}  {min(v):.3f}  {max(v):.3f}")
