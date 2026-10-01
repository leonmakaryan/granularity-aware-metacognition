import json, numpy as np
from collections import defaultdict
rows = [json.loads(l) for l in open("phase2_sweep_results.jsonl") if l.strip()]
g = defaultdict(list)
for r in rows:
    g[r["idk_reward"]].append(r)
print("\n=========== IDK_REWARD sweep (B'-spec, mean +/- std over seeds) ===========")
print("idk_rew | n |    spec_acc     |     idk%        |     d'          |   acc_hier")
def ms(rs, k):
    v = [r[k] for r in rs if r[k] is not None]
    return (np.mean(v), np.std(v)) if v else (float('nan'), float('nan'))
for idk in sorted(g):
    rs = g[idk]
    sa, ik, dp, ah = ms(rs,"spec_acc"), ms(rs,"idk"), ms(rs,"d_prime"), ms(rs,"acc_hier")
    print(f"  {idk:.1f}   | {len(rs)} | {sa[0]:.3f} +/- {sa[1]:.3f} | {ik[0]:.3f} +/- {ik[1]:.3f} | "
          f"{dp[0]:.2f} +/- {dp[1]:.2f} | {ah[0]:.3f} +/- {ah[1]:.3f}")
print("\nper-run d' (shows basin split):")
for idk in sorted(g):
    print(f"  idk={idk}: " + "  ".join(f"s{r['seed']}={(r['d_prime'] or 0):.2f}" for r in sorted(g[idk], key=lambda r: r['seed'])))
