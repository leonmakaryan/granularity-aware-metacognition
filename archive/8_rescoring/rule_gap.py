"""Compare the capability target rule (the finest level with reliability >= tau)
with the level that maximises I(L) * (2 rho(L) - 1), on the cached curves.
"""
import json
from collections import Counter

TAU = 0.5
PRIOR, FINEST = 5.0, 1.0          # locations information function anchors
CACHES = ["phase2_results/capability_loc.json",
          "phase2_results/capability_loc_7b.json"]


def info(norm_level):
    """I(l) for locations: linear in the normalised (log-granularity) level."""
    return max(0.0, (PRIOR - norm_level) / (PRIOR - FINEST))


def implemented(curve, tau=TAU):
    """Finest level with reliability >= tau, else None (abstain)."""
    for L, r in enumerate(curve):
        if r >= tau:
            return L
    return None


def value(curve, levels, L):
    """Expected signed score of committing at level L."""
    return 0.0 if L is None else info(levels[L]) * (2 * curve[L] - 1)


def optimal(curve, levels):
    """argmax of the expected score, or None when no level is worth committing."""
    best, best_v = None, 0.0
    for L in range(len(curve)):
        v = value(curve, levels, L)
        if v > best_v:
            best, best_v = L, v
    return best


def report(path):
    d = json.load(open(path))
    qs = d["questions"]
    qs = list(qs.values()) if isinstance(qs, dict) else qs
    n = len(qs)
    agree = ties = coarser = 0
    forgone = 0.0
    for r in qs:
        a = implemented(r["curve"])
        b = optimal(r["curve"], r["levels"])
        forgone += value(r["curve"], r["levels"], b) - value(r["curve"], r["levels"], a)
        if a == b:
            agree += 1
        elif b is None and abs(r["curve"][a] - TAU) < 1e-9:
            ties += 1                      # exact tie: both actions score 0
        elif a is not None and b is not None and b > a:
            coarser += 1                   # genuine: a coarser level pays more
        else:
            raise AssertionError(f"unexpected disagreement {a} vs {b}")
    print(f"\n{path}  (n={n}, k={d['k']}, tau={d['tau']})")
    print(f"  rules agree              {agree}/{n} = {agree/n:.3f}")
    print(f"  exact ties at r = tau    {ties}")
    print(f"  optimum strictly coarser {coarser}  ({coarser/n:.3f})")
    print(f"  expected score forgone   {forgone/n:.4f} per question")


if __name__ == "__main__":
    for c in CACHES:
        report(c)
