"""Control for the ESMA-style measures: cross each model's Yes/No signal with each
model's direct-answer correctness. If a base signal predicts a trained model's
correctness as well as that model's own signal, the gain came from the answers changing,
not from better self-knowledge.
    python analysis/esma_cross.py
"""
import numpy as np, pandas as pd
from pipeline import config as C
from pipeline.esma import load_signals, auroc, dprime
import sys
mdir = C.RESULTS / (sys.argv[1] if len(sys.argv) > 1 else "7b")
for domain in ("dates", "locations"):
    qids = list(pd.read_csv(mdir / "preds" / f"base__{domain}_test__deploy.csv", dtype={"qid": str}, keep_default_na=False).qid)
    fam = {"base": ["base"], "dates-trained": [f"dates_s{s}" for s in range(3)],
           "joint": [f"joint_s{s}" for s in range(3)], "locations-trained": [f"locations_s{s}" for s in range(3)]}
    sig = {k: [load_signals(mdir, a, domain, qids) for a in v] for k, v in fam.items()}
    def cell(sig_from, c_from, what):
        vals = []
        for a in sig[sig_from]:
            for b in sig[c_from]:
                c = b["c"]
                if what == "auroc":
                    vals.append(auroc(a["yes_no"], c))
                else:
                    y = a["behavioural"]
                    vals.append(dprime((y & c).sum(), c.sum(), (y & ~c).sum(), (~c).sum()))
        return np.mean(vals)
    for what, label in (("auroc", "explicit AUROC (meta log-odds)"), ("dprime", "behavioural d' (commit vs IDK)")):
        print(f"\n{mdir.name} {domain}: {label}; rows = whose signal, columns = whose correctness")
        cols = list(fam)
        print(f"{'':20s}" + "".join(f"{c:>19s}" for c in cols))
        for r in cols:
            print(f"{r:20s}" + "".join(f"{cell(r, c, what):19.3f}" for c in cols))
