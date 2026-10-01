"""Tables of the transfer grid (trained on one domain, evaluated on the other) from
phase2_results/transfer_*.json. Format failures are printed first: a model that
fails to transfer writes the other domain's marker.
"""
import json
from pathlib import Path

OUT = Path("phase2_results")

# adapter -> (trained-on domain, human label)
ORIGIN = {
    "base":               ("none",      "base + IDK prompt"),
    "sft_8b":             ("locations", "SFT (locations)"),
    "sft_surgical_8b":    ("locations", "surgical (locations)"),
    "dates_sft_7b_s0w2":  ("dates",     "SFT (dates)"),
    "sft_v2_loc":         ("locations", "SFT v2 (locations)"),
    "dates_sft_s0":       ("dates",     "SFT (dates)"),
    "joint_s0w2":         ("both",      "JOINT (dates+locations)"),
}

COLS = [("acc", "acc"), ("acc_strict", "acc_str"), ("idk", "idk"),
        ("wrong", "wrong"), ("info_signed", "signed"),
        ("info_signed_strict", "signed_str"), ("d", "d'")]


def load_all():
    rows = []
    for f in sorted(OUT.glob("transfer_*.json")):
        if "smoke" in f.name:
            continue
        d = json.load(open(f))
        a, dom = d["adapter"], d["domain"]
        m = {k: v for k, v in d["metrics"].items() if k != "domain"}
        size = "7B" if "7B" in d["model"] else "1.5B"
        origin, label = ORIGIN.get(a, ("?", a))
        rows.append(dict(size=size, adapter=a, origin=origin, label=label,
                         domain=dom, **m))
    return rows


def fmt(v, nd=3):
    if v is None:
        return "-"
    return f"{v:+.{nd}f}" if isinstance(v, float) and abs(v) < 10 else f"{v:.{nd}f}"


def table(rows, size):
    rs = [r for r in rows if r["size"] == size]
    if not rs:
        return
    print(f"\n### {size}\n")
    print("| trained on | evaluated on | fmt_fail | " +
          " | ".join(h for _, h in COLS) + " |")
    print("|---|---|---|" + "---|" * len(COLS))
    order = {"none": 0, "locations": 1, "dates": 2, "both": 3}
    for r in sorted(rs, key=lambda r: (r["domain"], order.get(r["origin"], 9))):
        cells = []
        for k, _ in COLS:
            v = r.get(k)
            cells.append(f"{v:.3f}" if k in ("acc", "acc_strict", "idk", "wrong")
                         and v is not None else fmt(v))
        mark = " **(transfer)**" if r["origin"] not in ("none", r["domain"], "both") else ""
        ff = f"{r['fmt_fail']}/{r['n']}"
        print(f"| {r['label']}{mark} | {r['domain']} | {ff} | " + " | ".join(cells) + " |")


def main():
    rows = load_all()
    if not rows:
        print("no transfer_*.json yet")
        return
    tot_ff = sum(r["fmt_fail"] for r in rows)
    print(f"# Cross-domain transfer grid ({len(rows)} cells)\n")
    print(f"**Format failures across all cells: {tot_ff}.** "
          + ("The marker survives cross-domain, so no cell is a disguised "
             "format break and every number below is a real behavioural result."
             if tot_ff == 0 else
             "NON-ZERO: cells with fmt_fail are partly format breaks, read those "
             "rows as 'the model emitted the wrong marker', not 'it adapted'."))
    for size in ("1.5B", "7B"):
        table(rows, size)
    print("\nlocations test = 303 Q (6 relations, combined). dates test = 503 Q.")
    print("locations cells fp16, dates cells bf16 (each domain's historical convention).")


if __name__ == "__main__":
    main()
