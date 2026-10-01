"""Flag any stored evaluation written before the adapter it claims to describe,
by comparing file times. It found three date evaluations whose adapters had been
retrained after the evaluation was saved.
"""

import json
from datetime import datetime
from pathlib import Path

OUT = Path("phase2_results")
ADAPTERS = [OUT / "run8b_artifacts", OUT / "checkpoint"]
SLACK = 3600          # a CSV up to an hour older than its adapter is just run order


def mtime(p):
    return p.stat().st_mtime if p and p.exists() else None


def stamp(t):
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") if t else "-"


def find_adapter(name):
    for root in ADAPTERS:
        d = root / name
        if d.is_dir():
            for f in ("adapter_model.safetensors", "adapter_model.bin",
                      "adapter_config.json"):
                if (d / f).exists():
                    return d / f
    return None


def main():
    rows, stale = [], []
    for jpath in sorted(OUT.glob("*.json")):
        try:
            meta = json.loads(jpath.read_text())
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        adapter_name = meta.get("adapter")
        if not adapter_name or adapter_name == "base":
            continue
        cpath = jpath.with_suffix(".csv")
        apath = find_adapter(adapter_name)
        jt, ct, at = mtime(jpath), mtime(cpath), mtime(apath)
        row = dict(artifact=jpath.name, adapter=adapter_name,
                   adapter_found=apath is not None,
                   adapter_mtime=stamp(at), json_mtime=stamp(jt), csv_mtime=stamp(ct))
        # The CSV must not predate the adapter it claims to describe.
        if at and ct and ct < at - SLACK:
            row["verdict"] = "STALE CSV"
            row["csv_older_by_hours"] = round((at - ct) / 3600, 1)
            stale.append(row)
        elif not apath:
            row["verdict"] = "adapter missing"
        elif not ct:
            row["verdict"] = "no csv"
        else:
            row["verdict"] = "ok"
        rows.append(row)

    width = max(len(r["artifact"]) for r in rows) if rows else 10
    print(f"{'artifact':{width}}  {'adapter':26} {'adapter':16} {'json':16} "
          f"{'csv':16} verdict")
    for r in sorted(rows, key=lambda r: (r["verdict"] == "ok", r["artifact"])):
        print(f"{r['artifact']:{width}}  {r['adapter'][:25]:26} "
              f"{r['adapter_mtime']:16} {r['json_mtime']:16} "
              f"{r['csv_mtime']:16} {r['verdict']}")

    print(f"\n{len(rows)} artifacts checked, {len(stale)} with a stale CSV")
    for r in stale:
        print(f"  {r['artifact']}: CSV is {r['csv_older_by_hours']}h older than "
              f"adapter {r['adapter']} -> the CSV describes a previous, "
              f"now-overwritten training run")
    path = OUT / "artifact_provenance_sep09.json"
    path.write_text(json.dumps(dict(checked=len(rows), stale=len(stale), rows=rows),
                               indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
