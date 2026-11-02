"""Collect SRNXR benchmark and accuracy results into CSV, JSON, Markdown tables and a Pareto plot.

  python srnxr_report.py out_dir --perf DIR [DIR ...] --accuracy compare.json [--tag NAME]

--perf     directories with the srnxr_bench / srnxr_coopvec JSON files named
           <model>_<workload>_<config>.json
--accuracy srnxr_compare.py JSON reports (one per model); the backend names in them must
           match the configs of the perf files to appear in the Pareto plot
Only measured values are written, missing combinations are reported as NOT TESTED.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--perf", nargs="+", default=[])
    ap.add_argument("--accuracy", nargs="*", default=[])
    ap.add_argument("--pareto-batch", type=int, default=65536)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for d in args.perf:
        tag = os.path.basename(os.path.normpath(d))
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            name = os.path.splitext(os.path.basename(path))[0]
            parts = name.split("_", 2)
            if len(parts) < 3:
                continue
            model = parts[0]
            rest = name[len(model) + 1:]
            for w in ("one_leaf", "coherent", "random", "overlap"):
                if rest.startswith(w + "_"):
                    workload, config = w, rest[len(w) + 1:]
                    break
            else:
                continue
            try:
                j = json.load(open(path))
            except json.JSONDecodeError:
                continue
            for b in j.get("batches", []):
                rows.append({
                    "build": tag, "model": model, "workload": workload, "config": config,
                    "gpu": j.get("gpu"), "sm": j.get("sm"), "driver": j.get("cuda_driver_version"),
                    "registers": j.get("kernel_registers"), "local_bytes": j.get("kernel_local_bytes"),
                    "block": j.get("block_size"), "batch": b["batch"], "median_ms": b["median_ms"],
                    "best_ms": b["best_ms"], "us_per_query": b["us_per_query"],
                    "mqueries_per_s": b["mqueries_per_s"],
                })
    with open(os.path.join(args.out_dir, "perf.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(args.out_dir, "perf.json"), "w") as f:
        json.dump(rows, f, indent=1)

    # Markdown: Mqueries/s at the largest batch and us/query at batch 1, per build/model/workload
    md = []
    builds = sorted({r["build"] for r in rows})
    configs = sorted({r["config"] for r in rows})
    for build in builds:
        for model in sorted({r["model"] for r in rows if r["build"] == build}):
            md.append(f"\n#### {build}, model {model}\n")
            md.append("| workload | " + " | ".join(configs) + " |")
            md.append("|---|" + "---|" * len(configs))
            for wl in ("one_leaf", "coherent", "random", "overlap"):
                cells = []
                for c in configs:
                    big = [r for r in rows if r["build"] == build and r["model"] == model and
                           r["workload"] == wl and r["config"] == c and
                           r["batch"] == args.pareto_batch]
                    one = [r for r in rows if r["build"] == build and r["model"] == model and
                           r["workload"] == wl and r["config"] == c and r["batch"] == 1]
                    if big:
                        cells.append(f"{big[0]['mqueries_per_s']:.2f} Mq/s, "
                                     f"{one[0]['us_per_query']:.0f} us@1" if one else
                                     f"{big[0]['mqueries_per_s']:.2f} Mq/s")
                    else:
                        cells.append("NOT TESTED")
                md.append(f"| {wl} | " + " | ".join(cells) + " |")
    with open(os.path.join(args.out_dir, "perf_tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    # Pareto: worst-field NRMSE (all points) against throughput on the coherent workload
    acc = {}
    for path in args.accuracy:
        j = json.load(open(path))
        model = os.path.basename(path).split("_")[0]
        for name, comp in j["comparisons"].items():
            if "all" in comp:
                acc[(model, name)] = max(v["nrmse"] for v in comp["all"].values())
    points = []
    for (model, name), err in acc.items():
        for r in rows:
            if r["model"] == model and r["config"] == name and r["workload"] == "coherent" and \
                    r["batch"] == args.pareto_batch:
                points.append((r["build"], model, name, err, r["mqueries_per_s"]))
    with open(os.path.join(args.out_dir, "pareto.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["build", "model", "config", "worst_field_nrmse", "mqueries_per_s_coherent"])
        w.writerows(points)
    if points:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for build, model, name, err, mq in points:
            ax.scatter(mq, err)
            ax.annotate(f"{name} ({build})", (mq, err), fontsize=7, xytext=(4, 2),
                        textcoords="offset points")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Mqueries/s, coherent workload, batch {args.pareto_batch}")
        ax.set_ylabel("worst-field NRMSE vs FP16 oracle")
        ax.set_title("Accuracy vs speed (SYNTHETIC model)")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "pareto.png"), dpi=110)
    print(f"{len(rows)} perf rows, {len(points)} pareto points -> {args.out_dir}")


if __name__ == "__main__":
    main()
