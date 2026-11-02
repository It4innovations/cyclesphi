"""Accuracy comparison of decoder outputs against the numpy oracle.

  python srnxr_compare.py model.srnxr points_accuracy.f32 --json out.json --csv out.csv \
      NAME=values.f32 [NAME=values.f32 ...]

References computed here (srnxr_reference.decode):
  fp32_master   float64 arithmetic, FP32 master weights (when the file has them)
  fp16_oracle   float64 arithmetic, the deployed FP16 weights
  fp16_act16    float64 arithmetic, FP16 weights, activations rounded to FP16 per layer
                (what the matrix units consume)

Every result is compared with fp16_oracle (backend arithmetic error); fp16_oracle is compared
with fp32_master (storage quantization error). Metrics per field and point category:
MAE, RMSE, NRMSE (RMSE / field range), P99 and max absolute error, mean relative error over
|ref| > 1e-3 * range (near-zero policy), PSNR over the field range.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

import srnxr_format as fmt
import srnxr_reference as ref


def metrics(test: np.ndarray, truth: np.ndarray, field_range: np.ndarray) -> dict:
    err = np.abs(test - truth)
    mse = np.mean(err ** 2, axis=0)
    rmse = np.sqrt(mse)
    out = {
        "mae": err.mean(axis=0),
        "rmse": rmse,
        "nrmse": rmse / field_range,
        "p99_abs": np.percentile(err, 99, axis=0),
        "max_abs": err.max(axis=0),
        "psnr_db": 20 * np.log10(field_range / np.maximum(rmse, 1e-30)),
    }
    rel = np.full(test.shape[1], np.nan)
    for k in range(test.shape[1]):
        mask = np.abs(truth[:, k]) > 1e-3 * field_range[k]
        if mask.any():
            rel[k] = np.mean(err[mask, k] / np.abs(truth[mask, k]))
    out["mean_rel_nonzero"] = rel
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("points")
    ap.add_argument("results", nargs="*")
    ap.add_argument("--categories", help="json with index ranges (default: <points>.json)")
    ap.add_argument("--json")
    ap.add_argument("--csv")
    ap.add_argument("--cache", help="npz cache of the references")
    args = ap.parse_args()

    m = fmt.read(args.model)
    p = np.fromfile(args.points, dtype=np.float32).reshape(-1, 4)
    cat_path = args.categories or os.path.splitext(args.points)[0] + ".json"
    cats = json.load(open(cat_path))["categories"] if os.path.exists(cat_path) else {
        "all": [0, p.shape[0]]}
    cats = dict(cats)
    cats["all"] = [0, p.shape[0]]

    if args.cache and os.path.exists(args.cache):
        c = np.load(args.cache)
        refs = {k: c[k] for k in c.files if k != "inside"}
        inside = c["inside"]
    else:
        refs = {}
        refs["fp16_oracle"], inside = ref.decode(m, p)
        refs["fp16_act16"], _ = ref.decode(m, p, act_fp16=True)
        if m.weights_fp32 is not None:
            refs["fp32_master"], _ = ref.decode(m, p, use_master=True)
        if args.cache:
            np.savez(args.cache, inside=inside, **refs)

    oracle = refs["fp16_oracle"]
    field_range = oracle[inside].max(axis=0) - oracle[inside].min(axis=0)
    field_range = np.where(field_range > 0, field_range, 1.0)

    comparisons = []
    if "fp32_master" in refs:
        comparisons.append(("storage_fp16_vs_fp32_master", oracle, refs["fp32_master"]))
    comparisons.append(("emulated_fp16_activations_vs_oracle", refs["fp16_act16"], oracle))
    for item in args.results:
        name, path = item.split("=", 1)
        v = np.fromfile(path, dtype=np.float32).reshape(p.shape[0], m.output_dim).astype(np.float64)
        bad = ~np.isfinite(v)
        if bad.any():
            print(f"{name}: {bad.any(axis=1).sum()} points with non-finite output")
        comparisons.append((name, v, oracle))

    rows = []
    report = {"model": args.model, "points": args.points, "fields": m.field_names,
              "synthetic_model": bool(m.metadata.get("synthetic")),
              "field_range": dict(zip(m.field_names, field_range.tolist())),
              "outside_points": int((~inside).sum()), "comparisons": {}}
    for name, test, truth in comparisons:
        report["comparisons"][name] = {}
        # Outside the domain every backend must return exactly the outside value.
        outside_mismatch = int(np.sum(np.any(test[~inside] != truth[~inside], axis=1)))
        report["comparisons"][name]["outside_value_mismatches"] = outside_mismatch
        for cat, (a, b) in cats.items():
            sel = np.zeros(p.shape[0], bool)
            sel[a:b] = True
            sel &= inside
            if not sel.any():
                continue
            mt = metrics(test[sel], truth[sel], field_range)
            report["comparisons"][name][cat] = {
                f: {k: float(v[i]) for k, v in mt.items()} for i, f in enumerate(m.field_names)}
            for i, f in enumerate(m.field_names):
                rows.append([name, cat, f, int(sel.sum())] + [float(mt[k][i]) for k in mt])
    header = ["comparison", "category", "field", "points", "mae", "rmse", "nrmse", "p99_abs",
              "max_abs", "psnr_db", "mean_rel_nonzero"]
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    # console summary: all points, worst field
    print(f"{'comparison':48s} {'category':18s} {'worst nrmse':>12s} {'max abs':>10s} {'psnr':>7s}")
    for name, _, _ in comparisons:
        for cat in cats:
            r = report["comparisons"][name].get(cat)
            if not r:
                continue
            worst = max(r, key=lambda f: r[f]["nrmse"])
            print(f"{name:48s} {cat:18s} {r[worst]['nrmse']:12.3e} {r[worst]['max_abs']:10.3e} "
                  f"{min(r[f]['psnr_db'] for f in r):7.1f}  ({worst})")
        print(f"{name:48s} outside mismatches: "
              f"{report['comparisons'][name]['outside_value_mismatches']}")


if __name__ == "__main__":
    main()
