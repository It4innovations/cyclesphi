"""Check of the certified per-leaf output ranges used for the volume majorants.

  python srnxr_bounds_check.py model.srnxr [--samples 4000] [--json out.json]

For every leaf the range [b - sum|W|, b + sum|W|] of the linear head (|sin| <= 1), mapped by
the output transform, must contain the leaf network output anywhere in its support, and the
blended field in the strict box must lie in the union of the ranges of the neighbors. The
script samples both and reports violations (must be 0) and how loose the bound is
(certified width / sampled width).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import srnxr_format as fmt
import srnxr_reference as ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--json")
    args = ap.parse_args()
    m = fmt.read(args.model)
    rng = np.random.default_rng(5)
    w = ref.dequantized_weights(m)
    bounds = ref.certified_leaf_bounds(m, w)
    neighbors = ref.leaf_neighbors(m)
    viol_leaf = viol_blend = 0
    loose = []
    for leaf in range(m.leaf_count):
        e = m.leaf_expanded_bounds[leaf].astype(np.float64)
        q = e[:4] + rng.random((args.samples, 4)) * (e[4:] - e[:4])
        it = m.leaf_input_transform[leaf].astype(np.float64)
        x = (q[:, :m.input_dim] - it[:m.input_dim]) * it[4:4 + m.input_dim]
        y = ref.eval_network(m, ref.network_params(m, int(m.leaf_network[leaf]), w), x)
        t = m.leaf_output_transform[leaf].astype(np.float64)
        u = y * t[:m.output_dim] + t[m.output_dim:]
        lo, hi = bounds[leaf, :, 0], bounds[leaf, :, 1]
        viol_leaf += int(np.sum((u < lo) | (u > hi)))
        loose.append((hi - lo) / np.maximum(u.max(axis=0) - u.min(axis=0), 1e-12))
        # blended field inside the strict box against the union of the neighbor ranges
        s = m.leaf_strict_bounds[leaf].astype(np.float64)
        qs = s[:4] + rng.random((args.samples // 4, 4)) * (s[4:] - s[:4])
        v, _ = ref.decode(m, qs)
        nlo = bounds[neighbors[leaf], :, 0].min(axis=0)
        nhi = bounds[neighbors[leaf], :, 1].max(axis=0)
        viol_blend += int(np.sum((v < nlo) | (v > nhi)))
    loose = np.array(loose)
    report = {
        "model": args.model,
        "synthetic_model": bool(m.metadata.get("synthetic")),
        "leaf_output_violations": viol_leaf,
        "blended_field_violations": viol_blend,
        "looseness_median": dict(zip(m.field_names, np.median(loose, axis=0).tolist())),
        "looseness_max": dict(zip(m.field_names, loose.max(axis=0).tolist())),
    }
    print(json.dumps(report, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
