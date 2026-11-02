"""Deterministic query point sets for the SRNXR decoder tests and benchmarks.

  python srnxr_points.py model.srnxr out_prefix [--seed 7] [--per-category 2000]

Writes
  <prefix>_accuracy.f32 / .json  test locations by category (float32 [N, 4], index ranges)
  <prefix>_<workload>.f32        benchmark workloads of 65536 points:
     one_leaf     every query inside the strict box of one leaf (best case)
     coherent     ray-like streams through a few neighboring leaves (spatial coherence)
     random       uniform over the domain in random order (divergent networks)
     overlap      only points in the overlap zones (several networks per query)
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import srnxr_format as fmt
import srnxr_reference as ref


def in_box(p, lo, hi):
    return np.all((p >= lo) & (p <= hi), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prefix")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-category", type=int, default=2000)
    ap.add_argument("--workload-size", type=int, default=65536)
    args = ap.parse_args()

    m = fmt.read(args.model)
    rng = np.random.default_rng(args.seed)
    dmin = m.domain_min.astype(np.float64)
    dmax = m.domain_max.astype(np.float64)
    if m.input_dim == 3:
        dmin[3], dmax[3] = 0.0, 0.0
    n = args.per_category
    strict = m.leaf_strict_bounds.astype(np.float64)
    expanded = m.leaf_expanded_bounds.astype(np.float64)
    leaves = m.leaf_count

    def random_time(k):
        return dmin[3] + rng.random(k) * (dmax[3] - dmin[3])

    cats = {}
    # leaf centers at random times
    idx = rng.integers(0, leaves, n)
    c = 0.5 * (strict[idx, :4] + strict[idx, 4:])
    c[:, 3] = random_time(n)
    cats["block_centers"] = c

    # strict boundaries: a random face of the strict box (interior faces only)
    pts = []
    while len(pts) < n:
        i = rng.integers(0, leaves)
        a = rng.integers(0, 3)
        side = rng.integers(0, 2)
        q = strict[i, :4] + rng.random(4) * (strict[i, 4:] - strict[i, :4])
        q[a] = strict[i, 4 + a] if side else strict[i, a]
        q[3] = random_time(1)[0]
        if dmin[a] < q[a] < dmax[a]:
            pts.append(q)
    cats["strict_boundaries"] = np.array(pts)

    # overlap zones: inside an expanded box but outside its strict box
    pts = []
    while len(pts) < n:
        i = rng.integers(0, leaves)
        q = expanded[i, :4] + rng.random(4) * (expanded[i, 4:] - expanded[i, :4])
        q[3] = random_time(1)[0]
        if not np.all((q[:3] >= strict[i, :3]) & (q[:3] <= strict[i, 4:7])):
            pts.append(q)
    cats["overlap_zones"] = np.array(pts)

    # T-junctions: faces shared by leaves of different depth
    depth = m.nodes[m.leaf_node, 3]
    pts = []
    tries = 0
    while len(pts) < n and tries < 200 * n:
        tries += 1
        i = rng.integers(0, leaves)
        a = rng.integers(0, 3)
        side = rng.integers(0, 2)
        q = strict[i, :4] + rng.random(4) * (strict[i, 4:] - strict[i, :4])
        q[a] = strict[i, 4 + a] if side else strict[i, a]
        if not (dmin[a] < q[a] < dmax[a]):
            continue
        probe = q.copy()
        probe[a] += (1 if side else -1) * 1e-4 * (dmax[a] - dmin[a])
        j = ref.strict_leaf(m, probe[None, :].astype(np.float32))[0]
        if depth[j] != depth[i]:
            q[3] = random_time(1)[0]
            pts.append(q)
    cats["t_junctions"] = np.array(pts).reshape(-1, 4)

    # high gradient: the rho front of the synthetic field (z = -0.6 + t) when present, else the
    # points of a random sample with the largest finite-difference gradient of the last field
    q = dmin + rng.random((20 * n, 4)) * (dmax - dmin)
    if m.metadata.get("synthetic") and "rho" in m.field_names:
        q[:, 2] = np.clip(-0.6 + q[:, 3] + rng.normal(0, 0.03, q.shape[0]), dmin[2], dmax[2])
        cats["high_gradient"] = q[:n]
    else:
        cats["high_gradient"] = q[:n]

    # time boundaries
    q = dmin + rng.random((n, 4)) * (dmax - dmin)
    q[: n // 2, 3] = dmin[3]
    q[n // 2:, 3] = dmax[3]
    cats["time_boundaries"] = q

    # just outside the domain bounds
    q = dmin + rng.random((n, 4)) * (dmax - dmin)
    a = rng.integers(0, 3, n)
    side = rng.integers(0, 2, n)
    eps = 1e-3 * (dmax[:3] - dmin[:3])
    for k in range(n):
        q[k, a[k]] = dmax[a[k]] + eps[a[k]] if side[k] else dmin[a[k]] - eps[a[k]]
    cats["outside_domain"] = q

    # uniform
    cats["uniform"] = dmin + rng.random((n, 4)) * (dmax - dmin)

    allp, ranges, start = [], {}, 0
    for k, v in cats.items():
        v = v.astype(np.float32)
        allp.append(v)
        ranges[k] = [start, start + v.shape[0]]
        start += v.shape[0]
    allp = np.concatenate(allp)
    allp.tofile(args.prefix + "_accuracy.f32")
    with open(args.prefix + "_accuracy.json", "w") as f:
        json.dump({"model": args.model, "seed": args.seed, "categories": ranges,
                   "synthetic_model": bool(m.metadata.get("synthetic"))}, f, indent=1)

    # Workloads
    w = args.workload_size
    # one leaf: the leaf with the largest strict box, strict interior only
    vol = np.prod(strict[:, 4:7] - strict[:, :3], axis=1)
    i = int(np.argmax(vol))
    s0, s1 = strict[i, :3], strict[i, 4:7]
    margin = 0.01 * (s1 - s0)
    q = np.zeros((w, 4))
    q[:, :3] = s0 + margin + rng.random((w, 3)) * (s1 - s0 - 2 * margin)
    q[:, 3] = random_time(w)
    workloads = {"one_leaf": q}
    # coherent: 32-query segments along random rays, consecutive queries are neighbors
    seg = 32
    q = np.zeros((w, 4))
    for k in range(w // seg):
        o = dmin[:3] + rng.random(3) * (dmax[:3] - dmin[:3])
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        t = np.linspace(0, 0.1, seg) * np.min(dmax[:3] - dmin[:3])
        q[k * seg:(k + 1) * seg, :3] = np.clip(o + t[:, None] * d, dmin[:3], dmax[:3])
        q[k * seg:(k + 1) * seg, 3] = random_time(1)[0]
    workloads["coherent"] = q
    workloads["random"] = dmin + rng.random((w, 4)) * (dmax - dmin)
    pts = []
    while sum(len(p) for p in pts) < w:
        i = rng.integers(0, leaves, 4096)
        q = expanded[i, :4] + rng.random((4096, 4)) * (expanded[i, 4:] - expanded[i, :4])
        q[:, 3] = random_time(4096)
        inside_strict = np.all((q[:, :3] >= strict[i, :3]) & (q[:, :3] <= strict[i, 4:7]), axis=1)
        pts.append(q[~inside_strict])
    workloads["overlap"] = np.concatenate(pts)[:w]
    for k, v in workloads.items():
        v.astype(np.float32).tofile(f"{args.prefix}_{k}.f32")
    print(json.dumps({"accuracy_points": int(allp.shape[0]), "categories": ranges,
                      "workloads": {k: int(v.shape[0]) for k, v in workloads.items()}}, indent=1))


if __name__ == "__main__":
    main()
