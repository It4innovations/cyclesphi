"""Tightness of the density majorant of a .srnxr version 2 model on a voxel grid.

  python srnxr_majorant_check.py model.srnxr [--grid 24] [--samples 4] [--field rho] [--time 0.5]

Compares, as sums over the voxels of the domain (the ratio to the summed mean field is the factor
of majorant collisions to real collisions of a null-scattering integrator, for a transfer function
max(field, 0) * scale):
  sampled max      maximum of the decoded field over the samples of a voxel (oracle decoder)
  certified        what the Cycles bake uses: union of the per-leaf output ranges b +- sum |W| over
                   the leaves whose expanded box touches the voxel
  neighbor union   the earlier bake: union over the neighbors of every strict leaf near the voxel
"""

from __future__ import annotations

import argparse

import numpy as np

import srnxr_format as fmt
import srnxr_reference as ref


def boxes_touch(boxes, lo, hi):
    """[voxels, leaves]: the closed voxel box [lo, hi] intersects the leaf boxes (xyz)."""
    return (np.all(boxes[None, :, 0:3] <= hi[:, None, :], -1) &
            np.all(lo[:, None, :] <= boxes[None, :, 4:7], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--field", default="rho")
    ap.add_argument("--time", type=float, default=0.5)
    args = ap.parse_args()

    model = fmt.read(args.model)
    field = list(model.field_names).index(args.field)
    weights = ref.dequantized_weights(model)
    own = ref.certified_leaf_bounds(model, weights)[:, field, :]
    neighbor_union = np.array([[own[n, 0].min(), own[n, 1].max()]
                               for n in ref.leaf_neighbors(model)])

    g = args.grid
    dmin = model.domain_min[:3].astype(np.float64)
    dmax = model.domain_max[:3].astype(np.float64)
    h = (dmax - dmin) / g
    idx = np.stack(np.meshgrid(*[np.arange(g)] * 3, indexing="ij"), -1).reshape(-1, 3)
    vmin = dmin + idx * h
    vmax = vmin + h
    strict = model.leaf_strict_bounds.astype(np.float64)
    expanded = model.leaf_expanded_bounds.astype(np.float64)

    near = boxes_touch(strict, vmin - h, vmax + h)
    maj_union = np.where(near, neighbor_union[None, :, 1], -np.inf).max(1)
    touch = boxes_touch(expanded, vmin, vmax)
    maj_cert = np.where(touch, own[None, :, 1], -np.inf).max(1)

    rng = np.random.default_rng(1)
    s = args.samples
    pts = vmin[:, None, :] + rng.random((len(vmin), s, 3)) * h
    p4 = np.concatenate([pts.reshape(-1, 3), np.full((len(vmin) * s, 1), args.time)], 1)
    values, _ = ref.decode(model, p4.astype(np.float32), dtype=np.float32)
    f = np.maximum(values[:, field].reshape(len(vmin), s), 0.0)
    total = f.mean(1).sum()

    print(f"grid {g}^3, {s} samples per voxel, field {args.field}, t = {args.time}")
    for name, m in (("sampled max", f.max(1)), ("certified", np.maximum(maj_cert, 0)),
                    ("neighbor union", np.maximum(maj_union, 0))):
        print(f"  {name:15s} {m.sum() / total:6.2f} x the mean field")


if __name__ == "__main__":
    main()
