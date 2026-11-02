"""Deterministic SYNTHETIC `.srnxr` version 2 surrogate.

No trained checkpoints of the adaptive-octree CFD surrogate are available in
this repository, so the tests use a synthetic stand-in. It mimics the layout
of the method (adaptive octree, 15% expanded leaves, Gaussian blending, one
SIREN per leaf over (x, y, z, t), fields ux uy uz p rho) but its networks are
fitted to an analytic field (--fit torch: Adam on all leaves at once,
--fit lstsq: initialized sine layers and a least squares head, no torch
needed but a much worse fit). It says nothing about the fidelity of real
models, every test that uses it is a SYNTHETIC test.

Usage:
  python srnxr_make_synthetic.py out.srnxr [--width 256] [--layers 3] [--max-depth 3]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import srnxr_format as fmt
import srnxr_reference as ref

FIELD_NAMES = ["ux", "uy", "uz", "p", "rho"]
DOMAIN_MIN = np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32)
DOMAIN_MAX = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)


def analytic_field(p, xp=np):
    """Smooth time dependent flow with a steep front, fields ux uy uz p rho.

    xp is numpy or torch, p has shape [N, 4]."""
    x, y, z, t = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    xs = x - 0.3 * t
    pi = np.pi
    ux = xp.sin(pi * xs) * xp.cos(pi * y) * xp.cos(pi * z)
    uy = -xp.cos(pi * xs) * xp.sin(pi * y) * xp.cos(pi * z)
    uz = 0.3 * xp.sin(2 * pi * t) * xp.sin(pi * z)
    pr = 0.25 * (xp.cos(2 * pi * xs) + xp.cos(2 * pi * y)) * (xp.cos(2 * pi * z) + 2.0)
    rxy2 = (x - 0.25 * xp.sin(2 * pi * t)) ** 2 + (y - 0.25 * xp.cos(2 * pi * t)) ** 2
    front = 0.5 * (1.0 - xp.tanh((z - (-0.6 + 1.0 * t)) / 0.06))
    rho = 0.05 + 0.9 * xp.exp(-rxy2 / (2 * 0.18 ** 2)) * front * (
        1.0 + 0.25 * xp.sin(6 * x + 4 * y + 3 * z))
    return xp.stack([ux, uy, uz, pr, rho], 1)


def build_octree(max_depth: int, min_depth: int, threshold: float, rng):
    """Breadth-first adaptive octree, refine where rho varies by more than threshold."""
    nodes = []  # [first_child, leaf_index, parent, depth]
    bounds = []  # [min xyz, max xyz]
    nodes.append([fmt.INVALID_INDEX, fmt.INVALID_INDEX, fmt.INVALID_INDEX, 0])
    bounds.append(np.concatenate([DOMAIN_MIN[:3], DOMAIN_MAX[:3]]).astype(np.float32))
    queue = [0]
    probe = np.stack(np.meshgrid(*[np.linspace(0, 1, 7)] * 3, indexing="ij"), -1).reshape(-1, 3)
    while queue:
        n = queue.pop(0)
        depth = nodes[n][3]
        b = bounds[n]
        refine = depth < min_depth
        if not refine and depth < max_depth:
            pts = b[:3] + probe * (b[3:] - b[:3])
            vals = []
            for t in (0.0, 0.5, 1.0):
                q = np.concatenate([pts, np.full((pts.shape[0], 1), t)], axis=1)
                vals.append(analytic_field(q)[:, 4])
            vals = np.concatenate(vals)
            refine = (vals.max() - vals.min()) > threshold
        if not refine:
            continue
        center = (np.float32(0.5) * (b[:3] + b[3:])).astype(np.float32)
        nodes[n][0] = len(nodes)
        for k in range(8):
            lo = np.where([(k >> a) & 1 for a in range(3)], center, b[:3]).astype(np.float32)
            hi = np.where([(k >> a) & 1 for a in range(3)], b[3:], center).astype(np.float32)
            nodes.append([fmt.INVALID_INDEX, fmt.INVALID_INDEX, n, depth + 1])
            bounds.append(np.concatenate([lo, hi]))
            queue.append(len(nodes) - 1)
    leaf_node = []
    for i, nd in enumerate(nodes):
        if nd[0] == fmt.INVALID_INDEX:
            nd[1] = len(leaf_node)
            leaf_node.append(i)
    return (np.array(nodes, dtype=np.uint32), np.array(bounds, dtype=np.float32),
            np.array(leaf_node, dtype=np.uint32))


def siren_hidden_init(rng, width, layers, in_dim, w0):
    """SIREN initialization (Sitzmann et al. 2020), biases as torch.nn.Linear."""
    params = []
    fan_in = in_dim
    for l in range(layers):
        if l == 0:
            bound = 1.0 / fan_in
        else:
            bound = np.sqrt(6.0 / fan_in) / w0[l]
        w = rng.uniform(-bound, bound, size=(width, fan_in))
        b = rng.uniform(-1.0 / np.sqrt(fan_in), 1.0 / np.sqrt(fan_in), size=width)
        params.append((w, b))
        fan_in = width
    return params


def fit_lstsq(args, params, expanded, input_tf, out_tf, w0, rng):
    """Keep the initialized sine layers, ridge least squares for the linear head."""
    out_dim = out_tf.shape[1] // 2
    heads, nrmse = [], []
    for leaf, hidden in enumerate(params):
        e = expanded[leaf]
        q = e[:4] + rng.random((args.samples, 4)) * (e[4:] - e[:4])
        yn = (analytic_field(q) - out_tf[leaf, out_dim:]) / out_tf[leaf, :out_dim]
        h = (q - input_tf[leaf, :4]) * input_tf[leaf, 4:]
        for l, (w, bb) in enumerate(hidden):
            h = np.sin(w0[l] * (h @ w.astype(np.float32).T + bb.astype(np.float32)))
        a = np.concatenate([h, np.ones((h.shape[0], 1))], axis=1)
        reg = args.ridge * a.shape[0] * np.eye(a.shape[1])
        reg[-1, -1] = 0.0
        sol = np.linalg.solve(a.T @ a + reg, a.T @ yn)
        heads.append((sol[:-1].T, sol[-1]))
        nrmse.append(np.sqrt(np.mean((a @ sol - yn) ** 2, axis=0)))
    return params, heads, np.array(nrmse)


def fit_torch(args, params, head, expanded, input_tf, out_tf, w0):
    """Adam on all leaves at once (batched matmuls), fresh uniform samples every step."""
    import torch

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    leaves = len(params)
    out_dim = out_tf.shape[1] // 2
    layers = len(params[0])

    def stack(i, j):
        return torch.tensor(np.stack([p[i][j] for p in params]), dtype=torch.float32, device=dev)

    ws = [stack(l, 0).requires_grad_() for l in range(layers)]
    bs = [stack(l, 1).requires_grad_() for l in range(layers)]
    hw = torch.tensor(np.stack([h[0] for h in head]), dtype=torch.float32,
                      device=dev).requires_grad_()
    hb = torch.tensor(np.stack([h[1] for h in head]), dtype=torch.float32,
                      device=dev).requires_grad_()
    emin = torch.tensor(expanded[:, :4], device=dev)
    esize = torch.tensor(expanded[:, 4:] - expanded[:, :4], device=dev)
    ioff = torch.tensor(input_tf[:, :4], device=dev)
    iscale = torch.tensor(input_tf[:, 4:], device=dev)
    oscale = torch.tensor(out_tf[:, :out_dim], device=dev)
    ooff = torch.tensor(out_tf[:, out_dim:], device=dev)
    w0t = [float(v) for v in w0]

    def forward(x):
        h = x
        for l in range(layers):
            h = torch.sin(w0t[l] * (torch.bmm(h, ws[l].transpose(1, 2)) + bs[l][:, None, :]))
        return torch.bmm(h, hw.transpose(1, 2)) + hb[:, None, :]

    def batch(n):
        q = emin[:, None, :] + torch.rand(leaves, n, 4, device=dev) * esize[:, None, :]
        y = analytic_field(q.reshape(-1, 4), torch).reshape(leaves, n, out_dim)
        x = (q - ioff[:, None, :]) * iscale[:, None, :]
        return x, (y - ooff[:, None, :]) / oscale[:, None, :]

    opt = torch.optim.Adam(ws + bs + [hw, hb], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    for step in range(args.steps):
        x, y = batch(args.batch)
        loss = torch.mean((forward(x) - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step} loss {loss.item():.3e}", flush=True)
    with torch.no_grad():
        x, y = batch(4096)
        nrmse = torch.sqrt(torch.mean((forward(x) - y) ** 2, dim=1)).cpu().numpy()
    out_params = [[(ws[l][i].detach().cpu().numpy(), bs[l][i].detach().cpu().numpy())
                   for l in range(layers)] for i in range(leaves)]
    out_head = [(hw[i].detach().cpu().numpy(), hb[i].detach().cpu().numpy())
                for i in range(leaves)]
    return out_params, out_head, nrmse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("output")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3, help="number of sine layers")
    ap.add_argument("--w0", type=float, nargs="+", default=[30.0])
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--min-depth", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--overlap", type=float, default=0.15)
    ap.add_argument("--sigma", type=float, default=0.4)
    ap.add_argument("--samples", type=int, default=6000, help="fit samples per leaf")
    ap.add_argument("--fit", choices=["torch", "lstsq"], default="torch")
    ap.add_argument("--steps", type=int, default=3000, help="torch: Adam steps")
    ap.add_argument("--batch", type=int, default=1024, help="torch: samples per leaf and step")
    ap.add_argument("--lr", type=float, default=1e-4, help="torch: learning rate")
    ap.add_argument("--ridge", type=float, default=1e-4, help="lstsq: ridge factor")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-master", action="store_true", help="do not store FP32 master weights")
    args = ap.parse_args()

    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    w0 = np.array(args.w0 * args.layers if len(args.w0) == 1 else args.w0, dtype=np.float32)
    assert w0.size == args.layers, "--w0 needs one value or one per sine layer"

    nodes, node_bounds, leaf_node = build_octree(args.max_depth, args.min_depth, args.threshold, rng)
    leaf_count = leaf_node.size
    in_dim, out_dim, width, layers = 4, len(FIELD_NAMES), args.width, args.layers

    strict = np.zeros((leaf_count, 8), np.float32)
    expanded = np.zeros((leaf_count, 8), np.float32)
    input_tf = np.zeros((leaf_count, 8), np.float32)
    blend = np.zeros((leaf_count, 9), np.float32)
    out_tf = np.zeros((leaf_count, 2 * out_dim), np.float32)
    layout = np.zeros((leaf_count, 4), np.uint32)

    for leaf in range(leaf_count):
        b = node_bounds[leaf_node[leaf]]
        smin = np.concatenate([b[:3], DOMAIN_MIN[3:]])
        smax = np.concatenate([b[3:], DOMAIN_MAX[3:]])
        size = smax - smin
        emin, emax = smin.copy(), smax.copy()
        emin[:3] = np.maximum(smin[:3] - args.overlap * size[:3], DOMAIN_MIN[:3])
        emax[:3] = np.minimum(smax[:3] + args.overlap * size[:3], DOMAIN_MAX[:3])
        strict[leaf] = np.concatenate([smin, smax])
        expanded[leaf] = np.concatenate([emin, emax])
        # local inputs: expanded box -> [-1, 1]^4
        input_tf[leaf, :4] = 0.5 * (emin + emax)
        input_tf[leaf, 4:] = 2.0 / (emax - emin)
        # blending distance in strict half extents, time does not take part
        blend[leaf, :4] = 0.5 * (smin + smax)
        blend[leaf, 4:7] = 2.0 / size[:3]
        blend[leaf, 7] = 0.0
        blend[leaf, 8] = args.sigma
        # per-leaf output normalization from the field statistics in the expanded box
        q = emin + rng.random((args.samples, 4)) * (emax - emin)
        target = analytic_field(q)
        out_tf[leaf, :out_dim] = np.maximum(target.std(axis=0), 1e-3)
        out_tf[leaf, out_dim:] = target.mean(axis=0)

    # SIREN initialization of every leaf, [leaf][layer] -> (W [out, in], b [out])
    params = [siren_hidden_init(rng, width, layers, in_dim, w0) for _ in range(leaf_count)]
    head_bound = np.sqrt(6 / width) / w0[-1]
    head = [(rng.uniform(-head_bound, head_bound, size=(out_dim, width)), np.zeros(out_dim))
            for _ in range(leaf_count)]
    if args.fit == "torch":
        params, head, fit_nrmse = fit_torch(args, params, head, expanded, input_tf, out_tf, w0)
    else:
        params, head, fit_nrmse = fit_lstsq(args, params, expanded, input_tf, out_tf, w0, rng)

    weights16, weights32, biases = [], [], []
    w_off = b_off = 0
    for leaf in range(leaf_count):
        # FP32 master: the trained values, FP16 storage: round to nearest even
        flat32 = np.concatenate([w.ravel() for w, _ in params[leaf]] +
                                [head[leaf][0].ravel()]).astype(np.float32)
        weights32.append(flat32)
        weights16.append(flat32.astype(np.float16))
        biases.append(np.concatenate([bb for _, bb in params[leaf]] +
                                     [head[leaf][1]]).astype(np.float32))
        layout[leaf] = (width, w_off, b_off, 0)
        w_off += flat32.size
        b_off += biases[-1].size

    model = fmt.SrnxrModel(
        input_dim=in_dim, output_dim=out_dim, hidden_layer_count=layers, hidden_width=width,
        field_names=FIELD_NAMES, layer_w0=w0, domain_min=DOMAIN_MIN, domain_max=DOMAIN_MAX,
        nodes=nodes, node_bounds=node_bounds, leaf_node=leaf_node,
        leaf_strict_bounds=strict, leaf_expanded_bounds=expanded, leaf_input_transform=input_tf,
        leaf_blend=blend, leaf_network=np.arange(leaf_count, dtype=np.uint32),
        leaf_output_transform=out_tf, network_layout=layout,
        weights=np.concatenate(weights16), biases=np.concatenate(biases),
        weights_fp32=None if args.no_master else np.concatenate(weights32),
        metadata={
            "synthetic": True,
            "generator": "tools/srnxr/srnxr_make_synthetic.py",
            "args": vars(args),
            "fit": ("Adam on uniform samples of the analytic field (torch, not bitwise "
                    "reproducible)" if args.fit == "torch" else
                    "SIREN-initialized sine layers, ridge least squares linear head"),
            "field_definition": "analytic_field() in srnxr_make_synthetic.py",
            "time_range": [float(DOMAIN_MIN[3]), float(DOMAIN_MAX[3])],
            "overlap_convention": "strict box extended by overlap * size on both sides of every "
                                  "spatial axis, clipped to the domain; time is not expanded",
            "blend_convention": "r = |(p - strict center) / strict half extent| over xyz, "
                                "w = exp(-r^2 / (2 sigma^2)), support = expanded box",
        },
    )
    fmt.write(args.output, model)

    fit_nrmse = np.array(fit_nrmse)
    # held-out reconstruction error of the blended field against the analytic field
    hold = DOMAIN_MIN + rng.random((20000, 4)) * (DOMAIN_MAX - DOMAIN_MIN)
    m2 = fmt.read(args.output)
    rec, inside = ref.decode(m2, hold)
    truth = analytic_field(hold.astype(np.float64))
    rng_field = truth.max(axis=0) - truth.min(axis=0)
    nrmse = np.sqrt(np.mean((rec - truth) ** 2, axis=0)) / rng_field
    storage = None
    if m2.weights_fp32 is not None:
        rec32, _ = ref.decode(m2, hold, use_master=True)
        storage = dict(zip(FIELD_NAMES, (np.sqrt(np.mean((rec - rec32) ** 2, axis=0)) /
                                         rng_field).tolist()))
    report = {
        "heldout_nrmse_fp16_vs_fp32_master": storage,
        "file": args.output,
        "leaves": int(leaf_count),
        "nodes": int(nodes.shape[0]),
        "max_depth": int(nodes[:, 3].max()),
        "weights_bytes_fp16": int(sum(w.nbytes for w in weights16)),
        "fit_nrmse_normalized_mean": fit_nrmse.mean(axis=0).tolist(),
        "heldout_nrmse_vs_analytic": dict(zip(FIELD_NAMES, nrmse.tolist())),
        "seconds": time.time() - t_start,
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
