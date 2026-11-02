"""Reference (oracle) decoder of `.srnxr` version 2 models.

Implements the decoding contract of tools/srnxr/neural_volume.md with numpy:

  p = (x, y, z, t)                 query in object space and simulation time
  outside the domain box           -> every field is 0, `inside` is False
  L = strict leaf of p             octree descent, a coordinate equal to the node
                                   center goes to the upper child
  for every leaf j in N(L)         N(L): leaves whose expanded box intersects the
                                   strict box of L (closed intervals)
    if p in expanded box of j      (closed)
      x_j = (p - offset_j) * scale_j               first input_dim components
      h   = x_j
      h   = sin(w0_l * (W_l h + b_l))              l = 0 .. hidden_layer_count - 1
      y   = W_head h + b_head                       linear head
      u_j = y * out_scale_j + out_offset_j
      e_j = -0.5 * sum_d ((p_d - c_jd) * s_jd)^2 / sigma_j^2
  u(p) = sum_j exp(e_j - m) u_j / sum_j exp(e_j - m),  m = max_j e_j

The max shift makes the normalization exact and keeps the weight sum >= 1,
so there is no zero weight sum case for points inside the domain (the strict
leaf always contributes).
"""

from __future__ import annotations

import numpy as np

import srnxr_format as fmt


def dequantized_weights(model: fmt.SrnxrModel, use_master: bool = False) -> np.ndarray:
    if use_master:
        if model.weights_fp32 is None:
            raise ValueError("model has no FP32 master weights")
        return model.weights_fp32.astype(np.float64)
    return model.weights.astype(np.float64)


def network_params(model: fmt.SrnxrModel, network: int, weights: np.ndarray):
    """Per-layer (W, b) of one network, W has shape [out, in]."""
    width, w_off, b_off, _ = (int(v) for v in model.network_layout[network])
    layers = []
    in_dim = model.input_dim
    for layer in range(model.hidden_layer_count):
        n = width * in_dim
        layers.append((weights[w_off:w_off + n].reshape(width, in_dim),
                       model.biases[b_off:b_off + width].astype(np.float64)))
        w_off += n
        b_off += width
        in_dim = width
    n = model.output_dim * width
    layers.append((weights[w_off:w_off + n].reshape(model.output_dim, width),
                   model.biases[b_off:b_off + model.output_dim].astype(np.float64)))
    return layers


def eval_network(model, layers, x, dtype=np.float64, act_fp16=False, fast_sin=False,
                 return_layers=False):
    """Evaluate one SIREN on local inputs x [N, input_dim].

    dtype:    arithmetic precision (float64 oracle or float32 like the GPU SIMT path)
    act_fp16: round every hidden activation to FP16 before the next layer, which is
              what the Tensor Core and cooperative vector paths feed to the matrix units
    """
    h = x.astype(dtype)
    trace = []
    for l, (w, b) in enumerate(layers[:-1]):
        z = h @ w.astype(dtype).T + b.astype(dtype)
        h = np.sin(dtype(model.layer_w0[l]) * z).astype(dtype)
        if act_fp16:
            h = h.astype(np.float16).astype(dtype)
        trace.append(h)
    w, b = layers[-1]
    y = h @ w.astype(dtype).T + b.astype(dtype)
    if return_layers:
        return y, trace
    return y


def strict_leaf(model: fmt.SrnxrModel, p: np.ndarray) -> np.ndarray:
    node = np.zeros(p.shape[0], dtype=np.int64)
    for _ in range(64):
        first_child = model.nodes[node, 0].astype(np.int64)
        internal = first_child != fmt.INVALID_INDEX
        if not internal.any():
            break
        nb = model.node_bounds[node].astype(np.float32)
        center = (np.float32(0.5) * (nb[:, 0:3] + nb[:, 3:6])).astype(np.float32)
        k = ((p[:, 0] >= center[:, 0]).astype(np.int64) |
             ((p[:, 1] >= center[:, 1]).astype(np.int64) << 1) |
             ((p[:, 2] >= center[:, 2]).astype(np.int64) << 2))
        node = np.where(internal, first_child + k, node)
    return model.nodes[node, 1].astype(np.int64)


def boxes_intersect(a: np.ndarray, b: np.ndarray, dims: int) -> bool:
    return bool(np.all(a[0:dims] <= b[4:4 + dims]) and np.all(b[0:dims] <= a[4:4 + dims]))


def leaf_neighbors(model: fmt.SrnxrModel) -> list[list[int]]:
    dims = 4 if model.input_dim == 4 else 3
    exp_ = model.leaf_expanded_bounds
    strict = model.leaf_strict_bounds
    result = []
    for i in range(model.leaf_count):
        lo_ok = np.all(exp_[:, 0:dims] <= strict[i, 4:4 + dims], axis=1)
        hi_ok = np.all(strict[i, 0:dims] <= exp_[:, 4:4 + dims], axis=1)
        result.append([int(j) for j in np.nonzero(lo_ok & hi_ok)[0]])
    return result


def certified_leaf_bounds(model: fmt.SrnxrModel, weights: np.ndarray) -> np.ndarray:
    """[leaf_count, output_dim, 2] interval of each leaf's denormalized output.

    |sin| <= 1, so y_k is within b_k +- sum_j |W_kj| of the linear head for any
    input. The bound uses the deployed weights and is independent of position.
    """
    out = np.zeros((model.leaf_count, model.output_dim, 2))
    for leaf in range(model.leaf_count):
        layers = network_params(model, int(model.leaf_network[leaf]), weights)
        w, b = layers[-1]
        radius = np.abs(w).sum(axis=1)
        lo, hi = b - radius, b + radius
        t = model.leaf_output_transform[leaf].astype(np.float64)
        scale, offset = t[:model.output_dim], t[model.output_dim:]
        a, c = lo * scale + offset, hi * scale + offset
        out[leaf, :, 0] = np.minimum(a, c)
        out[leaf, :, 1] = np.maximum(a, c)
    return out


def decode(model: fmt.SrnxrModel, p: np.ndarray, *, use_master=False, dtype=np.float64,
           act_fp16=False, return_stats=False):
    """Decode points p [N, 4] (t is ignored for 3D models). Returns [N, output_dim]."""
    p = np.asarray(p, dtype=np.float32)
    n = p.shape[0]
    dims = 4 if model.input_dim == 4 else 3
    weights = dequantized_weights(model, use_master)

    inside = np.all((p[:, :dims] >= model.domain_min[:dims]) &
                    (p[:, :dims] <= model.domain_max[:dims]), axis=1)
    result = np.zeros((n, model.output_dim), dtype=np.float64)
    contributions = np.zeros(n, dtype=np.int64)
    if not inside.any():
        return (result, inside, contributions) if return_stats else (result, inside)

    idx_all = np.nonzero(inside)[0]
    pi = p[idx_all]
    rows, exps, values = [], [], []
    for leaf in range(model.leaf_count):
        eb = model.leaf_expanded_bounds[leaf]
        mask = np.all((pi[:, :dims] >= eb[:dims]) & (pi[:, :dims] <= eb[4:4 + dims]), axis=1)
        if not mask.any():
            continue
        q = pi[mask].astype(np.float64)
        it = model.leaf_input_transform[leaf].astype(np.float64)
        x = (q[:, :model.input_dim] - it[:model.input_dim]) * it[4:4 + model.input_dim]
        layers = network_params(model, int(model.leaf_network[leaf]), weights)
        y = eval_network(model, layers, x, dtype=dtype, act_fp16=act_fp16).astype(np.float64)
        t = model.leaf_output_transform[leaf].astype(np.float64)
        u = y * t[:model.output_dim] + t[model.output_dim:]
        bl = model.leaf_blend[leaf].astype(np.float64)
        d = (q[:, :4] - bl[0:4]) * bl[4:8]
        e = -0.5 * np.sum(d * d, axis=1) / (bl[8] * bl[8])
        rows.append(np.nonzero(mask)[0])
        exps.append(e)
        values.append(u)

    rows = np.concatenate(rows)
    exps = np.concatenate(exps)
    values = np.concatenate(values)
    m = np.full(pi.shape[0], -np.inf)
    np.maximum.at(m, rows, exps)
    w = np.exp(exps - m[rows])
    num = np.zeros((pi.shape[0], model.output_dim))
    den = np.zeros(pi.shape[0])
    np.add.at(num, rows, w[:, None] * values)
    np.add.at(den, rows, w)
    cnt = np.bincount(rows, minlength=pi.shape[0])
    if np.any(cnt == 0):
        raise RuntimeError("point inside the domain without a contributing leaf: "
                           "the expanded boxes do not cover the domain")
    result[idx_all] = num / den[:, None]
    contributions[idx_all] = cnt
    if return_stats:
        return result, inside, contributions
    return result, inside
