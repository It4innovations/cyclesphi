"""Field-level accuracy analysis of SRNXR decoder outputs (slices, gradients, isosurface,
continuity, time consistency).

  python srnxr_analysis.py points model.srnxr prefix        write <prefix>_analysis.f32/.json
  python srnxr_analysis.py analyze model.srnxr prefix out_dir NAME=values.f32 ...

The analysis point set (deterministic):
  slice_t0/tm/tp   256 x 256 grid on z = SLICE_Z at t = 0.5 and t = 0.5 -+ dt
  jump_in/jump_out pairs straddling the support boundary of a leaf (expanded box face), at
                   distance eps inside and outside, and pairs straddling strict faces

Per backend it writes error maps (PNG) of rho, p and |u| on the slice and a JSON with
  gradient error      RMSE of the finite-difference gradient (x, y) relative to the oracle's
  isosurface          mean / P99 of |rho - rho_ref| / |grad rho_ref| near rho_ref = iso
  continuity          jump |u(out) - u(in)| across support / strict faces, oracle and backend
  temporal            RMSE of the central time difference relative to the oracle's
Every result is SYNTHETIC when the model metadata says so.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

import srnxr_format as fmt
import srnxr_reference as ref

N_SLICE = 256
DT = 1e-3
# z of the slice: below the rho front of the synthetic field at t = 0.5
SLICE_Z = -0.3


def make_points(model_path, prefix):
    m = fmt.read(model_path)
    rng = np.random.default_rng(11)
    dmin, dmax = m.domain_min.astype(np.float64), m.domain_max.astype(np.float64)
    xs = np.linspace(dmin[0], dmax[0], N_SLICE)
    ys = np.linspace(dmin[1], dmax[1], N_SLICE)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    tc = 0.5 * (dmin[3] + dmax[3]) if m.input_dim == 4 else 0.0
    sets = {}
    for name, t in (("slice_t0", tc), ("slice_tm", tc - DT), ("slice_tp", tc + DT)):
        sets[name] = np.stack(
            [gx.ravel(), gy.ravel(), np.full(gx.size, SLICE_Z), np.full(gx.size, t)], 1)

    # support boundary pairs: a face of an expanded box that is inside the domain
    eps = 1e-4 * float(np.min(dmax[:3] - dmin[:3]))
    pin, pout, sin_, sout = [], [], [], []
    e = m.leaf_expanded_bounds.astype(np.float64)
    s = m.leaf_strict_bounds.astype(np.float64)
    while len(pin) < 4000:
        i = rng.integers(0, m.leaf_count)
        a = rng.integers(0, 3)
        side = rng.integers(0, 2)
        face = e[i, 4 + a] if side else e[i, a]
        if not (dmin[a] + 2 * eps < face < dmax[a] - 2 * eps):
            continue
        q = e[i, :4] + rng.random(4) * (e[i, 4:] - e[i, :4])
        q[3] = dmin[3] + rng.random() * (dmax[3] - dmin[3])
        qi, qo = q.copy(), q.copy()
        qi[a] = face - eps if side else face + eps
        qo[a] = face + eps if side else face - eps
        pin.append(qi)
        pout.append(qo)
    while len(sin_) < 4000:
        i = rng.integers(0, m.leaf_count)
        a = rng.integers(0, 3)
        side = rng.integers(0, 2)
        face = s[i, 4 + a] if side else s[i, a]
        if not (dmin[a] + 2 * eps < face < dmax[a] - 2 * eps):
            continue
        q = s[i, :4] + rng.random(4) * (s[i, 4:] - s[i, :4])
        q[3] = dmin[3] + rng.random() * (dmax[3] - dmin[3])
        qi, qo = q.copy(), q.copy()
        qi[a] = face - eps if side else face + eps
        qo[a] = face + eps if side else face - eps
        sin_.append(qi)
        sout.append(qo)
    sets["support_in"] = np.array(pin)
    sets["support_out"] = np.array(pout)
    sets["strict_in"] = np.array(sin_)
    sets["strict_out"] = np.array(sout)

    allp, ranges, start = [], {}, 0
    for k, v in sets.items():
        allp.append(v.astype(np.float32))
        ranges[k] = [start, start + v.shape[0]]
        start += v.shape[0]
    np.concatenate(allp).tofile(prefix + "_analysis.f32")
    with open(prefix + "_analysis.json", "w") as f:
        json.dump({"model": model_path, "categories": ranges, "slice": N_SLICE, "dt": DT,
                   "eps": eps}, f, indent=1)
    print(prefix + "_analysis.f32", start, "points")


def analyze(model_path, prefix, out_dir, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = fmt.read(model_path)
    p = np.fromfile(prefix + "_analysis.f32", dtype=np.float32).reshape(-1, 4)
    meta = json.load(open(prefix + "_analysis.json"))
    cats = meta["categories"]
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "analysis_oracle.npz")
    if os.path.exists(cache):
        oracle = np.load(cache)["oracle"]
    else:
        oracle, _ = ref.decode(m, p)
        np.savez(cache, oracle=oracle)
    f_idx = {f: i for i, f in enumerate(m.field_names)}
    n = meta["slice"]
    dmin, dmax = m.domain_min, m.domain_max
    h = float(dmax[0] - dmin[0]) / (n - 1)

    def sl(v, name):
        a, b = cats[name]
        return v[a:b]

    def derived(v):
        out = {}
        if "rho" in f_idx:
            out["rho"] = v[:, f_idx["rho"]]
        if "p" in f_idx:
            out["p"] = v[:, f_idx["p"]]
        if all(k in f_idx for k in ("ux", "uy", "uz")):
            out["speed"] = np.linalg.norm(v[:, [f_idx["ux"], f_idx["uy"], f_idx["uz"]]], axis=1)
        if not out:
            out[m.field_names[0]] = v[:, 0]
        return out

    report = {"model": model_path, "synthetic_model": bool(m.metadata.get("synthetic")),
              "backends": {}}
    entries = [("oracle_fp16", oracle)] + [
        (it.split("=", 1)[0],
         np.fromfile(it.split("=", 1)[1], np.float32).reshape(p.shape[0], -1).astype(np.float64))
        for it in results]
    ref_slice = {k: v.reshape(n, n) for k, v in derived(sl(oracle, "slice_t0")).items()}
    for name, v in entries:
        r = {}
        test_slice = {k: x.reshape(n, n) for k, x in derived(sl(v, "slice_t0")).items()}
        # gradients
        g = {}
        for k in ref_slice:
            gy_r, gx_r = np.gradient(ref_slice[k], h)
            gy_t, gx_t = np.gradient(test_slice[k], h)
            num = np.sqrt(np.mean((gx_t - gx_r) ** 2 + (gy_t - gy_r) ** 2))
            den = np.sqrt(np.mean(gx_r ** 2 + gy_r ** 2))
            g[k] = {"gradient_rmse": float(num), "gradient_rel_rmse": float(num / den)}
        r["gradient"] = g
        # isosurface of rho at half its slice maximum
        if "rho" in ref_slice:
            iso = 0.5 * float(ref_slice["rho"].max())
            gy_r, gx_r = np.gradient(ref_slice["rho"], h)
            gm = np.sqrt(gx_r ** 2 + gy_r ** 2)
            band = (np.abs(ref_slice["rho"] - iso) < 0.05 * iso) & (gm > 1e-6)
            d = np.abs(test_slice["rho"] - ref_slice["rho"])[band] / gm[band]
            r["isosurface"] = {"iso": iso, "points": int(band.sum()),
                               "mean_distance": float(d.mean()) if d.size else None,
                               "p99_distance": float(np.percentile(d, 99)) if d.size else None,
                               "grid_spacing": h}
        # continuity across faces
        cont = {}
        for face in ("support", "strict"):
            jin, jout = derived(sl(v, face + "_in")), derived(sl(v, face + "_out"))
            cont[face] = {k: {"mean_jump": float(np.mean(np.abs(jout[k] - jin[k]))),
                              "max_jump": float(np.max(np.abs(jout[k] - jin[k])))} for k in jin}
        r["continuity"] = cont
        # time differences
        tm, tp = derived(sl(v, "slice_tm")), derived(sl(v, "slice_tp"))
        rtm, rtp = derived(sl(oracle, "slice_tm")), derived(sl(oracle, "slice_tp"))
        tmp = {}
        for k in tm:
            dt_t = (tp[k] - tm[k]) / (2 * meta["dt"])
            dt_r = (rtp[k] - rtm[k]) / (2 * meta["dt"])
            tmp[k] = {"dudt_rmse": float(np.sqrt(np.mean((dt_t - dt_r) ** 2))),
                      "dudt_rel_rmse": float(np.sqrt(np.mean((dt_t - dt_r) ** 2)) /
                                             max(np.sqrt(np.mean(dt_r ** 2)), 1e-30))}
        r["temporal"] = tmp
        report["backends"][name] = r

        # error maps
        if name != "oracle_fp16":
            fig, axes = plt.subplots(1, len(ref_slice), figsize=(4.2 * len(ref_slice), 3.6))
            axes = np.atleast_1d(axes)
            for ax, k in zip(axes, ref_slice):
                err = np.abs(test_slice[k] - ref_slice[k])
                im = ax.imshow(np.log10(err + 1e-12), origin="lower",
                               extent=[dmin[0], dmax[0], dmin[1], dmax[1]], cmap="magma",
                               vmin=-9, vmax=-2)
                ax.set_title(f"{name}: log10 |err| {k}")
                fig.colorbar(im, ax=ax, fraction=0.046)
            fig.suptitle("z = %.2f, t = %.3f%s" % (SLICE_Z, float(p[cats["slice_t0"][0], 3]),
                                                   " (SYNTHETIC model)"
                                                   if report["synthetic_model"] else ""))
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"error_map_{name}.png"), dpi=90)
            plt.close(fig)
    # the reference slice itself
    fig, axes = plt.subplots(1, len(ref_slice), figsize=(4.2 * len(ref_slice), 3.6))
    for ax, k in zip(np.atleast_1d(axes), ref_slice):
        im = ax.imshow(ref_slice[k], origin="lower", extent=[dmin[0], dmax[0], dmin[1], dmax[1]])
        ax.set_title(f"oracle {k}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "slice_oracle.png"), dpi=90)
    plt.close(fig)
    with open(os.path.join(out_dir, "analysis.json"), "w") as f:
        json.dump(report, f, indent=1)
    for name, r in report["backends"].items():
        g = r["gradient"].get("rho", next(iter(r["gradient"].values())))
        c = r["continuity"]["support"]
        k0 = next(iter(c))
        iso = r.get("isosurface", {}).get("mean_distance")
        print(f"{name:28s} grad rel {g['gradient_rel_rmse']:.2e}  iso mean dist "
              f"{iso if iso is None else f'{iso:.2e}'}  support jump {k0} "
              f"{c[k0]['mean_jump']:.2e} (max {c[k0]['max_jump']:.2e})")


if __name__ == "__main__":
    if sys.argv[1] == "points":
        make_points(sys.argv[2], sys.argv[3])
    else:
        analyze(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
