#!/usr/bin/env python3
"""
rdf_summary.py - read rdf.dat / rdf_check_rdf.dat files written by mace_md.py, print peak
positions, heights and coordination numbers, and overlay several files in one plot.

Usage
  python rdf_summary.py runs/bulkwater_300K/rdf.dat --box 12.0
  python rdf_summary.py mace=runs/bulkwater_300K/rdf.dat mpa=runs_mpa/bulkwater_300K/rdf.dat \
        aimd=aimd_rdf_rdf.dat --pairs Ow-Ow,Ow-Hw --out water_compare

Each argument is [label=]path. Columns are read from the header (r, g_A-B, n_A-B, ...).
--box L marks L/2 on the plots: beyond L/2 the RDF is dominated by periodic-image
(finite-size) effects and should not be interpreted.
"""
import argparse
import os

import numpy as np


def load(path):
    names = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                last = line[1:].strip()
                if last.startswith("r "):
                    names = last.split()
            else:
                break
    data = np.loadtxt(path)
    if names is None or len(names) != data.shape[1]:
        raise SystemExit(f"{path}: cannot parse column names from header")
    d = {n: data[:, k] for k, n in enumerate(names)}
    return d


def smooth(y, w=5):
    if w < 2:
        return y
    k = np.ones(w) / w
    return np.convolve(np.nan_to_num(y), k, mode="same")


def features(r, g, n, rstart=1.5):
    """first intermolecular peak (r >= rstart), following minimum, coordination number there."""
    out = {}
    gs = smooth(g)
    intra = r < 1.3
    if intra.any() and np.nanmax(g[intra]) > 5:
        k = np.nanargmax(g[intra])
        out["bond_peak_r"], out["bond_peak_g"] = r[intra][k], g[intra][k]
    idx = np.where(r >= rstart)[0]
    if len(idx) < 5:
        return out
    thr = 0.2 * np.nanmax(gs[idx])  # relative threshold: slab RDFs are diluted by vacuum, so g(r) << 1
    for i in idx[1:-1]:
        if gs[i] > gs[i - 1] and gs[i] >= gs[i + 1] and gs[i] > thr:
            out["peak1_r"], out["peak1_g"] = r[i], g[i]
            for j in range(i + 1, len(r) - 1):
                if gs[j] < gs[j - 1] and gs[j] <= gs[j + 1]:
                    out["min1_r"], out["min1_g"], out["n_at_min1"] = r[j], g[j], n[j]
                    break
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--pairs", default=None, help="comma list, e.g. Ow-Ow,Ow-Hw (default: all non-empty)")
    ap.add_argument("--box", type=float, default=None, help="smallest cell length (A); marks L/2")
    ap.add_argument("--rstart", type=float, default=1.5, help="ignore peaks below this r when finding peak 1")
    ap.add_argument("--rplot", type=float, default=8.0)
    ap.add_argument("--out", default="rdf_summary")
    args = ap.parse_args()

    sets = []
    for a in args.files:
        label, path = (a.split("=", 1) if "=" in a else (os.path.basename(os.path.dirname(os.path.abspath(a))) or a, a))
        sets.append((label, load(path)))

    allpairs = [k[2:] for k in sets[0][1] if k.startswith("g_")]
    pairs = args.pairs.split(",") if args.pairs else [
        p for p in allpairs if any(np.isfinite(d.get(f"g_{p}", np.array([np.nan]))).any() for _, d in sets)]

    if not pairs:
        raise SystemExit("No non-empty pairs found in these files (all columns are NaN). The atom groups "
                         "in the pair list do not exist in this system. Re-run 'mace_md.py rdf' with "
                         "--rdf-pairs, e.g. C-C for graphene-only systems.")
    print(f"{'pair':<8}{'set':<14}{'peak1 r':>9}{'g(peak1)':>10}{'min1 r':>9}{'g(min1)':>9}{'n(min1)':>9}   bond peak")
    for p in pairs:
        for label, d in sets:
            if f"g_{p}" not in d:
                continue
            f = features(d["r"], d[f"g_{p}"], d[f"n_{p}"], args.rstart)
            fmt = lambda k, w, nd: f"{f[k]:{w}.{nd}f}" if k in f else " " * (w - 1) + "-"
            bond = f"{f['bond_peak_r']:.2f} A (g={f['bond_peak_g']:.0f})" if "bond_peak_r" in f else ""
            print(f"{p:<8}{label:<14}{fmt('peak1_r', 9, 2)}{fmt('peak1_g', 10, 2)}{fmt('min1_r', 9, 2)}"
                  f"{fmt('min1_g', 9, 2)}{fmt('n_at_min1', 9, 2)}   {bond}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(2, len(pairs), figsize=(4.5 * len(pairs), 6), squeeze=False, sharex=True)
    for c, p in enumerate(pairs):
        for label, d in sets:
            if f"g_{p}" in d:
                ax[0, c].plot(d["r"], d[f"g_{p}"], label=label)
                ax[1, c].plot(d["r"], d[f"n_{p}"], label=label)
        ax[0, c].set(title=p, ylabel="g(r)")
        ax[1, c].set(xlabel="r (Å)", ylabel="n(r)", xlim=(0, args.rplot))
        if args.box:
            for a in (ax[0, c], ax[1, c]):
                a.axvline(args.box / 2, color="gray", ls=":", lw=1)
        ax[0, c].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150)
    print(f"wrote {args.out}.png")


if __name__ == "__main__":
    main()