#!/usr/bin/env python3
"""
rdf_summary.py - read rdf.dat / rdf_check_rdf.dat files written by mace_md.py, print peak
positions, heights and coordination numbers (with block-average uncertainties when the files
have them), compare sets, and overlay several files in one plot.

Usage
  python rdf_summary.py runs/bulkwater_300K/rdf.dat --box 12.0
  python rdf_summary.py mace=runs/bulkwater_300K/rdf.dat mpa=runs_mpa/bulkwater_300K/rdf.dat \
        aimd=aimd_rdf_rdf.dat --pairs Ow-Ow,Ow-Hw --out water_compare
  # interfacial water (pairs written by mace_md.py with --rdf-pairs "...,Ow-Ow@near=Pt:3.5")
  python rdf_summary.py mace=run/rdf.dat aimd=aimd_rdf.dat --pairs "Ow-Ow@near=Pt:3.5" \
        --zgroups Ow,Pt,C --out interface_compare

Each argument is [label=]path. Columns are read from the header (r, g_A-B, n_A-B and, if
present, gerr_A-B / nerr_A-B = standard error over blocks).
--box L marks L/2 on the plots: beyond L/2 the RDF is dominated by periodic-image
(finite-size) effects and should not be interpreted.

Comparison: the FIRST file is the reference. For every other set the table gives
z(g) and z(n) = |x - x_ref| / sqrt(err^2 + err_ref^2) for g(peak1) and n(min1). |z| < ~2 means
the difference is within the block-averaged noise; it says nothing about systematic errors
and block errors are underestimated when blocks are shorter than the correlation time.

Peak finding is a screening tool, not a metric: curves are smoothed over --smooth A, a peak
must stand out from the next minimum by --min-prom (fraction of the maximum), and the
peak/min positions are only accurate to about one smoothing width. For slab / cluster systems
g(r) is normalised with the cell volume (including vacuum), so heights are not bulk-like -
compare sets run in the same cell. n(r) is unambiguous.

--zgroups Ow,Pt,C additionally overlays the z-density profiles (zprofile.dat next to each
rdf file: rdf.dat -> zprofile.dat, X_rdf.dat -> X_zprofile.dat) and writes <out>_zprofile.png.
"""
import argparse
import os

import numpy as np


def load(path, first_col="r"):
    names = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                last = line[1:].strip()
                if last.startswith(first_col + " "):
                    names = last.split()
            else:
                break
    data = np.loadtxt(path)
    if names is None or len(names) != data.shape[1]:
        raise SystemExit(f"{path}: cannot parse column names from header")
    return {n: data[:, k] for k, n in enumerate(names)}


def zprofile_path(rdf_path):
    d, b = os.path.split(rdf_path)
    if b.endswith("_rdf.dat"):
        return os.path.join(d, b[:-len("_rdf.dat")] + "_zprofile.dat")
    return os.path.join(d, "zprofile.dat")


def fill_nan(y):
    """linear interpolation over NaNs (edge NaNs take the nearest valid value)"""
    y = np.asarray(y, float)
    bad = ~np.isfinite(y)
    if bad.all():
        return np.zeros_like(y)
    if bad.any():
        x = np.arange(len(y))
        y = y.copy()
        y[bad] = np.interp(x[bad], x[~bad], y[~bad])
    return y


def smooth(y, w=5):
    """moving average, edge-padded (no zero-padding artefacts at r=0 or r=rmax)"""
    y = fill_nan(y)
    w = int(w)
    if w < 2:
        return y
    if w % 2 == 0:
        w += 1
    h = w // 2
    yp = np.pad(y, h, mode="edge")
    return np.convolve(yp, np.ones(w) / w, mode="valid")


def features(r, g, n, rstart=1.5, win=5, min_prom=0.05, gerr=None, nerr=None):
    """first intermolecular peak (r >= rstart) with prominence, following minimum, coordination
    number there; uncertainties from block errors where available."""
    out = {}
    gs = smooth(g, win)
    gfin = np.where(np.isfinite(g), g, 0.0)
    intra = r < 1.3
    if intra.any() and np.nanmax(gfin[intra]) > 5:
        k = np.nanargmax(gfin[intra])
        out["bond_peak_r"], out["bond_peak_g"] = r[intra][k], g[intra][k]
    idx = np.where(r >= rstart)[0]
    if len(idx) < 5 or not np.isfinite(g[idx]).any():
        return out
    gmax = np.nanmax(gs[idx])
    thr = 0.2 * gmax  # relative threshold: slab RDFs are diluted by vacuum, so g(r) << 1
    for i in idx[1:-1]:
        if not (gs[i] > gs[i - 1] and gs[i] >= gs[i + 1] and gs[i] > thr):
            continue
        j = next((j for j in range(i + 1, len(r) - 1) if gs[j] < gs[j - 1] and gs[j] <= gs[j + 1]), None)
        if j is None:
            break
        if gs[i] - gs[j] < min_prom * gmax:   # shoulder / noise wiggle, not a real peak
            continue
        out["peak1_r"], out["peak1_g"] = r[i], g[i]
        out["min1_r"], out["min1_g"], out["n_at_min1"] = r[j], g[j], n[j]
        out["prom"] = float(gs[i] - gs[j])
        if gerr is not None and np.isfinite(gerr[i]):
            out["peak1_g_err"] = float(gerr[i])
        if nerr is not None and np.isfinite(nerr[j]):
            out["n_at_min1_err"] = float(nerr[j])
        break
    return out


def zscore(a, ea, b, eb):
    if a is None or b is None or ea is None or eb is None:
        return None
    s = np.hypot(ea, eb)
    return None if s <= 0 else abs(a - b) / s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--pairs", default=None, help="comma list, e.g. Ow-Ow,Ow-Hw (default: all non-empty)")
    ap.add_argument("--box", type=float, default=None, help="smallest cell length (A); marks L/2")
    ap.add_argument("--rstart", type=float, default=1.5, help="ignore peaks below this r when finding peak 1")
    ap.add_argument("--rplot", type=float, default=8.0)
    ap.add_argument("--smooth", type=float, default=0.1, help="smoothing width for peak finding (A)")
    ap.add_argument("--min-prom", type=float, default=0.05,
                    help="min peak-to-following-minimum height, fraction of max g (default 0.05)")
    ap.add_argument("--zgroups", default=None, help="comma list of groups for z-density overlay, e.g. Ow,Pt,C")
    ap.add_argument("--out", default="rdf_summary")
    args = ap.parse_args()

    sets, paths = [], []
    for a in args.files:
        label, path = (a.split("=", 1) if "=" in a else (os.path.basename(os.path.dirname(os.path.abspath(a))) or a, a))
        sets.append((label, load(path)))
        paths.append(path)

    allpairs = [k[2:] for k in sets[0][1] if k.startswith("g_")]
    pairs = args.pairs.split(",") if args.pairs else [
        p for p in allpairs if any(np.isfinite(d.get(f"g_{p}", np.array([np.nan]))).any() for _, d in sets)]

    if not pairs:
        raise SystemExit("No non-empty pairs found in these files (all columns are NaN). The atom groups "
                         "in the pair list do not exist in this system. Re-run 'mace_md.py rdf' with "
                         "--rdf-pairs, e.g. C-C for graphene-only systems.")

    def pm(f, k, ke, nd):
        return f"{f[k]:.{nd}f}" + (f"±{f[ke]:.{nd}f}" if ke in f else "") if k in f else "-"

    def zs(v):
        return "-" if v is None else f"{v:.1f}"

    print(f"{'pair':<22}{'set':<12}{'peak1 r':>8}{'g(peak1)':>13}{'min1 r':>8}{'g(min1)':>8}{'n(min1)':>13}"
          f"{'z(g)':>6}{'z(n)':>6}   bond peak")
    for p in pairs:
        ref = None
        for label, d in sets:
            if f"g_{p}" not in d:
                continue
            dr = float(np.nanmedian(np.diff(d["r"])))
            win = max(1, int(round(args.smooth / dr)))
            f = features(d["r"], d[f"g_{p}"], d[f"n_{p}"], args.rstart, win, args.min_prom,
                         d.get(f"gerr_{p}"), d.get(f"nerr_{p}"))
            if ref is None:
                ref, zg, zn = f, None, None
            else:
                zg = zscore(f.get("peak1_g"), f.get("peak1_g_err"), ref.get("peak1_g"), ref.get("peak1_g_err"))
                zn = zscore(f.get("n_at_min1"), f.get("n_at_min1_err"), ref.get("n_at_min1"), ref.get("n_at_min1_err"))
            bond = f"{f['bond_peak_r']:.2f} A (g={f['bond_peak_g']:.0f})" if "bond_peak_r" in f else ""
            r1 = f"{f['peak1_r']:.2f}" if "peak1_r" in f else "-"
            m1 = f"{f['min1_r']:.2f}" if "min1_r" in f else "-"
            gm = f"{f['min1_g']:.2f}" if "min1_g" in f else "-"
            print(f"{p:<22}{label:<12}{r1:>8}{pm(f, 'peak1_g', 'peak1_g_err', 2):>13}{m1:>8}{gm:>8}"
                  f"{pm(f, 'n_at_min1', 'n_at_min1_err', 2):>13}{zs(zg):>6}{zs(zn):>6}   {bond}")
    if not any(f"gerr_{p}" in d for p in pairs for _, d in sets):
        print("note: no gerr_/nerr_ columns in these files (written by an older mace_md.py?) - no uncertainties.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(2, len(pairs), figsize=(4.5 * len(pairs), 6), squeeze=False, sharex=True)
    for c, p in enumerate(pairs):
        for label, d in sets:
            if f"g_{p}" not in d:
                continue
            ln, = ax[0, c].plot(d["r"], d[f"g_{p}"], label=label)
            ax[1, c].plot(d["r"], d[f"n_{p}"], label=label, color=ln.get_color())
            for row, key, err in ((0, f"g_{p}", f"gerr_{p}"), (1, f"n_{p}", f"nerr_{p}")):
                if err in d:
                    y, e = d[key], np.nan_to_num(d[err])
                    ax[row, c].fill_between(d["r"], y - e, y + e, color=ln.get_color(), alpha=0.25, lw=0)
        ax[0, c].set_title(p, fontsize=9)
        ax[0, c].set(ylabel="g(r)")
        ax[1, c].set(xlabel="r (Å)", ylabel="n(r)", xlim=(0, args.rplot))
        if args.box:
            for a in (ax[0, c], ax[1, c]):
                a.axvline(args.box / 2, color="gray", ls=":", lw=1)
        ax[0, c].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150)
    print(f"wrote {args.out}.png")

    if args.zgroups:
        zg = args.zgroups.split(",")
        zsets = []
        for (label, _), path in zip(sets, paths):
            zp = zprofile_path(path)
            if os.path.exists(zp):
                zsets.append((label, load(zp, "z")))
            else:
                print(f"note: {zp} not found - skipped in z-profile plot")
        if zsets:
            fig, ax = plt.subplots(1, len(zg), figsize=(4.5 * len(zg), 3.6), squeeze=False, sharex=True)
            for c, g in enumerate(zg):
                for label, d in zsets:
                    if g in d:
                        ax[0, c].plot(d["z"], d[g], label=label)
                ax[0, c].set(title=g, xlabel="z (Å, wrapped)", ylabel="density (atoms/Å$^3$)")
                ax[0, c].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(args.out + "_zprofile.png", dpi=150)
            print(f"wrote {args.out}_zprofile.png")


if __name__ == "__main__":
    main()