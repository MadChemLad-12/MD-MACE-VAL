#!/usr/bin/env python3
"""
compare_models.py - evaluate two MLIPs on IDENTICAL frames and report how they differ.

Model A : your MACE model           (--model-a, --head)
Model B : a MACE-MP foundation model (--foundation, e.g. medium-mpa-0, optional --d3)
          or a second local model    (--model-b)
Optional: --dft extxyz with DFT energies/forces for the same frames (same order
          and count as the frames selected from --traj). Then A and B are also
          scored against DFT.

Why frames from a trajectory: comparing two separate MD runs mixes model
differences with sampling noise. Evaluating both models on the same structures
isolates the model difference.

Energies: absolute energies are not comparable between models (different E0 /
reference conventions, D3 offset). Energies are therefore compared as
fluctuations about each model's own mean over the selected frames.

Outputs (prefix given by --out):
  <out>_summary.txt / .json  force and energy statistics, per element and per group
  <out>_frames.csv           per-frame dE/atom, force RMSE, max force difference
  <out>.png                  parity plots, error vs time, error by group (needs matplotlib)

Example
  python compare_models.py --traj run/traj.extxyz --start 101 --max-frames 200 \
      --model-a V8.model --head Default --foundation medium-mpa-0 --d3 --out run/compare_mpa
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from ase.io import read

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mace_md import build_groups  # noqa: E402  (same group definitions as the MD script)


def make_calc(kind, args):
    if kind == "lj-test":
        from ase.calculators.lj import LennardJones
        return LennardJones(sigma=2.2, epsilon=0.01, rc=6.0)
    if kind == "lj-test2":
        from ase.calculators.lj import LennardJones
        return LennardJones(sigma=2.3, epsilon=0.012, rc=6.0)
    if kind == "a":
        from mace.calculators import MACECalculator
        kw = dict(model_paths=args.model_a, device=args.device, default_dtype=args.dtype)
        if args.head:
            kw["head"] = args.head
        return MACECalculator(**kw)
    if kind == "b":
        if args.foundation:
            from mace.calculators import mace_mp
            return mace_mp(model=args.foundation, device=args.device, default_dtype=args.dtype,
                           dispersion=args.d3)
        from mace.calculators import MACECalculator
        kw = dict(model_paths=args.model_b, device=args.device, default_dtype=args.dtype)
        if args.head_b:
            kw["head"] = args.head_b
        return MACECalculator(**kw)
    raise ValueError(kind)


def evaluate(frames, calc, label):
    E, F = [], []
    for k, fr in enumerate(frames):
        at = fr.copy()
        at.calc = calc
        E.append(at.get_potential_energy())
        F.append(at.get_forces().copy())
        if (k + 1) % 20 == 0 or k + 1 == len(frames):
            print(f"  {label}: {k + 1}/{len(frames)} frames", flush=True)
    return np.array(E), np.array(F)


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def fluct_stats(EA, EB, N):
    dA, dB = EA - EA.mean(), EB - EB.mean()
    d = dA - dB
    r = float(np.corrcoef(EA, EB)[0, 1]) if len(EA) > 2 and EA.std() > 0 and EB.std() > 0 else float("nan")
    return dict(rmse_meV_per_atom=1000 * rmse(d) / N, corr=r, std_A_meV_per_atom=1000 * EA.std() / N,
                std_B_meV_per_atom=1000 * EB.std() / N)


def force_stats(FA, FB, masks):
    """FA, FB: (frames, N, 3); returns dict overall / per group (meV/A)."""
    d = FA - FB
    out = {}

    def one(sel):
        dd, ref = d[:, sel, :], FB[:, sel, :]
        if dd.size == 0:
            return None
        cosn = np.sum(FA[:, sel, :] * ref, axis=-1)
        cosd = np.linalg.norm(FA[:, sel, :], axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-12
        return dict(n_atoms=int(sel.sum()), rmse_meV_A=1000 * rmse(dd),
                    mae_meV_A=1000 * float(np.mean(np.abs(dd))),
                    rms_ref_meV_A=1000 * rmse(ref),
                    rel_rmse=rmse(dd) / (rmse(ref) + 1e-12),
                    mean_cos=float(np.mean(cosn / cosd)))

    out["all"] = one(np.ones(FA.shape[1], bool))
    for g, m in masks.items():
        if m.any():
            out[g] = one(m)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=200, help="subsample evenly to at most this many frames")
    p.add_argument("--ref", default=None, help="structure used to define atom groups (default: first frame)")
    p.add_argument("--model-a", default=None)
    p.add_argument("--head", default=None)
    p.add_argument("--foundation", default=None, help="model B: MACE-MP name, e.g. medium-mpa-0")
    p.add_argument("--d3", action="store_true")
    p.add_argument("--model-b", default=None, help="model B: local .model file")
    p.add_argument("--head-b", default=None)
    p.add_argument("--dft", default=None, help="extxyz with DFT energy/forces for the selected frames")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--out", default="compare")
    p.add_argument("--test-calculators", action="store_true", help="LJ dry run of the script flow")
    args = p.parse_args()

    if not args.test_calculators and not args.model_a:
        sys.exit("give --model-a")
    if not args.test_calculators and not (args.foundation or args.model_b):
        sys.exit("give --foundation or --model-b for model B")

    sl = f"{args.start}:{args.stop if args.stop is not None else ''}:{args.stride}"
    frames = read(args.traj, index=sl)
    frames = frames if isinstance(frames, list) else [frames]
    if len(frames) > args.max_frames:
        idx = np.unique(np.linspace(0, len(frames) - 1, args.max_frames).round().astype(int))
        frames = [frames[i] for i in idx]
    else:
        idx = np.arange(len(frames))
    N = len(frames[0])
    steps = [fr.info.get("md_step", int(i)) for fr, i in zip(frames, idx)]
    times = [fr.info.get("time_ps", np.nan) for fr in frames]
    ref = read(args.ref, index=0) if args.ref else frames[0]
    masks = build_groups(ref)
    print(f"{len(frames)} frames, {N} atoms; groups: { {g: int(m.sum()) for g, m in masks.items() if m.any()} }")

    kA, kB = ("lj-test", "lj-test2") if args.test_calculators else ("a", "b")
    EA, FA = evaluate(frames, make_calc(kA, args), "model A")
    EB, FB = evaluate(frames, make_calc(kB, args), "model B")

    report = {"n_frames": len(frames), "n_atoms": N,
              "A_vs_B": {"energy_fluct": fluct_stats(EA, EB, N), "forces": force_stats(FA, FB, masks)}}

    if args.dft:
        dft = read(args.dft, index=":")
        dft = dft if isinstance(dft, list) else [dft]
        if len(dft) != len(frames):
            sys.exit(f"--dft has {len(dft)} frames but {len(frames)} frames were selected")
        ED = np.array([d.get_potential_energy() for d in dft])
        FD = np.array([d.get_forces() for d in dft])
        report["A_vs_DFT"] = {"energy_fluct": fluct_stats(EA, ED, N), "forces": force_stats(FA, FD, masks)}
        report["B_vs_DFT"] = {"energy_fluct": fluct_stats(EB, ED, N), "forces": force_stats(FB, FD, masks)}

    # ------------------------------------------------------------ text report
    lines = []
    for key in [k for k in report if k.endswith(("_vs_B", "_vs_DFT"))]:
        r = report[key]
        ef = r["energy_fluct"]
        lines.append(f"=== {key.replace('_', ' ')} ({len(frames)} frames) ===")
        lines.append(f"energy fluctuation difference: RMSE {ef['rmse_meV_per_atom']:.2f} meV/atom | "
                     f"correlation {ef['corr']:.4f} | std first {ef['std_A_meV_per_atom']:.2f}, "
                     f"second {ef['std_B_meV_per_atom']:.2f} meV/atom")
        lines.append(f"{'group':>6} {'n':>6} {'RMSE':>9} {'MAE':>9} {'rms(ref)':>10} {'rel':>7} {'<cos>':>7}   (forces, meV/A)")
        for g, s in r["forces"].items():
            if s:
                lines.append(f"{g:>6} {s['n_atoms']:>6} {s['rmse_meV_A']:>9.1f} {s['mae_meV_A']:>9.1f} "
                             f"{s['rms_ref_meV_A']:>10.1f} {s['rel_rmse']:>7.3f} {s['mean_cos']:>7.3f}")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    with open(args.out + "_summary.txt", "w") as fh:
        fh.write(txt + "\n")
    with open(args.out + "_summary.json", "w") as fh:
        json.dump(report, fh, indent=2)

    # ------------------------------------------------------------ per frame
    dF = FA - FB
    dE = ((EA - EA.mean()) - (EB - EB.mean())) / N * 1000.0
    frame_rmse = np.sqrt((dF ** 2).mean(axis=(1, 2))) * 1000.0
    frame_max = np.linalg.norm(dF, axis=2).max(axis=1) * 1000.0
    worst_atom = np.linalg.norm(dF, axis=2).argmax(axis=1)
    sym = np.array(ref.get_chemical_symbols())
    with open(args.out + "_frames.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_in_traj_slice", "md_step", "time_ps", "dE_fluct_meV_per_atom",
                    "force_rmse_meV_A", "force_maxdiff_meV_A", "worst_atom"])
        for k in range(len(frames)):
            w.writerow([int(idx[k]), steps[k], times[k], f"{dE[k]:.3f}", f"{frame_rmse[k]:.1f}",
                        f"{frame_max[k]:.1f}", f"{sym[worst_atom[k]]}{worst_atom[k]}"])
    worst = np.argsort(frame_rmse)[::-1][:5]
    print("largest force disagreement (feed these to DFT / the next active-learning round):")
    for k in worst:
        print(f"  slice-frame {int(idx[k])} step {steps[k]}: RMSE {frame_rmse[k]:.0f} meV/A, "
              f"max {frame_max[k]:.0f} meV/A on {sym[worst_atom[k]]}{worst_atom[k]}")

    # ------------------------------------------------------------ plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available: skipping plot")
        return
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    a, b = FA.reshape(-1), FB.reshape(-1)
    sel = np.random.default_rng(0).choice(a.size, size=min(a.size, 20000), replace=False)
    ax[0, 0].scatter(b[sel], a[sel], s=2, alpha=0.3)
    lim = np.abs(np.r_[a[sel], b[sel]]).max()
    ax[0, 0].plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax[0, 0].set(xlabel="model B force component (eV/Å)", ylabel="model A (eV/Å)", title="Force parity")
    dEA, dEB = (EA - EA.mean()) / N * 1000, (EB - EB.mean()) / N * 1000
    ax[0, 1].scatter(dEB, dEA, s=8)
    lo, hi = min(dEA.min(), dEB.min()), max(dEA.max(), dEB.max())
    ax[0, 1].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax[0, 1].set(xlabel="B: E - <E> (meV/atom)", ylabel="A: E - <E> (meV/atom)", title="Energy fluctuations")
    x = np.array(times) if np.isfinite(times).all() else np.array(steps)
    ax[1, 0].plot(x, frame_rmse, ".-", ms=3)
    ax[1, 0].set(xlabel="time (ps)" if np.isfinite(times).all() else "MD step",
                 ylabel="force RMSE A-B (meV/Å)", title="Model disagreement along the trajectory")
    gs = [(g, s["rmse_meV_A"]) for g, s in report["A_vs_B"]["forces"].items() if s and g != "all"]
    ax[1, 1].bar([g for g, _ in gs], [v for _, v in gs])
    ax[1, 1].set(ylabel="force RMSE A-B (meV/Å)", title="Disagreement by atom group")
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150)
    print(f"wrote {args.out}_summary.txt, _summary.json, _frames.csv, .png")


if __name__ == "__main__":
    main()