#!/usr/bin/env python3
"""
mace_md.py - ASE molecular dynamics driver for a MACE MLIP, built for
Pt / graphene / Nafion-analog / water validation runs.

Sub-commands
------------
  run   minimise -> equilibrate -> production MD, with a detailed log,
        thermo.csv, trajectory, restart file, RDFs and z-density profiles.
  rdf   re-compute RDFs / z-profiles from ANY extxyz/xyz trajectory
        (use this on your CP2K AIMD trajectory so MLIP-MD and AIMD are
        analysed with exactly the same code and atom-group definitions).

Ensembles
---------
  nvt : Langevin thermostat (default).
  nve : Langevin equilibration, then velocity-Verlet production.
Neither needs stress. Only NPT would need a stress-trained model.
Use --dtype float64 for NVE drift tests.

Examples
--------
  python mace_md.py run --structure sys.xyz --model V8.model --head Default \
      --temperature 300 --equil-ps 20 --prod-ps 1000 --outdir run_300K

  python mace_md.py rdf --traj run_300K/traj.extxyz --start 100 --out run_300K/rdf_check
  python mace_md.py rdf --traj aimd.xyz --stride 5 --out aimd_rdf

Atom groups (assigned from the FIRST frame, held fixed by index)
  Pt, C, F, S        by element
  Os                 O bonded to S (sulfonate oxygens)
  Ow                 all other O (water / hydronium)
  Hw / Hs / Hc       H bonded to Ow / to Os / to neither (e.g. C-H)
RDF pairs are written as "A-B", e.g. Pt-Ow,Ow-Ow,Pt-Os.

Note on slab RDFs: g(r) is normalised with the mean cell volume, which
includes vacuum. Absolute g(r) heights are therefore not bulk-like. Compare
MLIP vs AIMD (same cell) rather than to bulk. The cumulative coordination
number n(r) is unambiguous and is written alongside g(r).
"""
import argparse
import csv
import json
import logging
import os
import platform
import sys
import time

import numpy as np
from ase import units
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.geometry import find_mic
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.neighborlist import neighbor_list
from ase.optimize import FIRE
from ase.calculators.singlepoint import SinglePointCalculator

log = logging.getLogger("mace_md")

DEFAULT_PAIRS = ("Pt-Pt,Pt-Ow,Pt-Os,Pt-S,Pt-Hw,Ow-Ow,Ow-Hw,Os-Hw,S-Ow")
GROUP_ORDER = ["Pt", "C", "F", "S", "Ow", "Os", "Hw", "Hs", "Hc"]


# --------------------------------------------------------------------------
# Groups
# --------------------------------------------------------------------------
def build_groups(atoms):
    sym = np.array(atoms.get_chemical_symbols())
    n = len(atoms)
    isS, isO, isH = sym == "S", sym == "O", sym == "H"
    i, j, d = neighbor_list("ijd", atoms, 1.9)

    Os = np.zeros(n, bool)
    sel = isS[i] & isO[j]
    Os[j[sel]] = True
    Ow = isO & ~Os

    Hs = np.zeros(n, bool)
    Hw = np.zeros(n, bool)
    Hc = isH.copy()
    selh = isH[i] & isO[j] & (d < 1.25)
    for hi, oj in zip(i[selh], j[selh]):
        Hc[hi] = False
        if Os[oj]:
            Hs[hi] = True
        else:
            Hw[hi] = True

    return {"Pt": sym == "Pt", "C": sym == "C", "F": sym == "F", "S": isS,
            "Ow": Ow, "Os": Os, "Hw": Hw & ~Hs, "Hs": Hs, "Hc": Hc}


def parse_pairs(s, groups):
    """Parse "A-B,C-D". With the default pair list, pairs involving groups that do not
    exist in this system are dropped, and C-C is added if carbon is present
    (graphene-only systems would otherwise have no RDF at all)."""
    if s == DEFAULT_PAIRS:
        keep = [t for t in s.split(",") if all(groups[g].any() for g in t.split("-"))]
        if groups["C"].any():
            keep.append("C-C")
        s = ",".join(keep)
        if not s:
            raise SystemExit("No default RDF pairs apply to this system; give --rdf-pairs explicitly")
    pairs = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        a, b = tok.split("-")
        for g in (a, b):
            if g not in groups:
                raise SystemExit(f"Unknown group '{g}' in pair '{tok}'. Groups: {list(groups)}")
        pairs.append((a, b))
    return pairs


# --------------------------------------------------------------------------
# RDF and z-profile accumulators
# --------------------------------------------------------------------------
class RDFAccumulator:
    def __init__(self, groups, pairs, rmax=8.0, dr=0.02):
        self.groups, self.pairs, self.rmax = groups, pairs, rmax
        self.edges = np.arange(0.0, rmax + 0.5 * dr, dr)
        self.hist = {p: np.zeros(len(self.edges) - 1) for p in pairs}
        self.nframes = 0
        self.vol = 0.0

    def add(self, atoms):
        i, j, d = neighbor_list("ijd", atoms, self.rmax)
        for (a, b) in self.pairs:
            sel = self.groups[a][i] & self.groups[b][j]
            self.hist[(a, b)] += np.histogram(d[sel], bins=self.edges)[0]
        self.nframes += 1
        self.vol += atoms.get_volume()

    def write(self, path):
        if self.nframes == 0:
            return
        e = self.edges
        r = 0.5 * (e[1:] + e[:-1])
        shell = 4.0 * np.pi / 3.0 * (e[1:] ** 3 - e[:-1] ** 3)
        V = self.vol / self.nframes
        cols, names = [r], ["r"]
        info = []
        for (a, b), h in self.hist.items():
            Na, Nb = int(self.groups[a].sum()), int(self.groups[b].sum())
            nb = Nb - 1 if a == b else Nb
            if Na == 0 or nb <= 0:
                g = np.full_like(r, np.nan)
                cn = np.full_like(r, np.nan)
            else:
                g = h / (self.nframes * Na * (nb / V) * shell)
                cn = np.cumsum(h) / (self.nframes * Na)
            cols += [g, cn]
            names += [f"g_{a}-{b}", f"n_{a}-{b}"]
            info.append(f"{a}-{b}: N_a={Na} N_b={Nb}")
        header = (f"RDF over {self.nframes} frames, mean cell volume {V:.1f} A^3 "
                  f"(cell-volume normalisation, see script docstring)\n"
                  + "; ".join(info) + "\n" + " ".join(names))
        np.savetxt(path, np.column_stack(cols), header=header, fmt="%.6f")


class ZProfile:
    """Number density (atoms / A^3) of each group along z (wrapped)."""

    def __init__(self, groups, cell_z, dz=0.25):
        self.groups = groups
        self.nb = max(1, int(round(cell_z / dz)))
        self.edges = np.linspace(0.0, cell_z, self.nb + 1)
        self.hist = {g: np.zeros(self.nb) for g, m in groups.items() if m.any()}
        self.nframes = 0
        self.area = 0.0

    def add(self, atoms):
        z = atoms.get_positions(wrap=True)[:, 2]
        for g in self.hist:
            self.hist[g] += np.histogram(z[self.groups[g]], bins=self.edges)[0]
        self.nframes += 1
        self.area += np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))

    def write(self, path):
        if self.nframes == 0:
            return
        dz = self.edges[1] - self.edges[0]
        zc = 0.5 * (self.edges[1:] + self.edges[:-1])
        A = self.area / self.nframes
        cols = [zc] + [self.hist[g] / (self.nframes * A * dz) for g in self.hist]
        header = (f"z-density (atoms/A^3) over {self.nframes} frames; z is wrapped into the cell\n"
                  "z " + " ".join(self.hist))
        np.savetxt(path, np.column_stack(cols), header=header, fmt="%.6e")


# --------------------------------------------------------------------------
# Structural diagnostics (cheap sanity checks logged every log-interval)
# --------------------------------------------------------------------------
class Diagnostics:
    SKELETON = {("C", "C"), ("C", "F"), ("C", "S"), ("C", "O"), ("O", "S")}

    def __init__(self, atoms0, groups):
        self.sym = np.array(atoms0.get_chemical_symbols())
        self.Z = atoms0.get_atomic_numbers()
        self.rad = covalent_radii[self.Z]
        self.g = groups
        self.isO = self.sym == "O"
        self.isH = self.sym == "H"
        self.ref_pos = atoms0.get_positions().copy()

        i, j, d = neighbor_list("ijd", atoms0, 2.6)
        bi, bj = [], []
        for a, b, dist in zip(i, j, d):
            if a < b and tuple(sorted((self.sym[a], self.sym[b]))) in self.SKELETON \
                    and dist < 1.2 * (self.rad[a] + self.rad[b]):
                bi.append(a)
                bj.append(b)
        self.bi, self.bj = np.array(bi, int), np.array(bj, int)
        self.bthr = 1.5 * (self.rad[self.bi] + self.rad[self.bj]) if len(bi) else np.array([])

    def compute(self, atoms):
        out = {}
        n = len(atoms)
        i, j, d = neighbor_list("ijd", atoms, 3.5)

        # generic closest-approach check, relative to covalent radii sum
        ratio = d / (self.rad[i] + self.rad[j])
        k = int(np.argmin(ratio))
        out["dmin_ratio"] = float(ratio[k])
        out["dmin_pair"] = f"{self.sym[i[k]]}{i[k]}-{self.sym[j[k]]}{j[k]}"
        out["dmin"] = float(d[k])

        # covalent skeleton (Nafion backbone/sulfonate + graphene) integrity
        if len(self.bi):
            dd = atoms.get_distances(self.bi, self.bj, mic=True)
            broken = dd > self.bthr
            out["n_broken"] = int(broken.sum())
            out["broken_list"] = [(int(self.bi[m]), int(self.bj[m]), float(dd[m]))
                                  for m in np.where(broken)[0][:5]]
        else:
            out["n_broken"], out["broken_list"] = 0, []

        # oxygen protonation states (O with n H within 1.25 A)
        selh = self.isO[i] & self.isH[j] & (d < 1.25)
        nH = np.bincount(i[selh], minlength=n)
        nHw = nH[self.g["Ow"]]
        out["OH_minus"] = int((nHw == 1).sum())
        out["H2O"] = int((nHw == 2).sum())
        out["H3O"] = int((nHw == 3).sum())
        out["Ow_odd"] = int(((nHw == 0) | (nHw >= 4)).sum())
        out["Os_prot"] = int((nH[self.g["Os"]] >= 1).sum())

        # Pt metrics
        Pt = self.g["Pt"]
        if Pt.any():
            selpp = Pt[i] & Pt[j] & (d < 3.2)
            cn = np.bincount(i[selpp], minlength=n)[Pt]
            out["Pt_CN_min"] = int(cn.min())
            out["Pt_isolated"] = int((cn == 0).sum())
            selpo = Pt[i] & self.g["Os"][j] & (d < 2.6)
            out["PtOs_contacts"] = int(selpo.sum())
            out["Pt_with_Os"] = int(len(np.unique(i[selpo])))
            dv = atoms.get_positions()[Pt] - self.ref_pos[Pt]
            dv, dl = find_mic(dv, atoms.cell, atoms.pbc)
            out["Pt_maxdisp"] = float(dl.max())
        else:
            out.update(Pt_CN_min=0, Pt_isolated=0, PtOs_contacts=0, Pt_with_Os=0, Pt_maxdisp=0.0)
        return out


# --------------------------------------------------------------------------
# Calculator
# --------------------------------------------------------------------------
def make_calculator(args):
    if args.calculator == "lj":  # dry-run only, to test the script flow
        from ase.calculators.lj import LennardJones
        return LennardJones(sigma=2.2, epsilon=0.01, rc=6.0)
    if args.foundation:  # e.g. medium-mpa-0, medium, medium-omat-0 (downloads on first use)
        from mace.calculators import mace_mp
        return mace_mp(model=args.foundation, device=args.device, default_dtype=args.dtype,
                       dispersion=args.d3)
    from mace.calculators import MACECalculator
    kw = dict(model_paths=args.model, device=args.device, default_dtype=args.dtype)
    if args.head:
        kw["head"] = args.head
    if args.enable_cueq:
        kw["enable_cueq"] = True
    return MACECalculator(**kw)


def log_environment(args):
    log.info("=" * 100)
    log.info("mace_md.py  %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("host %s | python %s", platform.node(), platform.python_version())
    try:
        import ase
        log.info("ase %s", ase.__version__)
    except Exception:
        pass
    try:
        import torch
        log.info("torch %s | cuda available: %s", torch.__version__, torch.cuda.is_available())
        if torch.cuda.is_available():
            log.info("GPU: %s", torch.cuda.get_device_name(0))
    except Exception:
        pass
    try:
        import mace
        log.info("mace %s", getattr(mace, "__version__", "unknown"))
    except Exception:
        pass
    log.info("command line: %s", " ".join(sys.argv))
    log.info("settings: %s", json.dumps({k: v for k, v in vars(args).items() if k != "func"}, default=str))


# --------------------------------------------------------------------------
# MD driver
# --------------------------------------------------------------------------
class MDAbort(Exception):
    pass


class Observer:
    def __init__(self, atoms, diag, args, paths, rdf, zprof, dt_fs, n_atoms):
        self.atoms, self.diag, self.args, self.paths = atoms, diag, args, paths
        self.rdf, self.zprof, self.dt = rdf, zprof, dt_fs
        self.N = n_atoms
        self.phase = "equil"
        self.step0 = 0            # global step at start of current phase
        self.dyn = None
        self.t_wall0 = time.time()
        self.step_wall0 = 0
        self.e_ref = None
        self.prev_broken = 0
        self.prev_isolated = 0
        self.min_ratio_seen = 9.9
        self.csv_file = open(paths["csv"], "a", newline="")
        self.csv = csv.writer(self.csv_file)
        if os.path.getsize(paths["csv"]) == 0:
            self.csv.writerow(["step", "time_ps", "phase", "T_K", "Epot_eV", "Ekin_eV", "Etot_eV",
                               "Fmax", "Frms", "dmin_ratio", "n_broken", "Pt_isolated", "PtOs_contacts",
                               "Pt_maxdisp", "H3O", "OH_minus", "H2O", "Os_prot"])
        self.header_every = 40
        self.nlines = 0

    def gstep(self):
        return self.step0 + self.dyn.nsteps

    def set_phase(self, phase, dyn):
        self.phase, self.dyn, self.step0 = phase, dyn, self.step_offset_for(phase)
        self.e_ref = None

    def step_offset_for(self, phase):
        return getattr(self, "_next_step0", 0)

    def close_phase(self):
        self._next_step0 = self.gstep()

    # called from dyn.attach at every step; cheap filters inside
    def every_step(self):
        s = self.gstep()
        a = self.args
        if self.phase == "prod":
            if s % a.rdf_interval == 0:
                self.rdf.add(self.atoms)
                self.zprof.add(self.atoms)
            if s % a.rdf_write_interval == 0:
                self.write_analysis()
        if s % a.traj_interval == 0:
            self.write_frame(self.paths["traj"], append=True)
        if s % a.restart_interval == 0:
            self.write_frame(self.paths["restart"], append=False)
        if s % a.log_interval == 0:
            self.log_state(s)

    def write_analysis(self):
        self.rdf.write(self.paths["rdf"])
        self.zprof.write(self.paths["zprof"])

    def write_frame(self, path, append):
        at = self.atoms.copy()
        if self.args.traj_forces or not append:
            at.calc = SinglePointCalculator(at, energy=self.atoms.get_potential_energy(),
                                            forces=self.atoms.get_forces())
        at.info["md_step"] = int(self.gstep())
        at.info["md_phase"] = self.phase
        at.info["time_ps"] = self.gstep() * self.dt / 1000.0
        write(path, at, format="extxyz", append=append)

    def log_state(self, s):
        at = self.atoms
        epot = at.get_potential_energy()
        ekin = at.get_kinetic_energy()
        etot = epot + ekin
        T = at.get_temperature()
        f = at.get_forces()
        fn = np.linalg.norm(f, axis=1)
        fmax, frms = float(fn.max()), float(np.sqrt((fn ** 2).mean()))
        dg = self.diag.compute(at)
        t_ps = s * self.dt / 1000.0
        if self.e_ref is None:
            self.e_ref = etot
        drift = (etot - self.e_ref) / self.N * 1000.0

        elapsed = time.time() - self.t_wall0
        nsday = ((s - self.step_wall0) * self.dt * 1e-6) / (elapsed / 86400.0) if elapsed > 0 else 0.0

        if self.nlines % self.header_every == 0:
            log.info("%9s %9s %-6s %7s %12s %12s %10s %8s %8s | %s", "step", "t(ps)", "phase", "T(K)",
                     "Epot(eV)", "Etot(eV)", "dEtot/at", "Fmax", "Frms", "ns/day")
            log.info("%9s %9s %-6s %7s %12s %12s %10s %8s %8s   %s", "", "", "", "", "", "", "(meV)",
                     "(eV/A)", "(eV/A)",
                     "diag: dmin/rsum(pair) | broken | Pt_iso Pt_CNmin PtOs(nPt) Pt_maxdisp | H3O OH- H2O Os_H")
        self.nlines += 1
        log.info("%9d %9.3f %-6s %7.1f %12.4f %12.4f %10.3f %8.3f %8.3f | %.1f", s, t_ps, self.phase, T,
                 epot, etot, drift, fmax, frms, nsday)
        log.info("%9s %9s %-6s %7s %12s %12s %10s %8s %8s   %.2f(%s) | %d | %d %d %d(%d) %.2f | %d %d %d %d",
                 "", "", "", "", "", "", "", "", "", dg["dmin_ratio"], dg["dmin_pair"], dg["n_broken"],
                 dg["Pt_isolated"], dg["Pt_CN_min"], dg["PtOs_contacts"], dg["Pt_with_Os"], dg["Pt_maxdisp"],
                 dg["H3O"], dg["OH_minus"], dg["H2O"], dg["Os_prot"])

        self.csv.writerow([s, f"{t_ps:.4f}", self.phase, f"{T:.2f}", f"{epot:.6f}", f"{ekin:.6f}",
                           f"{etot:.6f}", f"{fmax:.4f}", f"{frms:.4f}", f"{dg['dmin_ratio']:.3f}",
                           dg["n_broken"], dg["Pt_isolated"], dg["PtOs_contacts"], f"{dg['Pt_maxdisp']:.3f}",
                           dg["H3O"], dg["OH_minus"], dg["H2O"], dg["Os_prot"]])
        self.csv_file.flush()

        # ---- events / warnings ----
        if dg["n_broken"] > self.prev_broken:
            log.warning("WARN step %d: covalent skeleton bonds broken: %d (first: %s)", s, dg["n_broken"],
                        [(self.diag.sym[a] + str(a), self.diag.sym[b] + str(b), round(x, 2))
                         for a, b, x in dg["broken_list"]])
        self.prev_broken = dg["n_broken"]
        if dg["dmin_ratio"] < 0.70 and dg["dmin_ratio"] < self.min_ratio_seen - 0.02:
            log.warning("WARN step %d: very close contact %s d=%.2f A (%.2f x covalent sum)", s,
                        dg["dmin_pair"], dg["dmin"], dg["dmin_ratio"])
        self.min_ratio_seen = min(self.min_ratio_seen, dg["dmin_ratio"])
        if dg["Pt_isolated"] != self.prev_isolated:
            log.info("EVENT step %d: number of Pt atoms with no Pt neighbour within 3.2 A: %d -> %d",
                     s, self.prev_isolated, dg["Pt_isolated"])
            self.prev_isolated = dg["Pt_isolated"]
            if self.args.save_events:
                self.write_frame(self.paths["events"], append=True)

        # ---- abort criteria ----
        if not np.isfinite(epot) or not np.all(np.isfinite(f)):
            raise MDAbort("non-finite energy/forces")
        if fmax > self.args.abort_fmax:
            raise MDAbort(f"max force {fmax:.1f} eV/A exceeds --abort-fmax")
        if self.phase == "prod" and T > self.args.abort_temp_factor * self.args.temperature:
            raise MDAbort(f"T={T:.0f} K exceeds {self.args.abort_temp_factor}x target")


def cmd_run(args):
    os.makedirs(args.outdir, exist_ok=True)
    paths = {k: os.path.join(args.outdir, v) for k, v in dict(
        log="md.log", csv="thermo.csv", traj="traj.extxyz", restart="restart.extxyz", rdf="rdf.dat",
        zprof="zprofile.dat", final="final.extxyz", summary="summary.json",
        events="events.extxyz", initial="initial_after_min.extxyz").items()}
    open(paths["csv"], "a").close()

    handlers = [logging.FileHandler(paths["log"], mode="a"), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers, force=True)
    log_environment(args)

    restarting = args.restart is not None
    atoms = read(args.restart if restarting else args.structure, index=-1)
    if not restarting and atoms.get_pbc().sum() == 0:
        log.warning("WARN: structure has no periodic boundary conditions")
    n = len(atoms)

    # composition and geometry
    groups = build_groups(atoms)
    log.info("system: %s | %d atoms | pbc=%s", atoms.get_chemical_formula(), n, atoms.pbc.tolist())
    log.info("cell (A): %s", np.round(atoms.cell.lengths(), 3).tolist() + np.round(atoms.cell.angles(), 2).tolist())
    log.info("groups: %s", {g: int(m.sum()) for g, m in groups.items() if m.any()})
    if groups["Os"].any() and groups["Ow"].any():
        log.info("water-like O: %d | sulfonate O: %d", groups["Ow"].sum(), groups["Os"].sum())

    # constraints
    if args.freeze_below is not None:
        els = set(args.freeze_elements.split(","))
        sym = np.array(atoms.get_chemical_symbols())
        mask = np.isin(sym, list(els)) & (atoms.positions[:, 2] < args.freeze_below)
        atoms.set_constraint(FixAtoms(mask=mask))
        log.info("frozen atoms (%s, z<%.2f): %d", args.freeze_elements, args.freeze_below, int(mask.sum()))

    atoms.calc = make_calculator(args)

    # ---------------- minimisation ----------------
    if not restarting and args.minimize_steps > 0:
        log.info("minimising with FIRE (fmax=%.3f eV/A, max %d steps)", args.fmax, args.minimize_steps)
        opt = FIRE(atoms, logfile=None)
        t0 = time.time()
        opt.run(fmax=args.fmax, steps=args.minimize_steps)
        f = np.linalg.norm(atoms.get_forces(), axis=1)
        log.info("minimisation done: %d steps, E=%.4f eV, Fmax=%.3f eV/A, %.1f s",
                 opt.nsteps, atoms.get_potential_energy(), f.max(), time.time() - t0)
        write(paths["initial"], atoms, format="extxyz")

    diag = Diagnostics(atoms, groups)
    log.info("covalent skeleton bonds tracked: %d", len(diag.bi))

    pairs = parse_pairs(args.rdf_pairs, groups)
    rdf = RDFAccumulator(groups, pairs, rmax=args.rdf_rmax, dr=args.rdf_dr)
    zprof = ZProfile(groups, atoms.cell[2, 2] if atoms.cell[2, 2] > 0 else 1.0, dz=args.z_dz)

    # velocities
    dt = args.timestep
    step_offset = 0
    if restarting:
        step_offset = int(atoms.info.get("md_step", 0))
        if atoms.get_momenta().any():
            log.info("restart: using momenta from %s (step %d)", args.restart, step_offset)
        else:
            log.warning("WARN: restart file has no momenta; drawing new velocities")
            MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=np.random.default_rng(args.seed))
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=np.random.default_rng(args.seed))
        if not atoms.constraints:
            Stationary(atoms)
    log.info("initial T = %.1f K", atoms.get_temperature())

    obs = Observer(atoms, diag, args, paths, rdf, zprof, dt, n)
    obs._next_step0 = step_offset

    def langevin():
        kw = dict(timestep=dt * units.fs, temperature_K=args.temperature,
                  friction=args.friction / (1000.0 * units.fs), rng=np.random.default_rng(args.seed + 1))
        try:
            return Langevin(atoms, fixcm=False, **kw)  # avoids deprecated COM handling in newer ASE
        except TypeError:
            return Langevin(atoms, **kw)

    n_eq = int(round(args.equil_ps * 1000.0 / dt))
    n_pr = int(round(args.prod_ps * 1000.0 / dt))
    log.info("dt = %.3f fs | equilibration %d steps (%.2f ps) | production %d steps (%.2f ps) | ensemble %s",
             dt, n_eq, args.equil_ps, n_pr, args.prod_ps, args.ensemble.upper())
    log.info("=" * 100)

    status = "completed"
    t_start = time.time()
    try:
        obs.t_wall0 = time.time()
        for phase, nsteps in (("equil", n_eq), ("prod", n_pr)):
            if nsteps <= 0:
                continue
            if phase == "prod" and args.ensemble == "nve":
                dyn = VelocityVerlet(atoms, timestep=dt * units.fs)
            else:
                dyn = langevin()
            obs.set_phase(phase, dyn)
            obs.step_wall0 = obs.step0
            obs.t_wall0 = time.time()
            log.info("---- phase: %s (%s) ----", phase, "NVE" if (phase == "prod" and args.ensemble == "nve")
                     else f"NVT Langevin, friction {args.friction} /ps")
            dyn.attach(obs.every_step, interval=1)
            dyn.run(nsteps)
            obs.close_phase()
    except MDAbort as e:
        status = f"aborted: {e}"
        log.error("ABORT at step %d: %s", obs.gstep(), e)
    except KeyboardInterrupt:
        status = "interrupted"
        log.error("interrupted by user at step %d", obs.gstep())

    # final outputs
    obs.write_analysis()
    obs.write_frame(paths["restart"], append=False)
    try:
        write(paths["final"], atoms, format="extxyz")
    except Exception:
        pass
    obs.csv_file.close()

    dg = diag.compute(atoms)
    summary = dict(status=status, n_atoms=n, final_step=int(obs.gstep()), final_time_ps=obs.gstep() * dt / 1000,
                   wall_s=time.time() - t_start, T_final=float(atoms.get_temperature()),
                   n_broken_final=dg["n_broken"], Pt_isolated_final=dg["Pt_isolated"],
                   dmin_ratio_min_seen=obs.min_ratio_seen, rdf_frames=rdf.nframes)
    with open(paths["summary"], "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("=" * 100)
    log.info("SUMMARY: %s", json.dumps(summary))
    log.info("outputs in %s: md.log thermo.csv traj.extxyz restart.extxyz rdf.dat zprofile.dat final.extxyz",
             args.outdir)
    return 0 if status == "completed" else 2


# --------------------------------------------------------------------------
# rdf sub-command (works on any trajectory, e.g. AIMD)
# --------------------------------------------------------------------------
def cmd_rdf(args):
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    sl = f"{args.start}:{args.stop if args.stop is not None else ''}:{args.stride}"
    frames = read(args.traj, index=sl)
    if not isinstance(frames, list):
        frames = [frames]
    ref = read(args.ref, index=0) if args.ref else frames[0]
    groups = build_groups(ref)
    log.info("frames: %d | groups: %s", len(frames), {g: int(m.sum()) for g, m in groups.items() if m.any()})
    pairs = parse_pairs(args.rdf_pairs, groups)
    rdf = RDFAccumulator(groups, pairs, rmax=args.rdf_rmax, dr=args.rdf_dr)
    zprof = ZProfile(groups, ref.cell[2, 2], dz=args.z_dz)
    for fr in frames:
        if len(fr) != len(ref):
            raise SystemExit("Atom count differs between frames and reference")
        rdf.add(fr)
        zprof.add(fr)
    rdf.write(args.out + "_rdf.dat")
    zprof.write(args.out + "_zprofile.dat")
    log.info("wrote %s_rdf.dat and %s_zprofile.dat", args.out, args.out)
    return 0


# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_rdf_args(q):
        q.add_argument("--rdf-pairs", default=DEFAULT_PAIRS)
        q.add_argument("--rdf-rmax", type=float, default=8.0)
        q.add_argument("--rdf-dr", type=float, default=0.02)
        q.add_argument("--z-dz", type=float, default=0.25, help="z-profile bin width (A)")

    r = sub.add_parser("run", help="run MD")
    r.add_argument("--structure", help="extxyz/xyz/POSCAR with cell + pbc")
    r.add_argument("--restart", help="extxyz written by a previous run (restart.extxyz)")
    r.add_argument("--model", help="MACE .model file")
    r.add_argument("--foundation", default=None,
                   help="use a MACE-MP foundation model instead of --model, e.g. medium-mpa-0")
    r.add_argument("--d3", action="store_true", help="add D3 dispersion (foundation models only; needs torch-dftd)")
    r.add_argument("--head", default=None, help="head name for multihead models")
    r.add_argument("--device", default="cuda")
    r.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    r.add_argument("--enable-cueq", action="store_true")
    r.add_argument("--calculator", default="mace", choices=["mace", "lj"], help="'lj' = dry run only")
    r.add_argument("--ensemble", default="nvt", choices=["nvt", "nve"])
    r.add_argument("--temperature", type=float, default=300.0)
    r.add_argument("--timestep", type=float, default=0.5, help="fs")
    r.add_argument("--friction", type=float, default=5.0, help="Langevin friction, 1/ps")
    r.add_argument("--minimize-steps", type=int, default=200)
    r.add_argument("--fmax", type=float, default=0.05)
    r.add_argument("--equil-ps", type=float, default=20.0)
    r.add_argument("--prod-ps", type=float, default=100.0)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--freeze-below", type=float, default=None, help="freeze atoms below this z (A)")
    r.add_argument("--freeze-elements", default="Pt", help="elements affected by --freeze-below")
    r.add_argument("--outdir", default="md_run")
    r.add_argument("--log-interval", type=int, default=200, help="steps")
    r.add_argument("--traj-interval", type=int, default=400, help="steps")
    r.add_argument("--no-traj-forces", dest="traj_forces", action="store_false",
                   help="do not store energy/forces in trajectory frames")
    r.add_argument("--restart-interval", type=int, default=2000)
    r.add_argument("--rdf-interval", type=int, default=20)
    r.add_argument("--rdf-write-interval", type=int, default=5000)
    r.add_argument("--save-events", action="store_true",
                   help="write a frame to events.extxyz whenever the isolated-Pt count changes")
    r.add_argument("--abort-fmax", type=float, default=50.0, help="abort if max |F| exceeds this (eV/A)")
    r.add_argument("--abort-temp-factor", type=float, default=3.0)
    add_rdf_args(r)
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("rdf", help="RDF / z-profile from a trajectory")
    d.add_argument("--traj", required=True)
    d.add_argument("--ref", default=None, help="frame used to define groups (default: first frame)")
    d.add_argument("--start", type=int, default=0)
    d.add_argument("--stop", type=int, default=None)
    d.add_argument("--stride", type=int, default=1)
    d.add_argument("--out", default="analysis")
    add_rdf_args(d)
    d.set_defaults(func=cmd_rdf)
    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "run":
        if args.restart is None and args.structure is None:
            raise SystemExit("run: give --structure or --restart")
        if args.calculator == "mace" and not (args.model or args.foundation):
            raise SystemExit("run: give --model or --foundation")
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()