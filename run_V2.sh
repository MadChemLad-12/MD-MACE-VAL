#!/usr/bin/env bash
# run_md_validation.sh - MACE MD validation over all test structures.
#
# Usage:
#   bash run_md_validation.sh                 # all structures, sequential
#   bash run_md_validation.sh Ptclose Ptwater # only the named structures
#   NGPU=2 bash run_md_validation.sh          # 2 jobs in parallel, one per GPU
#   PROD_PS=200 bash run_md_validation.sh     # override any setting below
#
# Re-running the script is safe: finished systems are skipped and interrupted
# ones resume from restart.extxyz (production only).
set -uo pipefail

# ----------------------------------------------------------------- settings
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mace_env

VALIDATE_SCRIPT="/home/h.jack/MACE_train/MD-val/mace_md.py"
MACE_MODEL="/home/h.jack/MACE_train/mace_active_learning/mace_V8_active_learning_stagetwo.model"
HEAD="Default"
STRUCT_DIR="/home/h.jack/MACE_train/MD-val"
OUT_ROOT="${OUT_ROOT:-$STRUCT_DIR/runs}"

TEMPS="${TEMPS:-300}"          # space separated, e.g. "300 350"
DT="${DT:-0.5}"                # fs
EQUIL_PS="${EQUIL_PS:-20}"
PROD_PS="${PROD_PS:-300}"
DTYPE="${DTYPE:-float32}"      # float32 is fine for NVT; use float64 for NVE drift tests
TRAJ_INT="${TRAJ_INT:-400}"    # steps between saved frames (400 x 0.5 fs = 0.2 ps)
NGPU="${NGPU:-1}"              # number of parallel jobs / GPUs
EXTRA_ARGS="${EXTRA_ARGS:-}"   # extra flags for mace_md.py run, e.g. --freeze-below 8

# Optional: after each run, evaluate your model and a foundation model on the same
# trajectory frames (see compare_models.py). Off unless COMPARE_WITH is set:
#   COMPARE_WITH=medium-mpa-0 COMPARE_D3=1 bash run_md_validation.sh
COMPARE_SCRIPT="/home/h.jack/MACE_train/MD-val/compare_models.py"
COMPARE_WITH="/home/h.jack/MACE_train/mace_active_learning/mace-mpa-0-medium.model"
COMPARE_D3=1
COMPARE_FRAMES="${COMPARE_FRAMES:-200}"

# name  |  description
STRUCTURES=(
  #"bulkwater"            # bulk water
  #"graphfragptatom"      # graphene + 20-atom Pt cluster + Nafion fragments in solution
  "Graphvac"             # 2-layer graphene, large vacancy in top layer
  "naf-frag"             # several Nafion fragments in water
  "Ptclose"              # Pt(111) in water, Nafion SO3 touching the surface
  "ptclustergraph"       # 20-atom Pt cluster on graphene, vacuum
  #"ptclustergraphsol"    # same, in water
  "Ptnaf"                # Nafion 7 A from a solvated Pt(111) cell
  "ptparticlegra"        # Pt particle on graphene, vacuum
  #"ptparticlegraphsol"   # same, in water
  #"Ptwater"              # Pt(111) with 20 A of water
)

# Short NVE energy-conservation checks (float64, no stress needed)
NVE_TESTS=("bulkwater" "Ptwater")
NVE_PROD_PS="${NVE_PROD_PS:-10}"

# ------------------------------------------------------------------ helpers
if [ "$#" -gt 0 ]; then STRUCTURES=("$@"); NVE_TESTS=(); fi

EQUIL_STEPS=$(python -c "print(int(round($EQUIL_PS*1000/$DT)))")
TOTAL_STEPS=$(python -c "print(int(round(($EQUIL_PS+$PROD_PS)*1000/$DT)))")
RDF_START=$(( EQUIL_STEPS / TRAJ_INT + 1 ))   # first production frame in traj.extxyz

run_one() {   # name temperature
  local name="$1" T="$2" gpu="$3"
  local cif="$STRUCT_DIR/$name.cif"
  local out="$OUT_ROOT/${name}_${T}K"
  [ -f "$cif" ] || { echo "[skip] missing $cif"; return 1; }
  mkdir -p "$out"

  if grep -q '"status": "completed"' "$out/summary.json" 2>/dev/null; then
    echo "[done] $name ${T}K already completed"
  else
    local args=(--model "$MACE_MODEL" --head "$HEAD" --dtype "$DTYPE" --temperature "$T"
                --timestep "$DT" --traj-interval "$TRAJ_INT" --outdir "$out" --save-events)
    local mode=(--structure "$cif" --equil-ps "$EQUIL_PS" --prod-ps "$PROD_PS")

    if [ -f "$out/restart.extxyz" ]; then
      local step
      step=$(python -c "
from ase.io import read
print(int(read('$out/restart.extxyz', index=-1).info.get('md_step', 0)))")
      if [ "$step" -gt "$EQUIL_STEPS" ] && [ "$step" -lt "$TOTAL_STEPS" ]; then
        local rem
        rem=$(python -c "print(($TOTAL_STEPS-$step)*$DT/1000)")
        echo "[resume] $name ${T}K from step $step ($rem ps left)"
        [ -f "$out/rdf.dat" ] && cp "$out/rdf.dat" "$out/rdf_before_step${step}.dat"
        mode=(--restart "$out/restart.extxyz" --equil-ps 0 --prod-ps "$rem")
      else
        echo "[restart-from-scratch] $name ${T}K (interrupted during equilibration)"
        rm -f "$out/traj.extxyz" "$out/thermo.csv" "$out/restart.extxyz" "$out/rdf.dat"
      fi
    fi

    echo "[run] $name ${T}K on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python "$VALIDATE_SCRIPT" run "${mode[@]}" "${args[@]}" $EXTRA_ARGS \
      >> "$out/stdout.txt" 2>&1
    echo "[exit $?] $name ${T}K"
  fi

  # RDF / z-profile from the production part of the trajectory.
  # --ref = minimised starting structure, so groups match the run and an AIMD comparison.
  if [ -f "$out/traj.extxyz" ]; then
    local ref=(); [ -f "$out/initial_after_min.extxyz" ] && ref=(--ref "$out/initial_after_min.extxyz")
    python "$VALIDATE_SCRIPT" rdf --traj "$out/traj.extxyz" --start "$RDF_START" \
      ${ref[@]+"${ref[@]}"} --out "$out/rdf_check" >> "$out/stdout.txt" 2>&1 \
      || echo "[warn] rdf analysis failed for $name ${T}K (run too short?)"
  fi

  # Optional model-vs-foundation comparison on identical frames
  if [ -n "$COMPARE_WITH" ] && [ -f "$out/traj.extxyz" ]; then
    local d3=(); [ -n "$COMPARE_D3" ] && d3=(--d3)
    local cref=(); [ -f "$out/initial_after_min.extxyz" ] && cref=(--ref "$out/initial_after_min.extxyz")
    CUDA_VISIBLE_DEVICES="$gpu" python "$COMPARE_SCRIPT" --traj "$out/traj.extxyz" --start "$RDF_START" \
      --max-frames "$COMPARE_FRAMES" --model-a "$MACE_MODEL" --head "$HEAD" --foundation "$COMPARE_WITH" \
      ${d3[@]+"${d3[@]}"} ${cref[@]+"${cref[@]}"} --out "$out/compare_$COMPARE_WITH" >> "$out/stdout.txt" 2>&1 \
      || echo "[warn] model comparison failed for $name ${T}K (see $out/stdout.txt)"
  fi
}

run_nve() {   # name gpu
  local name="$1" gpu="$2"
  local out="$OUT_ROOT/${name}_NVEtest"
  [ -f "$out/summary.json" ] && { echo "[done] $name NVE test"; return; }
  mkdir -p "$out"
  echo "[nve] $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python "$VALIDATE_SCRIPT" run --structure "$STRUCT_DIR/$name.cif" \
    --model "$MACE_MODEL" --head "$HEAD" --dtype float64 --ensemble nve --temperature 300 \
    --timestep "$DT" --equil-ps 5 --prod-ps "$NVE_PROD_PS" --outdir "$out" $EXTRA_ARGS \
    >> "$out/stdout.txt" 2>&1
  echo "[exit $?] $name NVE test  (check dEtot/at in md.log or Etot_eV in thermo.csv)"
}

# Simple job pool: at most NGPU jobs at once, job i uses GPU (i mod NGPU)
i=0
launch() {
  if [ "$NGPU" -gt 1 ]; then
    while [ "$(jobs -rp | wc -l)" -ge "$NGPU" ]; do wait -n; done
    "$@" $(( i % NGPU )) &
  else
    "$@" 0
  fi
  i=$(( i + 1 ))
}

mkdir -p "$OUT_ROOT"
for T in $TEMPS; do
  for name in "${STRUCTURES[@]}"; do
    launch run_one "$name" "$T"
  done
done
for name in "${NVE_TESTS[@]}"; do
  launch run_nve "$name"
done
wait
echo "All done. Results in $OUT_ROOT/<name>_<T>K/ (md.log, thermo.csv, rdf.dat, rdf_check_rdf.dat, zprofile.dat, traj.extxyz)"