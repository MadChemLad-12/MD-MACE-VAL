#!/bin/bash

# 1. Define standard string of keywords separated by spaces or use a native Bash array
KEYWORDS=("ptclustergraph" "ptclustergraphsol" "Ptnaf" "ptparticlegra")

# 2. Correct Bash function syntax (no parameters in the parentheses)
run_MD_plots() {
    local keyword="$1" # Read the first argument passed to the function

    # Note: Backslashes must be the absolute LAST character on a line (no spaces after them)
    python mace_md.py rdf --traj runs/${keyword}_300K/traj.extxyz \
    --ref runs/${keyword}_300K/initial_after_min.extxyz \
    --start 101 --stop 601 --out runs/${keyword}_300K/${keyword}1 \
    && python mace_md.py rdf --traj runs/${keyword}_300K/traj.extxyz \
    --ref runs/${keyword}_300K/initial_after_min.extxyz \
    --start 601 --stop 1101 --out runs/${keyword}_300K/${keyword}2 \
    && python mace_md.py rdf --traj runs/${keyword}_300K/traj.extxyz \
    --ref runs/${keyword}_300K/initial_after_min.extxyz \
    --start 1101 --out runs/${keyword}_300K/${keyword}3 \
    && python rdf_summary.py b1=runs/${keyword}_300K/${keyword}1_rdf.dat b2=runs/${keyword}_300K/${keyword}2_rdf.dat b3=runs/${keyword}_300K/${keyword}3_rdf.dat \
    && cp rdf_summary.png rdf_summary_${keyword}.png > runs/${keyword}_300K/md-rdf.log
}

# 3. Correct loop to iterate through the Bash array
for keyword in "${KEYWORDS[@]}"; do
  echo "Running MD for $keyword"
  # Call the function by passing the variable as a standard terminal argument
  run_MD_plots "$keyword"
done
