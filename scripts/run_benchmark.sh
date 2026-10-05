#!/usr/bin/env bash
# Regenerates every LiDAR-tier benchmark number from the raw captures.
# Usage: scripts/run_benchmark.sh <data_dir> <out_dir>
#   data_dir holds single_room/, single_scan_floor_only/, single_scan_with_ceiling/
set -euo pipefail
DATA=${1:-..}
OUT=${2:-out/bench}
PY=${PY:-.venv/bin/python}
mkdir -p "$OUT"
for cap in single_room single_scan_floor_only single_scan_with_ceiling; do
  $PY -m scan2plan run lidar "$DATA/$cap" --out "$OUT/$cap" | tee "$OUT/$cap.log"
done
# Repeatability: the two whole-apartment captures.
$PY scripts/repeatability.py "$OUT/single_scan_floor_only/plan.json" \
    "$OUT/single_scan_with_ceiling/plan.json" "$OUT/repeatability.json"
# Drift ablation: same captures with raw ARKit poses.
for cap in single_scan_floor_only single_scan_with_ceiling; do
  $PY -m scan2plan run lidar "$DATA/$cap" --out "$OUT/${cap}_drift_off" --no-drift --no-damage | tee "$OUT/${cap}_drift_off.log"
done
$PY scripts/ablation.py "$OUT" single_scan_floor_only single_scan_with_ceiling
# Schema check on every plan produced.
$PY - "$OUT" <<'PY'
import json, sys, glob, jsonschema
schema = json.load(open("schema/plan.schema.json"))
for p in sorted(glob.glob(f"{sys.argv[1]}/*/plan.json")):
    jsonschema.validate(json.load(open(p)), schema)
    print("schema ok:", p)
PY
