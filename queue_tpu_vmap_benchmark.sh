#!/usr/bin/env bash
set -euo pipefail

# Polite queue wrapper: never kills an existing TPU job. It waits until the
# known production training / D3Q27 benchmark processes are gone, then stages
# this boilerplate and launches the vmap/SRAM-fusion benchmark.

PROJECT="${PROJECT:-stunning-grin-493612-t4}"
ZONE="${ZONE:-asia-south1-b}"
TPU_NAME="${TPU_NAME:-shodh-ai-pod}"
WORKDIR="${WORKDIR:-~/skanda_simulation}"
POLL_SECONDS="${POLL_SECONDS:-300}"
COLLISION_CORE="${COLLISION_CORE:-/Users/apple/Desktop/jataka/skanda_simulation/shifted_cumulant_lbm.py}"

busy_regex="${BUSY_REGEX:-train_pido_xla.py|d3q27_soa_boilerplate.py}"

while true; do
  if gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --worker=all \
    --batch-size=16 \
    --command="pgrep -af '${busy_regex}' >/dev/null"; then
    echo "TPU pod is busy; sleeping ${POLL_SECONDS}s."
    sleep "${POLL_SECONDS}"
  else
    echo "TPU pod appears free; staging benchmark files."
    break
  fi
done

gcloud alpha compute tpus tpu-vm scp --worker=all \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  d3q27_soa_boilerplate.py "${COLLISION_CORE}" \
  "${TPU_NAME}:${WORKDIR}/"

COLLISION_MODE="${COLLISION_MODE:-voxel_vmap}" \
OUT="${OUT:-outputs/google_infra/d3q27_soa_boilerplate_vmap_1024x1024x1000_10step.json}" \
PROJECT="${PROJECT}" \
ZONE="${ZONE}" \
TPU_NAME="${TPU_NAME}" \
WORKDIR="${WORKDIR}" \
./launch_tpu_v6e64_d3q27_soa.sh
