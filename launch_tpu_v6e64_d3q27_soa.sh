#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-YOUR_GCP_PROJECT}"
ZONE="${ZONE:-YOUR_TPU_ZONE}"
TPU_NAME="${TPU_NAME:-YOUR_TPU_POD_NAME}"
WORKDIR="${WORKDIR:-~/skanda_simulation}"

GLOBAL_X="${GLOBAL_X:-1024}"
Y="${Y:-1024}"
Z="${Z:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
TIMED_STEPS="${TIMED_STEPS:-10}"
SCAN_CHUNK_STEPS="${SCAN_CHUNK_STEPS:-1}"
COLLISION_MODE="${COLLISION_MODE:-voxel_vmap}"
OUT="${OUT:-outputs/google_infra/d3q27_soa_boilerplate_vmap_1024x1024x1000_10step.json}"

gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  --worker=all \
  --batch-size=16 \
  --command="cd ${WORKDIR} && python3 -u d3q27_soa_boilerplate.py \
    --distributed-init \
    --global-x ${GLOBAL_X} \
    --y ${Y} \
    --z ${Z} \
    --warmup-steps ${WARMUP_STEPS} \
    --timed-steps ${TIMED_STEPS} \
    --scan-chunk-steps ${SCAN_CHUNK_STEPS} \
    --collision-mode ${COLLISION_MODE} \
    --output ${OUT}"
