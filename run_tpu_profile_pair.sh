#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-stunning-grin-493612-t4}"
ZONE="${ZONE:-asia-south1-b}"
TPU_NAME="${TPU_NAME:-shodh-ai-pod}"
WORKDIR="${WORKDIR:-/home/apple/skanda_simulation}"

GLOBAL_X="${GLOBAL_X:-512}"
Y="${Y:-512}"
Z="${Z:-500}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
TIMED_STEPS="${TIMED_STEPS:-3}"
SCAN_CHUNK_STEPS="${SCAN_CHUNK_STEPS:-1}"
PROFILE_ROOT="${PROFILE_ROOT:-/home/apple/skanda_simulation/profiles/google_infra}"
OUT_ROOT="${OUT_ROOT:-outputs/google_infra}"

run_one() {
  local mode="$1"
  local name="profile_${mode}_${GLOBAL_X}x${Y}x${Z}_${TIMED_STEPS}step"
  gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --worker=all \
    --batch-size=16 \
    --command="cd ${WORKDIR} && mkdir -p logs ${OUT_ROOT} ${PROFILE_ROOT}/${name} && python3 -u d3q27_soa_boilerplate.py \
      --distributed-init \
      --global-x ${GLOBAL_X} \
      --y ${Y} \
      --z ${Z} \
      --warmup-steps ${WARMUP_STEPS} \
      --timed-steps ${TIMED_STEPS} \
      --scan-chunk-steps ${SCAN_CHUNK_STEPS} \
      --collision-mode ${mode} \
      --profile-dir ${PROFILE_ROOT}/${name} \
      --profile-name ${name} \
      --output ${OUT_ROOT}/${name}.json > logs/${name}_\$(hostname).log 2>&1"
}

run_one local_aos
run_one voxel_vmap
