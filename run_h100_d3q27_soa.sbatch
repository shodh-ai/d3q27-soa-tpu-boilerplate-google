#!/usr/bin/env bash
#SBATCH --job-name=d3q27-soa-google-boilerplate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=logs/d3q27_soa_%j.out
#SBATCH --error=logs/d3q27_soa_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs/google_infra

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0"

python -u deliverables/google_infra/d3q27_soa_boilerplate.py \
  --global-x "${GLOBAL_X:-512}" \
  --y "${Y:-512}" \
  --z "${Z:-512}" \
  --warmup-steps "${WARMUP_STEPS:-1}" \
  --timed-steps "${TIMED_STEPS:-10}" \
  --scan-chunk-steps "${SCAN_CHUNK_STEPS:-1}" \
  --output "${OUT:-outputs/google_infra/d3q27_soa_boilerplate_h100_${SLURM_JOB_ID}.json}"
