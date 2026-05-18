#!/usr/bin/env bash
set -euo pipefail

# Retry only the failed/pending combinations from the few-channel full run.
# Run this after syncing code fixes and after the previous master launcher has
# stopped writing to runs_all_datasets_few_channels.
#
# Usage:
#   cd /data/mzy/LibEER
#   bash scripts/run_failed_few_channel_retries.sh
#
# Optional:
#   RETRY_GROUPS="RGNN FBSTCNET" bash scripts/run_failed_few_channel_retries.sh
#   FACED_DBN_BATCH_SIZE=4 bash scripts/run_failed_few_channel_retries.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_all_datasets_few_channel_networks.sh"

CACHE_ROOT="${CACHE_ROOT:-/data/mzy/libeer_few_channel_cache}"
SEEDS="${SEEDS:-2024}"
RETRY_GROUPS="${RETRY_GROUPS:-RGNN FBSTCNET SEEDV_FEATURE SEEDV_RAW FACED_NSAL FACED_DBN FACED_MSMDA}"

run_retry() {
  local name="$1"
  local datasets="$2"
  local channels="$3"
  local networks="$4"
  local gpus="$5"
  local batch_size="${6:-${BATCH_SIZE:-32}}"

  echo "[$(date '+%F %T')] RETRY group=$name datasets=$datasets channels=$channels networks=$networks gpus=$gpus batch_size=$batch_size"
  CACHE_ROOT="$CACHE_ROOT" \
  DATASETS="$datasets" \
  CHANNELS="$channels" \
  NETWORKS="$networks" \
  GPUS="$gpus" \
  MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}" \
  SEEDS="$SEEDS" \
  BATCH_SIZE="$batch_size" \
  bash "$RUNNER"
}

for group in $RETRY_GROUPS; do
  case "$group" in
    RGNN)
      run_retry "RGNN" "SEED SEEDV FACED" "2ch 4ch 8ch" "RGNN_official" "${RGNN_GPUS:-0 1 2}"
      ;;
    FBSTCNET)
      run_retry "FBSTCNET" "SEED SEEDV" "2ch 4ch 8ch" "FBSTCNet" "${FBSTCNET_GPUS:-0 1}"
      ;;
    SEEDV_FEATURE)
      run_retry "SEEDV_FEATURE_2CH" "SEEDV" "2ch" "DGCNN GCBNet HSLT" "${SEEDV_FEATURE_GPUS:-0 1 2}"
      run_retry "SEEDV_FEATURE_8CH" "SEEDV" "8ch" "HSLT" "${SEEDV_FEATURE_GPUS:-0}"
      ;;
    SEEDV_RAW)
      run_retry "SEEDV_RAW" "SEEDV" "2ch 4ch 8ch" "TSception ACRNN" "${SEEDV_RAW_GPUS:-0 1}"
      ;;
    FACED_NSAL)
      run_retry "FACED_NSAL" "FACED" "2ch 4ch 8ch" "NSAL_DGAT" "${FACED_NSAL_GPUS:-0 1 2}"
      ;;
    FACED_DBN)
      run_retry "FACED_DBN" "FACED" "2ch 4ch 8ch" "DBN" "${FACED_DBN_GPUS:-0}" "${FACED_DBN_BATCH_SIZE:-8}"
      ;;
    FACED_MSMDA)
      run_retry "FACED_MSMDA" "FACED" "8ch" "MsMda" "${FACED_MSMDA_GPUS:-0}"
      ;;
    *)
      echo "Unknown retry group: $group" >&2
      echo "Valid groups: RGNN FBSTCNET SEEDV_FEATURE SEEDV_RAW FACED_NSAL FACED_DBN FACED_MSMDA" >&2
      exit 1
      ;;
  esac
done

echo "[$(date '+%F %T')] Retry groups finished."
echo "Check status with: python scripts/check_few_channel_run_status.py --show-failed 100 --show-pending 100"
