#!/usr/bin/env bash
set -euo pipefail

# Generate LibEER few-channel caches on the server.
#
# Defaults use the server dataset locations:
#   SEED   -> /data/mzy/SEED/
#   SEED-V -> /data/mzy/EEG_EMOTION_DATA/SEED-V/
#   FACED  -> /data/mzy/DDSI/EEG_emotion_data/FACED/
#
# Example:
#   cd /data/mzy/LibEER
#   bash scripts/preprocess_all_few_channel_caches.sh
#
# Useful overrides:
#   CACHE_ROOT=/data/mzy/libeer_few_channel_cache
#   DATASETS="SEED SEEDV FACED"
#   CHANNELS="2ch 4ch 8ch"
#   OVERWRITE=1
#   DRY_RUN=1
#   PYTHON=python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PREPROCESS_SCRIPT="$SCRIPT_DIR/preprocess_few_channel_cache.py"

PYTHON="${PYTHON:-python}"
CACHE_ROOT="${CACHE_ROOT:-/data/mzy/libeer_few_channel_cache}"
DATASETS="${DATASETS:-SEED SEEDV FACED}"
CHANNELS="${CHANNELS:-2ch 4ch 8ch}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

SEED_SOURCE_PATH="${SEED_SOURCE_PATH:-/data/mzy/SEED/}"
SEED_DATASET="${SEED_DATASET:-seed_de_lds}"
SEED_SETTING="${SEED_SETTING:-seed_sub_dependent_train_val_test_setting}"
SEED_EXTRA_ARGS="${SEED_EXTRA_ARGS:-}"

SEEDV_SOURCE_PATH="${SEEDV_SOURCE_PATH:-/data/mzy/EEG_EMOTION_DATA/SEED-V/}"
SEEDV_DATASET="${SEEDV_DATASET:-seedv_raw}"
SEEDV_SETTING="${SEEDV_SETTING:-seedv_sub_dependent_train_val_test_setting}"
SEEDV_8CH_FALLBACK="${SEEDV_8CH_FALLBACK:-FC3_FC4}"
SEEDV_EXTRA_ARGS="${SEEDV_EXTRA_ARGS:-}"

FACED_SOURCE_PATH="${FACED_SOURCE_PATH:-/data/mzy/DDSI/EEG_emotion_data/FACED/}"
FACED_DATASET="${FACED_DATASET:-faced_de_lds}"
FACED_SETTING="${FACED_SETTING:-faced_sub_independent_train_val_test_setting}"
FACED_EXTRA_ARGS="${FACED_EXTRA_ARGS:-}"

COMMON_EXTRA_ARGS="${COMMON_EXTRA_ARGS:-}"

read -r -a DATASET_LIST <<< "$DATASETS"
read -r -a CHANNEL_LIST <<< "$CHANNELS"
read -r -a COMMON_EXTRA_ARG_LIST <<< "$COMMON_EXTRA_ARGS"

if [[ ! -f "$PREPROCESS_SCRIPT" ]]; then
  echo "Preprocess script not found: $PREPROCESS_SCRIPT" >&2
  exit 1
fi

overwrite_args=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "DRY_RUN:"
    printf " %q" "$@"
    printf "\n"
  else
    "$@"
  fi
}

run_dataset() {
  local key="$1"
  local dataset="$2"
  local source_path="$3"
  local setting="$4"
  local output_root="$CACHE_ROOT/$key"
  local dataset_extra_args="$5"
  local -a dataset_extra_arg_list=()

  if [[ -n "$dataset_extra_args" ]]; then
    read -r -a dataset_extra_arg_list <<< "$dataset_extra_args"
  fi

  local -a cmd=(
    "$PYTHON" "$PREPROCESS_SCRIPT"
    -dataset "$dataset"
    -dataset_path "$source_path"
    -setting "$setting"
    --channels "${CHANNEL_LIST[@]}"
    --output_root "$output_root"
    "${overwrite_args[@]}"
    "${COMMON_EXTRA_ARG_LIST[@]}"
    "${dataset_extra_arg_list[@]}"
  )

  if [[ "$key" == "SEEDV" ]]; then
    cmd+=(--seedv_8ch_fallback "$SEEDV_8CH_FALLBACK")
  fi

  echo "[$(date '+%F %T')] Preprocess $key dataset=$dataset source=$source_path output=$output_root"
  run_cmd "${cmd[@]}"
}

cd "$PROJECT_ROOT"

for dataset_key in "${DATASET_LIST[@]}"; do
  case "$dataset_key" in
    SEED)
      run_dataset "SEED" "$SEED_DATASET" "$SEED_SOURCE_PATH" "$SEED_SETTING" "$SEED_EXTRA_ARGS"
      ;;
    SEEDV)
      run_dataset "SEEDV" "$SEEDV_DATASET" "$SEEDV_SOURCE_PATH" "$SEEDV_SETTING" "$SEEDV_EXTRA_ARGS"
      ;;
    FACED)
      run_dataset "FACED" "$FACED_DATASET" "$FACED_SOURCE_PATH" "$FACED_SETTING" "$FACED_EXTRA_ARGS"
      ;;
    *)
      echo "Unsupported DATASETS entry '$dataset_key'. Use SEED, SEEDV, FACED." >&2
      exit 1
      ;;
  esac
done

cat <<EOF
Finished. Cache paths are under:
  $CACHE_ROOT/SEED/$SEED_DATASET/$SEED_SETTING/{2ch,4ch,8ch}/libeer_cache.pkl
  $CACHE_ROOT/SEEDV/$SEEDV_DATASET/$SEEDV_SETTING/{2ch,4ch,8ch}/libeer_cache.pkl
  $CACHE_ROOT/FACED/$FACED_DATASET/$FACED_SETTING/{2ch,4ch,8ch}/libeer_cache.pkl
EOF
