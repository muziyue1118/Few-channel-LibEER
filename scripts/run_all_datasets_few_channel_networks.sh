#!/usr/bin/env bash
set -u

# Master launcher for:
#   SEED / SEEDIV / SEEDV / FACED x 2ch / 4ch / 8ch x networks x seeds x multi-GPU queue
#
# This script schedules one atomic job per dataset/channel/network/seed.  Each
# atomic job calls scripts/run_all_few_channel_networks.sh with a single
# DATASET, CHANNELS, NETWORKS, SEEDS, and GPU value.
#
# Example usage:
#   cd /data/mzy/LibEER
#
#   export GPUS="0 1 2 3"
#   export MAX_JOBS_PER_GPU=1
#   export SEEDS="2024"
#   export EPOCHS=80
#   export BATCH_SIZE=32
#   export LR=0.001
#
#   export SEED_PATH_2CH=/data/mzy/SEED/ExtractedFeatures/ext_fea/fea_r512/2ch
#   export SEED_PATH_4CH=/data/mzy/SEED/ExtractedFeatures/ext_fea/fea_r512/4ch
#   export SEED_PATH_8CH=/data/mzy/SEED/ExtractedFeatures/ext_fea/fea_r512/8ch
#   export SEEDIV_PATH_2CH=/data/mzy/EEG_EMOTION_DATA/SEED-IV/SEED_IV/eeg_feature_smooth/2ch
#   export SEEDIV_PATH_4CH=/data/mzy/EEG_EMOTION_DATA/SEED-IV/SEED_IV/eeg_feature_smooth/4ch
#   export SEEDIV_PATH_8CH=/data/mzy/EEG_EMOTION_DATA/SEED-IV/SEED_IV/eeg_feature_smooth/8ch
#   export SEEDV_PATH_2CH=/data/mzy/EEG_EMOTION_DATA/SEED-V/EEG_DE_features/2ch
#   export SEEDV_PATH_4CH=/data/mzy/EEG_EMOTION_DATA/SEED-V/EEG_DE_features/4ch
#   export SEEDV_PATH_8CH=/data/mzy/EEG_EMOTION_DATA/SEED-V/EEG_DE_features/8ch
#   export FACED_PATH_2CH=/data/mzy/DDSI/EEG_emotion_data/FACED/EEG_Features/2ch
#   export FACED_PATH_4CH=/data/mzy/DDSI/EEG_emotion_data/FACED/EEG_Features/4ch
#   export FACED_PATH_8CH=/data/mzy/DDSI/EEG_emotion_data/FACED/EEG_Features/8ch
#
#   bash scripts/run_all_datasets_few_channel_networks.sh
#
# Required dataset path variables. Set the ones for datasets/channels you run:
#   SEED_PATH_2CH=/path/to/seed_2ch
#   SEED_PATH_4CH=/path/to/seed_4ch
#   SEED_PATH_8CH=/path/to/seed_8ch
#   SEEDIV_PATH_2CH=/path/to/seediv_2ch
#   SEEDIV_PATH_4CH=/path/to/seediv_4ch
#   SEEDIV_PATH_8CH=/path/to/seediv_8ch
#   SEEDV_PATH_2CH=/path/to/seedv_2ch
#   SEEDV_PATH_4CH=/path/to/seedv_4ch
#   SEEDV_PATH_8CH=/path/to/seedv_8ch
#   FACED_PATH_2CH=/path/to/faced_2ch
#   FACED_PATH_4CH=/path/to/faced_4ch
#   FACED_PATH_8CH=/path/to/faced_8ch
#
# Common controls:
#   DATASETS="SEED SEEDIV SEEDV FACED"
#   CHANNELS="2ch 4ch 8ch"
#   GPUS="0 1 2 3"
#   MAX_JOBS_PER_GPU=1
#   SEEDS="2024"
#   NETWORKS="DGCNN GCBNet ..."
#   EPOCHS=80
#   BATCH_SIZE=32
#   LR=0.001
#   EXTRA_ARGS="-time_window 1 -feature_type de_lds"
#   DRY_RUN=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKER_SCRIPT="$SCRIPT_DIR/run_all_few_channel_networks.sh"

DATASETS="${DATASETS:-SEED SEEDIV SEEDV FACED}"
CHANNELS="${CHANNELS:-2ch 4ch 8ch}"
GPUS="${GPUS:-0}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
SEEDS="${SEEDS:-2024}"

DEFAULT_NETWORKS="DGCNN CoralDgcnn GCBNet GCBNet_BLS CDCN DBN HSLT BiDANN R2GSTNN NSAL_DGAT EEGNet TSception ACRNN RGNN_official MsMda FBSTCNet PRRL"
NETWORKS="${NETWORKS:-$DEFAULT_NETWORKS}"

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
DEVICE="${DEVICE:-cuda}"
METRICS="${METRICS:-acc macro-f1}"
METRIC_CHOOSE="${METRIC_CHOOSE:-macro-f1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs_all_datasets_few_channels}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
KILL_ON_FAIL="${KILL_ON_FAIL:-0}"
DRY_RUN="${DRY_RUN:-0}"
QUEUE_SLEEP_SECONDS="${QUEUE_SLEEP_SECONDS:-10}"

mkdir -p "$RUN_ROOT/launcher_logs" "$RUN_ROOT/job_status" "$RUN_ROOT/worker_runs"
MASTER_SUMMARY="$RUN_ROOT/master_summary.tsv"
if [[ ! -f "$MASTER_SUMMARY" ]]; then
  printf "time\tdataset_key\tdataset\tchannel\tseed\tnetwork\tgpu\tstatus\texit_code\tseconds\tlauncher_log\tworker_run_root\n" > "$MASTER_SUMMARY"
fi

read -r -a DATASET_LIST <<< "$DATASETS"
read -r -a CHANNEL_LIST <<< "$CHANNELS"
read -r -a GPU_LIST <<< "$GPUS"
read -r -a SEED_LIST <<< "$SEEDS"
read -r -a NETWORK_LIST <<< "$NETWORKS"

if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "GPUS must contain at least one GPU id." >&2
  exit 1
fi

if [[ "$MAX_JOBS_PER_GPU" -lt 1 ]]; then
  echo "MAX_JOBS_PER_GPU must be >= 1." >&2
  exit 1
fi

if [[ ! -f "$WORKER_SCRIPT" ]]; then
  echo "Worker script not found: $WORKER_SCRIPT" >&2
  exit 1
fi

dataset_name_for_key() {
  case "$1" in
    SEED) printf "%s" "${SEED_DATASET:-seed_de_lds}" ;;
    SEEDIV) printf "%s" "${SEEDIV_DATASET:-seediv_de_lds}" ;;
    SEEDV) printf "%s" "${SEEDV_DATASET:-seedv_raw}" ;;
    FACED) printf "%s" "${FACED_DATASET:-faced_de_lds}" ;;
    *)
      echo "Unsupported dataset key '$1'. Use SEED, SEEDIV, SEEDV, FACED." >&2
      return 1
      ;;
  esac
}

setting_for_key() {
  case "$1" in
    SEED) printf "%s" "${SEED_SETTING:-seed_sub_dependent_train_val_test_setting}" ;;
    SEEDIV) printf "%s" "${SEEDIV_SETTING:-seediv_sub_dependent_train_val_test_setting}" ;;
    SEEDV) printf "%s" "${SEEDV_SETTING:-seedv_sub_dependent_train_val_test_setting}" ;;
    FACED) printf "%s" "${FACED_SETTING:-faced_sub_independent_train_val_test_setting}" ;;
    *)
      echo "Unsupported dataset key '$1'. Use SEED, SEEDIV, SEEDV, FACED." >&2
      return 1
      ;;
  esac
}

extra_args_for_key() {
  case "$1" in
    SEED) printf "%s" "${SEED_EXTRA_ARGS:-$EXTRA_ARGS}" ;;
    SEEDIV) printf "%s" "${SEEDIV_EXTRA_ARGS:-$EXTRA_ARGS}" ;;
    SEEDV) printf "%s" "${SEEDV_EXTRA_ARGS:-$EXTRA_ARGS}" ;;
    FACED) printf "%s" "${FACED_EXTRA_ARGS:-$EXTRA_ARGS}" ;;
    *) printf "%s" "$EXTRA_ARGS" ;;
  esac
}

path_for_key_channel() {
  local dataset_key="$1"
  local channel="$2"
  case "$dataset_key:$channel" in
    SEED:2ch) printf "%s" "${SEED_PATH_2CH:-}" ;;
    SEED:4ch) printf "%s" "${SEED_PATH_4CH:-}" ;;
    SEED:8ch) printf "%s" "${SEED_PATH_8CH:-}" ;;
    SEEDIV:2ch) printf "%s" "${SEEDIV_PATH_2CH:-}" ;;
    SEEDIV:4ch) printf "%s" "${SEEDIV_PATH_4CH:-}" ;;
    SEEDIV:8ch) printf "%s" "${SEEDIV_PATH_8CH:-}" ;;
    SEEDV:2ch) printf "%s" "${SEEDV_PATH_2CH:-}" ;;
    SEEDV:4ch) printf "%s" "${SEEDV_PATH_4CH:-}" ;;
    SEEDV:8ch) printf "%s" "${SEEDV_PATH_8CH:-}" ;;
    FACED:2ch) printf "%s" "${FACED_PATH_2CH:-}" ;;
    FACED:4ch) printf "%s" "${FACED_PATH_4CH:-}" ;;
    FACED:8ch) printf "%s" "${FACED_PATH_8CH:-}" ;;
    *)
      echo "Unsupported dataset/channel pair '$dataset_key/$channel'." >&2
      return 1
      ;;
  esac
}

path_var_name_for_key_channel() {
  local dataset_key="$1"
  local channel="$2"
  case "$dataset_key:$channel" in
    SEED:2ch) echo "SEED_PATH_2CH" ;;
    SEED:4ch) echo "SEED_PATH_4CH" ;;
    SEED:8ch) echo "SEED_PATH_8CH" ;;
    SEEDIV:2ch) echo "SEEDIV_PATH_2CH" ;;
    SEEDIV:4ch) echo "SEEDIV_PATH_4CH" ;;
    SEEDIV:8ch) echo "SEEDIV_PATH_8CH" ;;
    SEEDV:2ch) echo "SEEDV_PATH_2CH" ;;
    SEEDV:4ch) echo "SEEDV_PATH_4CH" ;;
    SEEDV:8ch) echo "SEEDV_PATH_8CH" ;;
    FACED:2ch) echo "FACED_PATH_2CH" ;;
    FACED:4ch) echo "FACED_PATH_4CH" ;;
    FACED:8ch) echo "FACED_PATH_8CH" ;;
  esac
}

declare -A GPU_RUNNING=()
declare -A PID_GPU=()
declare -A PID_INFO=()
declare -A PID_STATUS_FILE=()
declare -A PID_LOG_FILE=()
declare -A PID_START_SECONDS=()
declare -A PID_WORKER_ROOT=()
PIDS=()
STOP_REQUESTED=0

for gpu in "${GPU_LIST[@]}"; do
  GPU_RUNNING["$gpu"]=0
done

select_gpu() {
  local gpu
  for gpu in "${GPU_LIST[@]}"; do
    if [[ "${GPU_RUNNING[$gpu]}" -lt "$MAX_JOBS_PER_GPU" ]]; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

reap_finished_jobs() {
  local remaining=()
  local pid gpu info status_file log_file start_seconds end_seconds elapsed exit_code status worker_root
  local dataset_key dataset channel seed network

  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      remaining+=("$pid")
      continue
    fi

    wait "$pid" 2>/dev/null || true
    gpu="${PID_GPU[$pid]}"
    info="${PID_INFO[$pid]}"
    status_file="${PID_STATUS_FILE[$pid]}"
    log_file="${PID_LOG_FILE[$pid]}"
    start_seconds="${PID_START_SECONDS[$pid]}"
    worker_root="${PID_WORKER_ROOT[$pid]}"
    end_seconds="$(date +%s)"
    elapsed="$((end_seconds - start_seconds))"

    if [[ -f "$status_file" ]]; then
      exit_code="$(cat "$status_file")"
    else
      exit_code="999"
    fi

    if [[ "$exit_code" == "0" ]]; then
      status="ok"
    else
      status="failed"
      if [[ "$STOP_ON_FAIL" == "1" ]]; then
        STOP_REQUESTED=1
      fi
    fi

    IFS="|" read -r dataset_key dataset channel seed network <<< "$info"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$(date '+%F %T')" "$dataset_key" "$dataset" "$channel" "$seed" "$network" "$gpu" \
      "$status" "$exit_code" "$elapsed" "$log_file" "$worker_root" >> "$MASTER_SUMMARY"

    GPU_RUNNING["$gpu"]="$((GPU_RUNNING[$gpu] - 1))"
    echo "[$(date '+%F %T')] FINISH dataset=$dataset_key channel=$channel seed=$seed network=$network gpu=$gpu status=$status"

    unset "PID_GPU[$pid]" "PID_INFO[$pid]" "PID_STATUS_FILE[$pid]" "PID_LOG_FILE[$pid]" \
      "PID_START_SECONDS[$pid]" "PID_WORKER_ROOT[$pid]"
  done

  PIDS=("${remaining[@]}")
}

kill_running_jobs() {
  local pid
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

launch_job() {
  local dataset_key="$1"
  local dataset="$2"
  local setting="$3"
  local channel="$4"
  local dataset_path="$5"
  local seed="$6"
  local network="$7"
  local gpu="$8"
  local dataset_extra_args="$9"

  local job_id="${dataset_key}_${channel}_${network}_seed${seed}_gpu${gpu}_$(date +%s%N)"
  local status_file="$RUN_ROOT/job_status/${job_id}.status"
  local launcher_log="$RUN_ROOT/launcher_logs/${job_id}.log"
  local worker_root="$RUN_ROOT/worker_runs/${dataset_key}/${channel}/${network}/seed${seed}"

  mkdir -p "$(dirname "$launcher_log")" "$(dirname "$status_file")" "$worker_root"

  (
    set +e
    export DATASET="$dataset"
    export SETTING="$setting"
    export CHANNELS="$channel"
    export SEEDS="$seed"
    export NETWORKS="$network"
    export GPU="$gpu"
    export EPOCHS="$EPOCHS"
    export BATCH_SIZE="$BATCH_SIZE"
    export LR="$LR"
    export DEVICE="$DEVICE"
    export METRICS="$METRICS"
    export METRIC_CHOOSE="$METRIC_CHOOSE"
    export EXTRA_ARGS="$dataset_extra_args"
    export RUN_ROOT="$worker_root"
    export STOP_ON_FAIL=1
    export DRY_RUN="$DRY_RUN"

    unset DATASET_PATH_2CH DATASET_PATH_4CH DATASET_PATH_8CH
    case "$channel" in
      2ch) export DATASET_PATH_2CH="$dataset_path" ;;
      4ch) export DATASET_PATH_4CH="$dataset_path" ;;
      8ch) export DATASET_PATH_8CH="$dataset_path" ;;
    esac

    echo "[$(date '+%F %T')] LAUNCH dataset=$dataset_key/$dataset channel=$channel seed=$seed network=$network gpu=$gpu"
    bash "$WORKER_SCRIPT"
    code="$?"
    echo "$code" > "$status_file"
    echo "[$(date '+%F %T')] EXIT code=$code"
    exit "$code"
  ) > "$launcher_log" 2>&1 &

  local pid="$!"
  PIDS+=("$pid")
  PID_GPU["$pid"]="$gpu"
  PID_INFO["$pid"]="$dataset_key|$dataset|$channel|$seed|$network"
  PID_STATUS_FILE["$pid"]="$status_file"
  PID_LOG_FILE["$pid"]="$launcher_log"
  PID_START_SECONDS["$pid"]="$(date +%s)"
  PID_WORKER_ROOT["$pid"]="$worker_root"
  GPU_RUNNING["$gpu"]="$((GPU_RUNNING[$gpu] + 1))"

  echo "[$(date '+%F %T')] START dataset=$dataset_key channel=$channel seed=$seed network=$network gpu=$gpu log=$launcher_log"
}

for dataset_key in "${DATASET_LIST[@]}"; do
  dataset="$(dataset_name_for_key "$dataset_key")" || exit 1
  setting="$(setting_for_key "$dataset_key")" || exit 1
  dataset_extra_args="$(extra_args_for_key "$dataset_key")"

  for channel in "${CHANNEL_LIST[@]}"; do
    dataset_path="$(path_for_key_channel "$dataset_key" "$channel")" || exit 1
    if [[ -z "$dataset_path" ]]; then
      var_name="$(path_var_name_for_key_channel "$dataset_key" "$channel")"
      echo "Missing $var_name for dataset=$dataset_key channel=$channel." >&2
      exit 1
    fi

    for seed in "${SEED_LIST[@]}"; do
      for network in "${NETWORK_LIST[@]}"; do
        while true; do
          reap_finished_jobs

          if [[ "$STOP_REQUESTED" == "1" ]]; then
            echo "STOP_ON_FAIL=1 and a job failed; no new jobs will be launched." >&2
            if [[ "$KILL_ON_FAIL" == "1" ]]; then
              kill_running_jobs
            fi
            break 5
          fi

          gpu="$(select_gpu || true)"
          if [[ -n "${gpu:-}" ]]; then
            launch_job "$dataset_key" "$dataset" "$setting" "$channel" "$dataset_path" "$seed" "$network" "$gpu" "$dataset_extra_args"
            break
          fi

          sleep "$QUEUE_SLEEP_SECONDS"
        done
      done
    done
  done
done

while [[ "${#PIDS[@]}" -gt 0 ]]; do
  reap_finished_jobs
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    sleep "$QUEUE_SLEEP_SECONDS"
  fi
done

if [[ "$STOP_REQUESTED" == "1" ]]; then
  echo "Some jobs failed. Summary: $MASTER_SUMMARY" >&2
  exit 1
fi

echo "All queued jobs finished. Summary: $MASTER_SUMMARY"
