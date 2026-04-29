#!/usr/bin/env bash
set -u

# Run all LibEER model training entrypoints on already-cropped few-channel data.
#
# Required environment variables:
#   DATASET_PATH_2CH=/path/to/2ch/data
#   DATASET_PATH_4CH=/path/to/4ch/data
#   DATASET_PATH_8CH=/path/to/8ch/data
#
# Common optional variables:
#   DATASET=seed_de_lds
#   SETTING=seed_sub_dependent_train_val_test_setting
#   CHANNELS="2ch 4ch 8ch"
#   SEEDS="2024"
#   GPU=0
#   EPOCHS=80
#   BATCH_SIZE=32
#   LR=0.001
#   EXTRA_ARGS="-time_window 1 -feature_type de_lds"
#   NETWORKS="DGCNN CoralDgcnn GCBNet GCBNet_BLS CDCN DBN HSLT BiDANN R2GSTNN NSAL_DGAT EEGNet TSception ACRNN RGNN_official MsMda FBSTCNet PRRL"
#   STOP_ON_FAIL=0
#   DRY_RUN=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIBEER_DIR="$PROJECT_ROOT/LibEER"

DATASET="${DATASET:-seed_de_lds}"
SETTING="${SETTING:-seed_sub_dependent_train_val_test_setting}"
CHANNELS="${CHANNELS:-2ch 4ch 8ch}"
SEEDS="${SEEDS:-2024}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
DEVICE="${DEVICE:-cuda}"
METRICS="${METRICS:-acc macro-f1}"
METRIC_CHOOSE="${METRIC_CHOOSE:-macro-f1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs_all_networks}"

DEFAULT_NETWORKS="DGCNN CoralDgcnn GCBNet GCBNet_BLS CDCN DBN HSLT BiDANN R2GSTNN NSAL_DGAT EEGNet TSception ACRNN RGNN_official MsMda FBSTCNet PRRL"
NETWORKS="${NETWORKS:-$DEFAULT_NETWORKS}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results" "$RUN_ROOT/processed"
SUMMARY="$RUN_ROOT/summary.tsv"
if [[ ! -f "$SUMMARY" ]]; then
  printf "time\tchannel\tseed\tnetwork\tstatus\texit_code\tseconds\tlog_file\n" > "$SUMMARY"
fi

dataset_path_for_channel() {
  case "$1" in
    2ch) printf "%s" "${DATASET_PATH_2CH:-}" ;;
    4ch) printf "%s" "${DATASET_PATH_4CH:-}" ;;
    8ch) printf "%s" "${DATASET_PATH_8CH:-}" ;;
    *)
      echo "Unsupported CHANNELS entry '$1'. Use 2ch, 4ch, 8ch." >&2
      return 1
      ;;
  esac
}

script_for_network() {
  case "$1" in
    ACRNN) echo "ACRNN_train.py" ;;
    BiDANN) echo "BiDANN_train.py" ;;
    CDCN) echo "CDCN_train.py" ;;
    CoralDgcnn) echo "CoralDgcnnTrain.py" ;;
    DBN) echo "DBN_train.py" ;;
    DGCNN) echo "DGCNN_train.py" ;;
    EEGNet) echo "EEGNet_train.py" ;;
    FBSTCNet) echo "FBSTCNet_train.py" ;;
    GCBNet) echo "GCBNet_train.py" ;;
    GCBNet_BLS) echo "GCBNet_BLS_train.py" ;;
    HSLT) echo "HSLT_train.py" ;;
    MsMda) echo "Msmda_train.py" ;;
    NSAL_DGAT) echo "NSAL_DGAT_train.py" ;;
    PRRL) echo "PR_RL_train.py" ;;
    R2GSTNN) echo "R2GSTNN_train.py" ;;
    RGNN_official) echo "RGNN_train.py" ;;
    svm) echo "svm_train.py" ;;
    TSception) echo "TSception_train.py" ;;
    *)
      echo "Unknown network '$1'." >&2
      return 1
      ;;
  esac
}

model_arg_for_network() {
  case "$1" in
    RGNN_official) echo "RGNN_official" ;;
    PRRL) echo "PRRL" ;;
    MsMda) echo "MsMda" ;;
    svm) echo "svm" ;;
    *) echo "$1" ;;
  esac
}

read -r -a CHANNEL_LIST <<< "$CHANNELS"
read -r -a SEED_LIST <<< "$SEEDS"
read -r -a NETWORK_LIST <<< "$NETWORKS"
read -r -a METRIC_LIST <<< "$METRICS"
read -r -a EXTRA_ARG_LIST <<< "${EXTRA_ARGS:-}"

if [[ -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

cd "$LIBEER_DIR" || exit 1

for channel in "${CHANNEL_LIST[@]}"; do
  dataset_path="$(dataset_path_for_channel "$channel")" || exit 1
  if [[ -z "$dataset_path" ]]; then
    case "$channel" in
      2ch) missing_var="DATASET_PATH_2CH" ;;
      4ch) missing_var="DATASET_PATH_4CH" ;;
      8ch) missing_var="DATASET_PATH_8CH" ;;
      *) missing_var="DATASET_PATH_<CHANNEL>" ;;
    esac
    echo "Missing $missing_var. Set the path for $channel cropped data." >&2
    exit 1
  fi

  for seed in "${SEED_LIST[@]}"; do
    for network in "${NETWORK_LIST[@]}"; do
      script_name="$(script_for_network "$network")" || exit 1
      model_name="$(model_arg_for_network "$network")"
      log_file="$RUN_ROOT/logs/${DATASET}_${channel}_${network}_seed${seed}.log"
      output_dir="$RUN_ROOT/results/${DATASET}/${channel}/${network}/seed${seed}"
      data_dir="$RUN_ROOT/processed/${DATASET}/${channel}/${network}/seed${seed}"
      mkdir -p "$(dirname "$log_file")" "$output_dir" "$data_dir"

      cmd=(
        python "$script_name"
        -model "$model_name"
        -dataset "$DATASET"
        -dataset_path "$dataset_path"
        -setting "$SETTING"
        -metrics "${METRIC_LIST[@]}"
        -metric_choose "$METRIC_CHOOSE"
        -batch_size "$BATCH_SIZE"
        -epochs "$EPOCHS"
        -lr "$LR"
        -seed "$seed"
        -device "$DEVICE"
        -output_dir "$output_dir"
        -log_dir "$RUN_ROOT/state_logs"
        -data_dir "$data_dir"
        "${EXTRA_ARG_LIST[@]}"
      )

      echo "[$(date '+%F %T')] START channel=$channel seed=$seed network=$network"
      printf "Command:" | tee "$log_file"
      printf " %q" "${cmd[@]}" | tee -a "$log_file"
      printf "\n" | tee -a "$log_file"

      if [[ "$DRY_RUN" == "1" ]]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "$(date '+%F %T')" "$channel" "$seed" "$network" "dry-run" "0" "0" "$log_file" >> "$SUMMARY"
        continue
      fi

      start_seconds="$(date +%s)"
      "${cmd[@]}" >> "$log_file" 2>&1
      exit_code="$?"
      end_seconds="$(date +%s)"
      elapsed="$((end_seconds - start_seconds))"

      if [[ "$exit_code" -eq 0 ]]; then
        status="ok"
      else
        status="failed"
      fi
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date '+%F %T')" "$channel" "$seed" "$network" "$status" "$exit_code" "$elapsed" "$log_file" >> "$SUMMARY"
      echo "[$(date '+%F %T')] DONE channel=$channel seed=$seed network=$network status=$status log=$log_file"

      if [[ "$exit_code" -ne 0 && "$STOP_ON_FAIL" == "1" ]]; then
        exit "$exit_code"
      fi
    done
  done
done

echo "All requested runs finished. Summary: $SUMMARY"
