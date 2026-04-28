#!/bin/bash

# 少通道网络实验脚本
# 运行所有支持少通道的网络，使用两种通道配置
#
# 命令行选项：
#   -g, --gpu <gpu_id>  指定使用的GPU设备ID（可选）
#                     如果不指定，将自动检测空闲GPU
#
# 使用示例：
#   ./run_few_channels_experiments.sh      # 自动检测空闲GPU
#   ./run_few_channels_experiments.sh -g 0  # 使用GPU 0
#   ./run_few_channels_experiments.sh --gpu=1  # 使用GPU 1

# 设置基本参数
BASE_DIR="/data/mzy/LibEER"
SCRIPT_DIR="$BASE_DIR/LibEER"
DATASET_PATH="/data/mzy/SEED/"

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 初始化指定GPU变量
SPECIFIED_GPU=""

# 解析命令行参数
while getopts ":g:-:" opt; do
    case $opt in
        g)  
            SPECIFIED_GPU="$OPTARG"
            ;;
        -)
            case "${OPTARG}" in
                gpu=*)
                    SPECIFIED_GPU="${OPTARG#*=}"
                    ;;
                *)
                    echo "无效的长选项: --${OPTARG}" >&2
                    exit 1
                    ;;
            esac
            ;;
        \?)
            echo "无效的选项: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

# 定义结果和日志目录，加入时间戳
RESULT_DIR="$BASE_DIR/result/few_channels_experiments/$TIMESTAMP"
LOG_DIR="$BASE_DIR/log/few_channels_experiments/$TIMESTAMP"

# 创建结果和日志目录
mkdir -p "$RESULT_DIR"
mkdir -p "$LOG_DIR"

# 定义结果文件
RESULT_FILE="$RESULT_DIR/experiment_results.txt"

# 清空结果文件
> "$RESULT_FILE"

# 检测空闲GPU的函数
# 返回空闲GPU的索引，如果没有空闲GPU返回-1
get_free_gpu() {
    # 使用nvidia-smi获取GPU信息，过滤出显存使用率低于20%的GPU
    free_gpus=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
                awk -F, '$2/$3 < 0.2 {print $1}')
    
    # 如果有空闲GPU，返回第一个
    if [ -n "$free_gpus" ]; then
        echo $free_gpus | head -n 1
    else
        echo -1
    fi
}

# 定义通道配置
# 4通道配置：FP1 FP2 T7 T8
CHANNELS_4="FP1 FP2 T7 T8"
# 8通道配置：FP1 FP2 T7 T8 F7 F8 CP5 CP6
CHANNELS_8="FP1 FP2 T7 T8 F7 F8 CP5 CP6"
# 全通道配置：所有62个通道
CHANNELS_FULL="FP1 FPZ FP2 AF3 AF4 F7 F5 F3 F1 FZ F2 F4 F6 F8 FT7 FC5 FC3 FC1 FCZ FC2 FC4 FC6 FT8 T7 C5 C3 C1 CZ C2 C4 C6 T8 TP7 CP5 CP3 CP1 CPZ CP2 CP4 CP6 TP8 P7 P5 P3 P1 PZ P2 P4 P6 P8 PO7 PO5 PO3 POZ PO4 PO6 PO8 CB1 O1 OZ O2 CB2"

# 定义网络配置
# 格式：网络名称 训练脚本 批量大小 训练轮数 学习率 GPU设备 数据集 其他参数
declare -A NETWORKS=(  
    ["ACRNN"]="run_acrnn_few_channels.py 16 1000 0.0001 cuda:2 seed_de_lds -only_seg -onehot"  
    ["BiDANN"]="run_bidann_few_channels.py 16 80 0.00004 cuda:2 seed_de_lds -onehot"  
    ["DBN"]="run_dbn_few_channels.py 512 100 0.02 cuda:2 seed_de_lds"  
    ["DGCNN"]="run_dgcnn_few_channels.py 16 150 0.0015 cuda:2 seed_de_lds -onehot"  
    ["EEGNet"]="run_eegnet_few_channels.py 256 100 0.001 cuda:3 seed_raw -only_seg -onehot"  
    ["FBSTCNet"]="run_fbstcnet_few_channels.py 32 150 0.001 cuda:3 seed_raw -only_seg"  
    ["R2GSTNN"]="run_r2gstnn_cross_subject_few_channels.py 32 60 0.00002 cuda:3 seed_raw"  
    ["RGNN"]="run_rgnn_few_channels.py 32 150 0.001 cuda:3 seed_de_lds"  
    ["SVM"]="run_svm_few_channels.py 16 50 0.001 cpu seed_raw -only_seg -onehot"  
    ["TSception"]="run_tsception_few_channels.py 32 200 0.001 cuda:3 seed_de_lds -only_seg"
)

# 记录实验结果
log_result() {
    local network=$1
    local channels=$2
    local channel_count=$3
    local acc=$4
    local f1=$5
    local time=$6
    
    echo "$network,$channels,$channel_count,$acc,$f1,$time" >> "$RESULT_FILE"
    echo "$network $channels: acc=$acc, f1=$f1, time=$time" | tee -a "$LOG_DIR/summary.log"
}

# 运行实验
run_experiment() {
    local network=$1
    local channels=$2
    local channel_count=$3
    
    echo "\n========================================="
    echo "开始运行 $network 实验，使用 $channel_count 个通道"
    echo "使用通道: $channels"
    echo "========================================="
    
    # 获取网络配置
    IFS=' ' read -r -a config <<< "${NETWORKS[$network]}"
    script="${config[0]}"
    batch_size="${config[1]}"
    epochs="${config[2]}"
    lr="${config[3]}"
    device="${config[4]}"
    dataset="${config[5]}"
    other_params="${config[@]:6}"
    
    # 检测空闲GPU并更新设备配置
    if [[ "$device" == cuda* ]]; then
        if [ -n "$SPECIFIED_GPU" ]; then
            # 使用用户指定的GPU
            device="cuda:$SPECIFIED_GPU"
            echo "使用指定GPU: $device"
        else
            # 尝试获取空闲GPU，最多重试5次
            retries=0
            max_retries=5
            
            while [ $retries -lt $max_retries ]; do
                free_gpu=$(get_free_gpu)
                if [ $free_gpu -ne -1 ]; then
                    # 找到空闲GPU，更新设备配置
                    device="cuda:$free_gpu"
                    echo "找到空闲GPU: $device"
                    break
                else
                    # 没有空闲GPU，等待30秒后重试
                    echo "没有空闲GPU，等待30秒后重试..."
                    sleep 30
                    retries=$((retries + 1))
                fi
            done
            
            # 如果重试后仍没有空闲GPU，退出脚本
            if [ $free_gpu -eq -1 ]; then
                echo "错误：没有找到空闲GPU！"
                exit 1
            fi
        fi
    fi
    
    # 设置CUDA设备
    export CUDA_VISIBLE_DEVICES="${device#cuda:}"
    
    # 构建命令行参数
    cmd=()
    
    # 将通道名称转换为索引
    # 完整的62通道映射，使用if-else语句避免语法问题
    channel_indices=$(echo "$channels" | tr ' ' '\n' | awk '{
        if ($1 == "FP1") print "0";
        else if ($1 == "FPZ") print "1";
        else if ($1 == "FP2") print "2";
        else if ($1 == "AF3") print "3";
        else if ($1 == "AF4") print "4";
        else if ($1 == "F7") print "5";
        else if ($1 == "F5") print "6";
        else if ($1 == "F3") print "7";
        else if ($1 == "F1") print "8";
        else if ($1 == "FZ") print "9";
        else if ($1 == "F2") print "10";
        else if ($1 == "F4") print "11";
        else if ($1 == "F6") print "12";
        else if ($1 == "F8") print "13";
        else if ($1 == "FT7") print "14";
        else if ($1 == "FC5") print "15";
        else if ($1 == "FC3") print "16";
        else if ($1 == "FC1") print "17";
        else if ($1 == "FCZ") print "18";
        else if ($1 == "FC2") print "19";
        else if ($1 == "FC4") print "20";
        else if ($1 == "FC6") print "21";
        else if ($1 == "FT8") print "22";
        else if ($1 == "T7") print "23";
        else if ($1 == "C5") print "24";
        else if ($1 == "C3") print "25";
        else if ($1 == "C1") print "26";
        else if ($1 == "CZ") print "27";
        else if ($1 == "C2") print "28";
        else if ($1 == "C4") print "29";
        else if ($1 == "C6") print "30";
        else if ($1 == "T8") print "31";
        else if ($1 == "TP7") print "32";
        else if ($1 == "CP5") print "33";
        else if ($1 == "CP3") print "34";
        else if ($1 == "CP1") print "35";
        else if ($1 == "CPZ") print "36";
        else if ($1 == "CP2") print "37";
        else if ($1 == "CP4") print "38";
        else if ($1 == "CP6") print "39";
        else if ($1 == "TP8") print "40";
        else if ($1 == "P7") print "41";
        else if ($1 == "P5") print "42";
        else if ($1 == "P3") print "43";
        else if ($1 == "P1") print "44";
        else if ($1 == "PZ") print "45";
        else if ($1 == "P2") print "46";
        else if ($1 == "P4") print "47";
        else if ($1 == "P6") print "48";
        else if ($1 == "P8") print "49";
        else if ($1 == "PO7") print "50";
        else if ($1 == "PO5") print "51";
        else if ($1 == "PO3") print "52";
        else if ($1 == "POZ") print "53";
        else if ($1 == "PO4") print "54";
        else if ($1 == "PO6") print "55";
        else if ($1 == "PO8") print "56";
        else if ($1 == "CB1") print "57";
        else if ($1 == "O1") print "58";
        else if ($1 == "OZ") print "59";
        else if ($1 == "O2") print "60";
        else if ($1 == "CB2") print "61";
    }' | tr '\n' ' ' | sed 's/ $//')
    
    # 特殊处理某些网络
    case "$network" in
        "RGNN")
            # RGNN使用Python函数调用，需要特殊处理
            # 设置默认seed值
            seed="2024"
            cmd=(python3 "$SCRIPT_DIR/$script" \
                --channel_names $channels \
                --device "$device" \
                --batch_size "$batch_size" \
                --epochs "$epochs" \
                --lr "$lr" \
                --seed "$seed"
            )
            ;;
        "R2GSTNN")
            # R2GSTNN的脚本格式不同，需要重新构建命令
            # 注意：设置CUDA_VISIBLE_DEVICES后，Python进程只能看到指定GPU，所以使用cuda:0
            cmd=(python3 "$SCRIPT_DIR/R2GSTNN_train.py" \
                -device cuda:0 \
                -batch_size "$batch_size" \
                -epochs "$epochs" \
                -lr "$lr" \
                -dataset_path "$DATASET_PATH" \
                -dataset "$dataset" \
                -experiment_mode "subject-independent" \
                -selected_channels $channel_indices \
                -seed "42" \
                -setting "seed_sub_independent_train_val_test_setting" \
                -sample_length "9" \
                -onehot)
            ;;
        "DBN")
            # 注意：设置CUDA_VISIBLE_DEVICES后，Python进程只能看到指定GPU，所以使用cuda:0
            cmd=(python3 "$SCRIPT_DIR/$script" \
                -selected_channels $channel_indices \
                -batch_size "$batch_size" \
                -epochs "$epochs" \
                -lr "$lr" \
                -device cuda:0 \
                -dataset_path "$DATASET_PATH" \
                -dataset "$dataset" \
                $other_params)
            ;;
        *)
            # 标准运行方式，使用通道索引参数
            # 注意：设置CUDA_VISIBLE_DEVICES后，Python进程只能看到指定GPU，所以使用cuda:0
            cmd=(python3 "$SCRIPT_DIR/$script" \
                -selected_channels $channel_indices \
                -batch_size "$batch_size" \
                -epochs "$epochs" \
                -lr "$lr" \
                -dataset_path "$DATASET_PATH" \
                -dataset "$dataset" \
                -device cuda:0 \
                $other_params)
            ;;
        esac
    
    # 构建日志文件路径
    log_file="$LOG_DIR/${network}_${channel_count}ch.log"
    
    # 运行命令，同时输出到日志文件和控制台
    echo "运行命令: ${cmd[*]}"
    
    # 保存命令返回值
    "${cmd[@]}" 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
    
    # 检查命令返回值
    if [ $exit_code -ne 0 ]; then
        echo "错误：$network 网络运行失败！"
        echo "详细日志请查看：$log_file"
        exit 1
    fi
    
    # 提取结果
    # 支持多种输出格式
    # 格式1: best_test_acc: 0.35, best_test_macro-f1: 0.18
    acc=$(grep -o "best_test_acc: [0-9.]*" "$log_file" | tail -n 1 | awk '{print $2}')
    f1=$(grep -o "best_test_macro-f1: [0-9.]*" "$log_file" | tail -n 1 | awk '{print $2}')
    
    # 格式2: ALLRound Mean and Std of acc : 0.3466/0.0000
    if [ -z "$acc" ]; then
        acc=$(grep -o "ALLRound Mean and Std of acc : [0-9.]*/[0-9.]*" "$log_file" | tail -n 1 | awk '{print $7}' | cut -d'/' -f1)
    fi
    if [ -z "$f1" ]; then
        f1=$(grep -o "ALLRound Mean and Std of macro-f1 : [0-9.]*/[0-9.]*" "$log_file" | tail -n 1 | awk '{print $8}' | cut -d'/' -f1)
    fi
    
    # 格式3: |  Result  |      acc      |   macro-f1    |
    if [ -z "$acc" ]; then
        acc=$(grep -A1 "|  Result  |      acc      |   macro-f1    |" "$log_file" | tail -n 1 | awk '{print $3}')
    fi
    if [ -z "$f1" ]; then
        f1=$(grep -A1 "|  Result  |      acc      |   macro-f1    |" "$log_file" | tail -n 1 | awk '{print $4}')
    fi
    
    # 如果没有找到结果，使用默认值
    if [ -z "$acc" ]; then
        acc="N/A"
    fi
    if [ -z "$f1" ]; then
        f1="N/A"
    fi
    
    # 记录结果
    log_result "$network" "$channels" "$channel_count" "$acc" "$f1" "$(date +%Y-%m-%d_%H-%M-%S)"
    
    echo "$network $channel_count 通道实验完成"
    echo "acc: $acc, macro-f1: $f1"
}

# 测试模式：只运行一个网络的一个通道配置
# 取消注释下面的代码，注释掉完整循环来启用测试模式
TEST_MODE = false

if [ "$TEST_MODE" = true ]; then
    # 测试单个网络
    network="ACRNN"
    echo "\n========================================="
    echo "测试模式：处理网络: $network"
    echo "========================================="
    
    # 测试全通道配置
    run_experiment "$network" "$CHANNELS_FULL" 62
else
    # 定义固定的网络运行顺序
    NETWORK_ORDER=("ACRNN" "BiDANN" "DBN" "DGCNN" "EEGNet" "FBSTCNet" "R2GSTNN" "RGNN" "TSception" "SVM" "MLP" )

# 主循环：运行所有网络，按照固定顺序
    for network in "${NETWORK_ORDER[@]}"; do
        echo "\n========================================="
        echo "处理网络: $network"
        echo "========================================="
        
        # # 运行4通道配置
        # run_experiment "$network" "$CHANNELS_4" 4
        
        # # 运行8通道配置
        # run_experiment "$network" "$CHANNELS_8" 8
        
        # 运行全通道配置
        run_experiment "$network" "$CHANNELS_FULL" 62
    done
fi

# 显示结果摘要
echo "\n========================================="
echo "所有实验完成！"
echo "结果保存在: $RESULT_FILE"
echo "日志保存在: $LOG_DIR"
echo "========================================="
echo "\n实验结果摘要："
cat "$RESULT_FILE"
