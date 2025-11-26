import os
import subprocess
import sys

# 设置CUDA设备为cuda:0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run_rgnn_few_channels(channel_indices=None, channel_names=None, setting='seed_sub_independent_train_val_test_setting', 
                           dataset_path='/data/mzy/SEED/', dataset='seed_de_lds', batch_size=1, epochs=150, 
                            seed=2024, metrics=['acc', 'macro-f1'], metric_choose='macro-f1'):
    """
    运行RGNN模型的少通道跨个体实验
    
    Args:
        channel_indices: 要使用的通道索引列表
        channel_names: 要使用的通道名称列表
        setting: 实验设置
        dataset_path: 数据集路径
        dataset: 数据集名称
        batch_size: 批量大小
        epochs: 训练轮数
        seed: 随机种子
        metrics: 评估指标
        metric_choose: 选择的评估指标
    """
    # 构建命令行参数
    cmd = [
        sys.executable, 'RGNN_train.py',
        '-metrics', *metrics,
        '-model', 'RGNN_official',
        '-metric_choose', metric_choose,
        '-setting', setting,
        '-dataset_path', dataset_path,
        '-dataset', dataset,
        '-batch_size', '1',
        '-epochs', str(epochs),
        '-time_window', '1',
        '-feature_type', 'de_lds',
        '-seed', str(seed),
        '-device', 'cpu'
    ]
    
    # 添加通道选择参数
    if channel_names and len(channel_names) > 0:
        cmd.extend(['-selected_channel_names', *channel_names])
    elif channel_indices and len(channel_indices) > 0:
        cmd.extend(['-selected_channels', *map(str, channel_indices)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # 执行命令
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("\n=== 命令执行成功 ===\n")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n=== 命令执行失败: {e.returncode} ===\n")
        print("错误输出:")
        print(e.stderr)
        return False

if __name__ == "__main__":
    # 示例：使用部分前额叶通道进行跨个体实验
    # 这里使用FP1、FP2、F3、F4、F7、F8等前额叶通道
    # 可以根据需要修改通道列表
    frontal_channels = ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8']
    
    print(f"使用通道: {frontal_channels}")
    run_rgnn_few_channels(
        channel_names=frontal_channels,
        setting='seed_sub_independent_train_val_test_setting',
        dataset_path='/data/mzy/SEED/',
        dataset='seed_de_lds',
        batch_size=1,
        epochs=150,
        seed=2024
    )
