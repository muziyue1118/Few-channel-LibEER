import os
import subprocess
import sys
import argparse

# 设置CUDA设备为cuda:0，后续会根据命令行参数覆盖
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run_rgnn_few_channels(channel_indices=None, channel_names=None, setting='seed_sub_independent_train_val_test_setting', 
                           dataset_path='/data/mzy/SEED/', dataset='seed_de_lds', batch_size=1, epochs=150, lr=0.001, device='cpu',
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
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable, os.path.join(script_dir, 'RGNN_train.py'),
        '-metrics', *metrics,
        '-model', 'RGNN_official',
        '-metric_choose', metric_choose,
        '-setting', setting,
        '-dataset_path', dataset_path,
        '-dataset', dataset,
        '-batch_size', str(batch_size),
        '-epochs', str(epochs),
        '-lr', str(lr),
        '-time_window', '1',
        '-feature_type', 'de_lds',
        '-seed', str(seed),
        '-device', device
    ]
    
    # 添加通道选择参数
    if channel_names and len(channel_names) > 0:
        cmd.extend(['-selected_channel_names', *channel_names])
    elif channel_indices and len(channel_indices) > 0:
        cmd.extend(['-selected_channels', *map(str, channel_indices)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # 执行命令
    try:
        print("\n开始执行命令...")
        print(f"命令：{' '.join(cmd)}")
        # 使用stdout=subprocess.PIPE和stderr=subprocess.STDOUT，实时输出日志
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # 实时输出日志
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        returncode = process.poll()
        if returncode == 0:
            print("\n=== 命令执行成功 ===\n")
            return True
        else:
            print(f"\n=== 命令执行失败: {returncode} ===\n")
            return False
    except Exception as e:
        print(f"\n=== 命令执行出错: {e} ===\n")
        return False

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="运行RGNN模型的少通道跨个体实验")
    parser.add_argument('--channel_names', type=str, nargs='+', default=None, 
                        help='要使用的通道名称列表（例如：FP1 FP2 T7 T8）')
    parser.add_argument('--setting', type=str, default='seed_sub_independent_train_val_test_setting',
                        help='实验设置')
    parser.add_argument('--dataset_path', type=str, default='/data/mzy/SEED/',
                        help='数据集路径')
    parser.add_argument('--dataset', type=str, default='seed_de_lds',
                        help='数据集名称')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批量大小')
    parser.add_argument('--epochs', type=int, default=150,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--seed', type=int, default=2024,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cpu',
                        help='训练设备（例如：cpu, cuda, cuda:0）')
    
    args = parser.parse_args()
    
    # 使用解析得到的通道列表，如果没有提供，则使用默认的前额叶通道
    channel_names = args.channel_names if args.channel_names else ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8']
    
    print(f"使用通道: {channel_names}")
    
    # 如果设备是cuda，更新CUDA_VISIBLE_DEVICES环境变量
    if args.device.startswith('cuda'):
        if ':' in args.device:
            # 提取GPU索引
            gpu_idx = args.device.split(':')[1]
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx
        else:
            # 使用所有可用GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        # 当设置了CUDA_VISIBLE_DEVICES后，可见GPU在程序中会被识别为cuda:0
        device_arg = 'cuda:0'
    else:
        # CPU设备保持不变
        device_arg = args.device
    
    run_rgnn_few_channels(
        channel_names=channel_names,
        setting=args.setting,
        dataset_path=args.dataset_path,
        dataset=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=device_arg
    )
