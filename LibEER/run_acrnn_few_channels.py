#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
少通道跨个体 ACRNN 训练脚本
默认按照跨个体（subject-independent）设置运行，并在输入阶段裁剪到指定的少通道。
可以通过命令行再次传入任意参数（例如不同的通道列表、epochs、学习率等）来覆盖默认配置，
从而方便地开展多组少通道实验。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ACRNN_train import main
from utils.args import get_args_parser


def build_default_args():
    """构造一组跨个体少通道 ACRNN 的默认参数"""
    return [
        '-model', 'ACRNN',
        '-metrics', 'acc', 'macro-f1',
        '-metric_choose', 'macro-f1',
        '-setting', 'seed_sub_independent_train_val_test_setting',
        '-dataset', 'seed_de_lds',
        '-dataset_path', '/data/mzy/SEED/',  # TODO: 根据本地数据路径修改
        '-batch_size', '16',
        '-epochs', '200',
        '-lr', '0.0001',
        '-seed', '2024',
        '-sample_length', '128',
        '-stride', '128',
        '-only_seg',
        '-onehot',
        # 默认使用 0-15 共 16 个通道，可在命令行中用 -selected_channels 覆盖
        '-selected_channels', '0', '1', '2', '3', '4', '5', '6', '7',
                             '8', '9', '10', '11', '12', '13', '14', '15'
    ]


if __name__ == '__main__':
    parser = get_args_parser()
    default_args = build_default_args()

    # 允许用户通过命令行追加参数覆盖默认配置；后出现的参数优先生效
    cli_args = sys.argv[1:]
    merged_args = default_args + cli_args
    args = parser.parse_args(merged_args)

    print("=" * 80)
    print("开始运行少通道跨个体 ACRNN 实验")
    print("=" * 80)
    print("实验模式: 跨个体 (subject-independent)")
    print(f"数据集: {args.dataset}")
    print(f"数据集路径: {args.dataset_path}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print(f"样本长度: {args.sample_length}, 步长: {args.stride}")
    if args.selected_channels is None:
        print("未指定 -selected_channels，将使用全部可用通道。")
    else:
        print(f"使用通道索引: {args.selected_channels}")
        print(f"通道数量: {len(args.selected_channels)}")
    print("提示: 直接在命令行追加参数即可覆盖默认配置，例如：")
    print("      python run_acrnn_few_channels.py -selected_channels 0 2 4 6 -epochs 100")
    print("=" * 80)

    main(args)


