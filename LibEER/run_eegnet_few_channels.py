#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
少通道跨个体 EEGNet 训练脚本。
默认选择 8 个通道，可通过命令行覆盖 -selected_channels。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from EEGNet_train import main
from utils.args import get_args_parser


if __name__ == '__main__':
    parser = get_args_parser()
    default_args = [
        '-metrics', 'acc', 'macro-f1',
        '-metric_choose', 'macro-f1',
        '-setting', 'seed_sub_independent_train_val_test_setting',
        '-dataset', 'seed_raw',
        '-dataset_path', '/data/mzy/SEED/',
        '-batch_size', '16',
        '-seed', '2024',
        '-epochs', '50',
        '-lr', '0.001',
        '-device', 'cuda:3',
        '-onehot',
        '-sample_length', '200',
        '-stride', '200',
        '-only_seg',
        '-selected_channels', '0', '2', '4', '6', '8', '10', '12', '14'
    ]
    # 解析参数：合并默认参数和命令行参数，命令行参数优先
    cli_args = sys.argv[1:]
    merged_args = default_args + cli_args
    args = parser.parse_args(merged_args)

    print("=" * 80)
    print("开始运行少通道跨个体 EEGNet 实验")
    print("=" * 80)
    print("实验模式: 跨个体 (subject-independent)")
    if args.selected_channels is None:
        print("当前未指定通道，将默认使用全部通道。")
    else:
        print(f"使用通道索引: {args.selected_channels}")
        print(f"通道数量: {len(args.selected_channels)}")
    print(f"数据集: {args.dataset}")
    print(f"数据集路径: {args.dataset_path}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print("=" * 80)

    main(args)

