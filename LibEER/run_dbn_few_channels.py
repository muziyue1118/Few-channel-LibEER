#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
少通道跨个体 DBN 训练脚本。
默认使用 12 个通道，如需调整可通过 -selected_channels 覆盖。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DBN_train import main
from utils.args import get_args_parser


if __name__ == '__main__':
    parser = get_args_parser()
    test_args = [
        '-metrics', 'acc', 'macro-f1',
        '-metric_choose', 'macro-f1',
        '-setting', 'seed_sub_independent_train_val_test_setting',
        '-dataset', 'seed_de_lds',
        '-dataset_path', '/data/mzy/SEED/',
        '-batch_size', '64',
        '-seed', '2024',
        '-epochs', '5',
        '-lr', '0.02',
        '-device', 'cuda:3',
        '-selected_channels', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'
    ]
    # 解析参数：合并默认参数和命令行参数，命令行参数优先
    cli_args = sys.argv[1:]
    merged_args = test_args + cli_args
    args = parser.parse_args(merged_args)

    print("=" * 80)
    print("开始运行少通道跨个体 DBN 实验")
    print("=" * 80)
    print(f"实验模式: 跨个体 (subject-independent)")
    print(f"使用通道: {args.selected_channels}")
    print(f"通道数量: {len(args.selected_channels)}")
    print(f"数据集: {args.dataset}")
    print(f"数据集路径: {args.dataset_path}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print("=" * 80)

    main(args)


