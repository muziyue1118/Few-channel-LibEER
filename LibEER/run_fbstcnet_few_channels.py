#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
少通道跨个体 FBSTCNet 训练脚本。
默认选择 8 个通道，可通过命令行覆盖 -selected_channels。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from FBSTCNet_train import main
from utils.args import get_args_parser
from utils.channel_selector import apply_channel_name_selection


if __name__ == '__main__':
    parser = get_args_parser()
    
    # 设置默认参数
    default_args = [
        '-metrics', 'acc', 'macro-f1',
        '-dataset', 'seed_raw',
        '-data_dir', '/data/mzy/SEED/',
        '-model', 'FBSTCNet',
        '-batch_size', '4',
        '-seed', '2024',
        '-epochs', '10',
        '-lr', '0.001',
        '-device', 'cuda:3',
        '-sample_length', '200',
        '-stride', '200',
        '-only_seg'
    ]
    
    # 解析命令行参数：合并默认参数和命令行参数，命令行参数优先
    cli_args = sys.argv[1:]
    merged_args = default_args + cli_args
    args = parser.parse_args(merged_args)
    
    # 强制设置跨个体训练模式
    args.setting = 'seed_sub_independent_train_val_test_setting'
    args.experiment_mode = 'subject-independent'  # 明确设置实验模式为跨个体
    
    # 设置默认通道（如果没有指定）
    if args.selected_channels is None:
        args.selected_channels = [0, 2, 4, 6, 8, 10, 12, 14]  # 默认8通道
    else:
        # 确保selected_channels是整数列表
        args.selected_channels = list(map(int, args.selected_channels))
    
    # 确保GPU设备可用
    # 注意：设置CUDA_VISIBLE_DEVICES后，Python进程只能看到指定GPU，所以使用cuda:0
    if args.device.startswith('cuda:'):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(':')[1]
        # 重置设备为cuda:0，因为CUDA_VISIBLE_DEVICES会重新映射GPU索引
        args.device = 'cuda:0'
    
    # 打印实际使用的命令行参数
    print(f"实际使用的通道索引: {args.selected_channels}")
    
    # 应用通道名称到索引的转换
    apply_channel_name_selection(args)

    print("=" * 80)
    print("开始运行少通道跨个体 FBSTCNet 实验")
    print("=" * 80)
    print(f"实验模式: {args.experiment_mode}")
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
