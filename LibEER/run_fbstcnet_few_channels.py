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
    
    # 解析命令行参数，使用默认参数作为基础
    args = parser.parse_args()
    
    # 强制设置所有关键参数，确保正确的值被传递
    args.dataset_path = '/data/mzy/SEED/'
    args.data_dir = '/data/mzy/SEED/'
    args.dataset = 'seed_raw'
    args.batch_size = 4  # 小批量，避免显存不足
    args.epochs = 10
    args.lr = 0.001
    args.device = 'cuda:3'
    args.sample_length = 200
    args.stride = 200
    args.setting = 'seed_sub_independent_train_val_test_setting'  # 设置跨个体训练模式
    
    # 设置默认通道（如果没有指定）
    if args.selected_channels is None:
        args.selected_channels = [0, 2, 4, 6, 8, 10, 12, 14]  # 默认8通道
    else:
        # 确保selected_channels是整数列表
        args.selected_channels = list(map(int, args.selected_channels))
    
    # 确保GPU设备可用
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(':')[1] if args.device.startswith('cuda:') else args.device
    
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
