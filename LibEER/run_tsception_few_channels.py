#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
少通道跨个体TSception训练脚本
支持选择使用哪些通道的数据进行训练
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from TSception_train import main
from utils.args import get_args_parser

if __name__ == '__main__':
    # 创建参数解析器
    parser = get_args_parser()
    
    # 设置跨个体实验参数
    test_args = [
        '-metrics', 'acc', 'macro-f1',
        '-metric_choose', 'macro-f1',
        '-setting', 'seed_sub_independent_train_val_test_setting',
        '-dataset', 'seed_de_lds',
        '-dataset_path', '/data/mzy/SEED/',  # SEED数据集路径
        '-batch_size', '16',
        '-seed', '2024',
        '-epochs', '5',  # 使用较少epoch进行快速测试
        '-lr', '0.001',
        '-only_seg',
        '-sample_length', '200',
        '-stride', '200',
        '-device', 'cuda:1',  # 使用cuda:1
        '-selected_channels', '0', '1', '2', '3', '4', '5', '6', '7', 
                             '8', '9', '10'  # 使用前16个通道
    ]
    
    # 解析参数：合并默认参数和命令行参数，命令行参数优先
    cli_args = sys.argv[1:]
    merged_args = test_args + cli_args
    args = parser.parse_args(merged_args)
    
    print("=" * 80)
    print("开始运行少通道跨个体TSception实验")
    print("=" * 80)
    print(f"实验模式: 跨个体 (subject-independent)")
    print(f"使用通道: {args.selected_channels}")
    print(f"通道数量: {len(args.selected_channels)}")
    print(f"数据集: {args.dataset}")
    print(f"数据集路径: {args.dataset_path}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print(f"使用设备: {args.device}")
    print("=" * 80)
    
    # 运行主函数
    main(args)