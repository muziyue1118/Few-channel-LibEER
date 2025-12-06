#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
少通道跨个体BiDANN训练脚本
支持选择使用哪些通道的数据进行训练，在输入时支持通道选择
不同次跑的实验可以使用不同数量的通道
注意：BiDANN会分脑区（左右半球）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from BiDANN_train import main
from utils.args import get_args_parser

if __name__ == '__main__':
    # 创建参数解析器
    parser = get_args_parser()
    
    # 设置跨个体实验参数，使用16个通道（通道索引0-15）
    # 注意：需要根据实际数据集路径修改 -dataset_path 参数
    # 可以根据需要修改通道数量和具体通道索引
    test_args = [
        '-metrics', 'acc', 'macro-f1',
        '-metric_choose', 'macro-f1',
        '-setting', 'seed_sub_independent_train_val_test_setting',
        '-dataset', 'seed_de_lds',
        '-dataset_path', '/data/mzy/SEED/',  # SEED数据集路径
        '-batch_size', '16',
        '-seed', '2024',
        '-epochs', '10',  # 使用较少epoch进行快速测试
        '-lr', '0.00004',  # BiDANN通常使用较小的学习率
        '-onehot',
        '-sample_length', '9',  # BiDANN需要sample_length参数
        '-selected_channels', '0', '1', '2', '3', '4', '5', '6', '7', 
                             '8', '9', '10', '11', '12', '13', '14', '15'  # 使用前16个通道
    ]
    
    # 解析参数：合并默认参数和命令行参数，命令行参数优先
    cli_args = sys.argv[1:]
    merged_args = test_args + cli_args
    args = parser.parse_args(merged_args)
    
    print("=" * 80)
    print("开始运行少通道跨个体BiDANN实验")
    print("=" * 80)
    print(f"实验模式: 跨个体 (subject-independent)")
    print(f"使用通道: {args.selected_channels}")
    print(f"通道数量: {len(args.selected_channels)}")
    print(f"数据集: {args.dataset}")
    print(f"数据集路径: {args.dataset_path}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print(f"样本长度: {args.sample_length}")
    print("注意: BiDANN模型会自动将选择的通道分为左右脑区进行处理")
    print("=" * 80)
    
    # 运行主函数
    main(args)


