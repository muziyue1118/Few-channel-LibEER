#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
少通道跨个体R2GSTNN训练脚本
支持指定通道进行训练
"""

import sys
import os

# 设置命令行参数
# 示例：使用指定的6个通道进行跨个体训练
# 选择的通道索引
selected_channels = ["0", "1", "2", "4", "36"]

# 构建命令行参数
test_args = [
    "R2GSTNN_train.py",
    "-device", "cuda:1",              # 使用指定的GPU
    "-batch_size", "8",             # 批量大小
    "-epochs", "5",                 # 训练轮数（为了快速测试，使用较小的值）
    "-lr", "0.00002",               # 学习率
    "-dataset_path", "/data/mzy/SEED/", # 数据路径，使用正确的参数名称
    "-dataset", "seed_raw",             # 数据集，使用正确的格式名称
    "-experiment_mode", "subject-independent",  # 跨个体模式
] + ["-selected_channels"] + selected_channels + ["-seed", "42"]  # 通道索引作为单独的参数

# 打印实验参数
print("\n===== 少通道跨个体R2GSTNN实验参数 =====")
print(f"数据集路径: /data/mzy/SEED/")
print(f"数据集参数: -dataset_path /data/mzy/SEED/")
print(f"数据集名称: seed_raw")
print(f"实验模式: subject-independent (跨个体)")
print(f"使用的通道索引: 0,1,2,3,4,36")
print(f"GPU设备: cuda:1")
print(f"批量大小: 8")
print(f"训练轮数: 5")
print(f"学习率: 0.00002")
print(f"随机种子: 42")
print("==========================================\n")

# 将参数传递给主模块
sys.argv = test_args

# 导入并运行主模块
from R2GSTNN_train import main
from utils.args import get_args_parser

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print("开始运行R2GSTNN少通道跨个体训练...")
    main(args)
    print("R2GSTNN少通道跨个体训练完成！")
