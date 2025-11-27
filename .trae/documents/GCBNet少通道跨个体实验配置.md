# GCBNet少通道跨个体实验配置

## 1. 模型分析

GCBNet模型已经支持动态通道数量：
- 构造函数接受`num_electrodes`参数，用于指定输入通道数
- 在`forward`方法中动态初始化全连接层，适应不同通道数量的输入
- 数据加载和处理代码已经支持通道选择，通过`selected_channels`参数

## 2. 实现方案

### 2.1 运行命令行代码

使用现有的`GCBNet_train.py`脚本，添加通道选择参数即可：

```bash
python GCBNet_train.py -metrics 'acc' 'macro-f1' -model GCBNet -metric_choose 'macro-f1' -setting seed_sub_independent_train_val_test_setting -dataset_path /data/mzy/SEED/ -dataset seed_de_lds -batch_size 16 -epochs 150 -lr 0.001 -seed 2024 -device cuda:1 -selected_channels 0 1 2 3 4
```

### 2.2 在launch.json中添加配置

在`launch.json`文件中添加以下配置：

```json
{
    "name": "GCBNet: 少通道跨个体实验 (5通道)",
    "type": "debugpy",
    "request": "launch",
    "program": "/data/mzy/LibEER/LibEER/GCBNet_train.py",
    "console": "integratedTerminal",
    "cwd": "/data/mzy/LibEER/LibEER",
    "args": [
        "-metrics", "acc", "macro-f1",
        "-model", "GCBNet",
        "-metric_choose", "macro-f1",
        "-setting", "seed_sub_independent_train_val_test_setting",
        "-dataset", "seed_de_lds",
        "-dataset_path", "/data/mzy/SEED/",
        "-batch_size", "16",
        "-epochs", "150",
        "-lr", "0.001",
        "-seed", "2024",
        "-device", "cuda:1",
        "-selected_channels", "0", "1", "2", "3", "4"
    ]
}
```

## 3. 验证步骤

1. 运行命令行代码，确认模型可以正常训练
2. 在VS Code中使用新添加的配置，确认可以正常调试
3. 尝试不同的通道数量，确认模型可以适应不同的通道数量

## 4. 注意事项

- 显卡目前只有cuda:1可用，所以在命令行和配置中都需要指定`-device cuda:1`
- 可以根据需要调整`selected_channels`参数，选择不同的通道数量
- 可以调整其他参数，如batch_size、epochs、lr等，以获得更好的性能

## 5. 预期结果

- 模型可以正常训练，支持不同通道数量的输入
- 在跨个体实验中取得合理的性能
- 可以通过命令行和VS Code调试两种方式运行