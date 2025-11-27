# GCBNet_BLS少通道跨个体实验支持计划

## 问题分析
1. 模型中存在硬编码的全连接层输入维度（1100），无法适应不同通道数
2. 模型需要根据输入通道数自动调整网络结构
3. 数据加载模块已经支持通道选择功能
4. 参数解析器已经支持通道选择参数

## 修复步骤

### 1. 修复模型硬编码问题
- 修改 `GCBNet_BLS.py` 中的全连接层输入维度，使其能够根据通道数自动计算
- 确保所有依赖通道数的层都能正确调整

### 2. 验证模型支持不同通道数
- 确保模型构造函数正确使用 `num_electrodes` 参数
- 检查图卷积层、邻接矩阵等是否正确适应通道数变化

### 3. 编写运行命令
- 编写跨个体实验的命令行代码，支持少通道设置
- 确保命令中包含正确的通道选择参数

### 4. 添加launch.json配置
- 在 `.vscode/launch.json` 中添加GCBNet_BLS少通道跨个体实验的配置
- 支持不同通道数的实验

## 预期结果
- 模型能够正确处理不同通道数的输入
- 可以通过命令行参数选择使用哪些通道
- 跨个体实验能够正常运行
- launch.json中包含可直接运行的配置

## 具体实现细节

### 1. 修复模型硬编码
- 将第63行的 `self.fc = nn.Linear(1100, self.num_classes, bias=True)` 修改为动态计算输入维度
- 计算方式：根据特征节点和增强节点的数量和维度动态计算

### 2. 运行命令示例
```bash
python GCBNet_BLS_train.py -metrics acc macro-f1 -metric_choose macro-f1 -setting seed_sub_independent_train_val_test_setting -dataset_path /data/mzy/SEED/ -dataset seed_de_lds -batch_size 16 -epochs 150 -lr 0.001 -seed 2024 -device cuda:1 -selected_channels 0 1 2 3 4
```

### 3. launch.json配置
添加一个新的配置项，包含必要的参数和通道选择选项

## 测试计划
1. 运行不同通道数的实验，验证模型能够正确处理
2. 检查输出结果是否合理
3. 确保没有运行时错误