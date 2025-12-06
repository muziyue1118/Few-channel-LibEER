# 修复RGNN代码中的问题

## 问题分析

1. **通道输入处理问题**：
   - 从`run_rgnn_few_channels.py`的代码中可以看到，当直接运行脚本时，它会使用硬编码的`frontal_channels`列表（FP1, FP2, F3, F4, F7, F8），而忽略命令行传入的`--channel_names`参数
   - 这是因为脚本没有解析命令行参数，而是直接调用了`run_rgnn_few_channels`函数，并传入了硬编码的`frontal_channels`列表
   - 这导致用户无法通过命令行参数来指定要使用的通道

2. **CPU使用率高的问题**：
   - RGNN模型默认使用CPU进行训练（在`run_rgnn_few_channels.py`中设置了`-device cpu`）
   - 在`SymSimGCNNet`类的`forward`方法中，有一些复杂的矩阵操作，特别是在处理邻接矩阵和边权重时：
     - `edge_weight = edge_weight + edge_weight.transpose(1, 0) - torch.diag(edge_weight.diagonal())`
     - `edge_weight = edge_weight.reshape(-1).repeat(batch_size)`
   - 这些操作在CPU上运行可能会导致较高的CPU使用率
   - 另外，`batch_size`设置为1，这可能导致训练过程效率低下

## 解决方案

### 1. 修复通道输入处理问题

在`run_rgnn_few_channels.py`中添加命令行参数解析功能，允许用户通过`--channel_names`参数指定要使用的通道，并使用解析得到的通道列表，而不是硬编码的`frontal_channels`列表。

### 2. 优化CPU使用率问题

- 允许用户选择使用GPU进行训练，而不是默认使用CPU
- 优化模型的`forward`方法，减少不必要的矩阵操作
- 考虑增加`batch_size`，提高训练效率

## 修复步骤

1. **修改`run_rgnn_few_channels.py`脚本**：
   - 添加命令行参数解析功能，支持`--channel_names`参数
   - 使用解析得到的通道列表，而不是硬编码的`frontal_channels`列表
   - 添加`--device`参数，允许用户选择使用GPU还是CPU
   - 添加`--batch_size`参数，允许用户调整batch_size

2. **优化`SymSimGCNNet`类的`forward`方法**：
   - 减少不必要的矩阵操作
   - 优化邻接矩阵和边权重的处理

## 代码修改

### 1. 修改`run_rgnn_few_channels.py`脚本

- 添加`argparse`模块的导入
- 添加命令行参数解析功能
- 使用解析得到的参数调用`run_rgnn_few_channels`函数

### 2. 优化`SymSimGCNNet`类的`forward`方法

- 优化邻接矩阵和边权重的处理逻辑
- 减少不必要的矩阵操作

## 预期效果

1. 修复后，用户可以通过命令行参数`--channel_names`来指定要使用的通道
2. 用户可以通过`--device`参数选择使用GPU还是CPU，从而降低CPU使用率
3. 用户可以通过`--batch_size`参数调整batch_size，提高训练效率
4. 优化后的模型`forward`方法将减少不必要的计算，进一步降低CPU使用率