## 实现计划

### 1. 创建MLP模型文件
- **文件路径**: `/data/mzy/LibEER/LibEER/models/MLP.py`
- **实现内容**: 
  - 定义MLP类，接受通道数、特征维度和类别数作为参数
  - 实现forward方法，支持形状为(batch_size, channels, datapoints)的输入
  - 确保模型能够处理不同数量的通道输入

### 2. 创建MLP训练脚本
- **文件路径**: `/data/mzy/LibEER/LibEER/MLP_train.py`
- **实现内容**: 
  - 参考其他训练脚本（如EEGNet_train.py）
  - 实现main函数，调用通用训练框架
  - 支持少通道输入设置

### 3. 创建少通道运行脚本
- **文件路径**: `/data/mzy/LibEER/LibEER/run_mlp_few_channels.py`
- **实现内容**: 
  - 参考其他少通道运行脚本
  - 设置默认参数，包括默认通道选择
  - 调用MLP_train.py的main函数

### 4. 注册MLP模型
- **文件路径**: `/data/mzy/LibEER/LibEER/models/Models.py`
- **实现内容**: 
  - 导入MLP类
  - 在Model字典中添加MLP条目

### 5. 更新实验脚本
- **文件路径**: `/data/mzy/LibEER/run_few_channels_experiments.sh`
- **实现内容**: 
  - 在NETWORK_ORDER列表中添加"MLP"

### 6. 实现细节
- MLP模型将包含多个全连接层，使用ReLU激活函数
- 输入数据将先展平为(batch_size, channels * datapoints)形状
- 模型将支持不同通道数，自动适应输入形状
- 训练脚本将复用项目现有的训练框架和工具函数

### 7. 测试
- 确保MLP能够在少通道设置下正常运行
- 验证模型能够处理不同数量的通道输入
- 确保与其他模型具有相同的接口和使用方式