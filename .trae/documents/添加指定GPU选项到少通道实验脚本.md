1. 添加命令行参数解析，支持`--gpu`或`-g`选项
2. 修改`run_experiment()`函数，检查是否指定了GPU
3. 如果指定了GPU，直接使用该GPU，跳过空闲GPU检测
4. 否则，继续使用原来的空闲GPU检测机制
5. 更新脚本头部注释，说明新的命令行选项

具体修改点：

* 在脚本开头添加参数解析逻辑

* 修改`run_experiment()`函数中的GPU选择逻辑

* 支持通过命令行指定单个或多个GPU（可选）

* 保持向后兼容性，不指定GPU时仍使用原有空闲GPU检测

修改后的使用方式：

```bash
# 使用默认空闲GPU检测
./run_few_channels_experiments.sh

# 指定使用GPU 0
./run_few_channels_experiments.sh --gpu 0

# 指定使用GPU 1
./run_few_channels_experiments.sh -g 1
```

