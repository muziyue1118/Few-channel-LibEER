# 修复RGNN少通道实验运行问题

## 问题分析
从日志文件 `/data/mzy/LibEER/log/few_channels_experiments/20251202_140635/RGNN_4ch.log` 中可以看到错误：
```
/home/mzy/anaconda3/bin/python3: can't open file '/data/mzy/LibEER/RGNN_train.py': [Errno 2] No such file or directory
```

**根本原因**：
1. `run_rgnn_few_channels.py` 脚本在调用 `RGNN_train.py` 时，直接使用了相对路径 `RGNN_train.py`
2. 当 `run_few_channels_experiments.sh` 运行时，它会在 `/data/mzy/LibEER/` 目录下执行 `python3 "$SCRIPT_DIR/$script"`
3. 因此，`run_rgnn_few_channels.py` 会在 `/data/mzy/LibEER/` 目录下执行，而 `RGNN_train.py` 实际位于 `/data/mzy/LibEER/LibEER/` 目录下
4. 这导致系统在错误的位置查找 `RGNN_train.py` 文件

## 解决方案
修改 `/data/mzy/LibEER/LibEER/run_rgnn_few_channels.py` 脚本，让它使用正确的绝对路径来调用 `RGNN_train.py`。

## 修复步骤
1. 获取 `run_rgnn_few_channels.py` 脚本所在目录的绝对路径
2. 将该路径与 `RGNN_train.py` 拼接，形成完整的绝对路径
3. 使用这个完整路径来调用 `RGNN_train.py`

## 代码修改
在 `run_rgnn_few_channels.py` 脚本中，修改 `run_rgnn_few_channels` 函数中的命令构建部分，将：
```python
cmd = [
    sys.executable, 'RGNN_train.py',
    # ... 其他参数
]
```

修改为：
```python
# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
cmd = [
    sys.executable, os.path.join(script_dir, 'RGNN_train.py'),
    # ... 其他参数
]
```

## 预期效果
修复后，`run_rgnn_few_channels.py` 脚本将能够正确找到并调用 `RGNN_train.py` 文件，RGNN的少通道实验将能够正常运行。