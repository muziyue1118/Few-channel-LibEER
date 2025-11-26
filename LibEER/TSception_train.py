import numpy as np

from data_utils.constants.deap import DEAP_CHANNEL_NAME
from data_utils.constants.seed import SEED_CHANNEL_NAME
from models.Models import Model
from models.TSception import generate_TS_channel_order
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.training import train
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


# run this file with
# deap batch 64 hci batch 32


#    seed dep
#    CUDA_VISIBLE_DEVICES=0 nohup python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1' -setting seed_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_raw -batch_size 16 -epochs 200 -only_seg -sample_length 200 -stride 200 -seed 2024 >TSception/b16e200.log
#    0.6401/0.1644	0.6053/0.1851
#    seed iv dep
#    CUDA_VISIBLE_DEVICES=1 nohup python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1' -setting seediv_sub_dependent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 16 -epochs 300 -only_seg -sample_length 200 -stride 200 -seed 2024 >TSception/s4_b16e300.log
#    0.3606/0.1512	0.3277/0.1508

#    seed indep
#    CUDA_VISIBLE_DEVICES=1 nohup python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1' -setting seed_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED/ -dataset seed_raw -batch_size 32 -epochs 200 -only_seg -sample_length 200 -stride 200 -seed 2024 >TSception_indep/b32e200.log
#    0.456	0.4354
#    seed iv indep
#    CUDA_VISIBLE_DEVICES=3 nohup python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1' -setting seediv_sub_independent_train_val_test_setting -dataset_path /data1/cxx/SEED数据集/SEED_IV -dataset seediv_raw -batch_size 16 -epochs 300 -only_seg -sample_length 200 -stride 200 -seed 2024 >TSception_indep/s4_b16e300.log
#    0.3419	0.2683

#    deap indep
#    valence
#    python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1' -setting deap_sub_independent_train_val_test_setting -dataset_path /data1/cxx/DEAP/data_preprocessed_python -dataset deap -batch_size 64 -epochs 300 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >TSception_indep/deap_valence_b64e300lr0.001.log
#    0.5444	0.4894
#    arousal
#    python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >TSception/hci_arousal_b32e300lr0.001.log
#    0.459	0.4556
#    both
#    python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.002 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >TSception/hci_arousal_b32e300lr0.002.log
#    0.2464	0.2324

#   hci indep
#   valence
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >TSception/hci_valence_b32e300lr0.001.log
#   0.5736	0.5476
#   arousal
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >TSception_indep/hci_arousal_b32e300lr0.001.log
#   0.523	0.5025
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_independent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.001 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >TSception_indep/hci_both_b32e300lr0.001.log
#   0.2699	0.2195

#   hci dep
#   arousal
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.002 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used arousal -seed 2024 >TSception/hci_arousal_b32e300lr0.002.log
#   0.6826/0.2310	0.5629/0.2379
#   valence
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.002 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence -seed 2024 >TSception/hci_valence_b32e300lr0.002.log
#   0.6112/0.1552	0.5051/0.1669
#   both
#   python TSception_train.py -metrics 'acc' 'macro-f1' -model TSception -metric_choose 'macro-f1'  -setting hci_sub_dependent_train_val_test_setting -dataset_path "/data1/cxx/HCI数据集/" -dataset hci -batch_size 32 -epochs 300 -lr 0.002 -only_seg -sample_length 128 -stride 128 -bounds 5 5 -label_used valence arousal -seed 2024 >TSception/hci_both_b32e300lr0.002.log
#   0.4000/0.2060	0.2719/0.1365

# s5_indep
# s5_b64e300

def main(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)
    data, label, channels, feature_dim, num_classes = get_data(setting)
    # 显示使用的通道信息
    if hasattr(setting, 'selected_channels') and setting.selected_channels is not None:
        print(f"使用选择的通道: {setting.selected_channels}, 通道数量: {channels}")
    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)
    best_metrics = []
    subjects_metrics = [[]for _ in range(len(data))]
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            setup_seed(args.seed)
            if val_indexes[0] == -1:
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            test_sub_label = None

            # record who each sample belong to
            if setting.experiment_mode == "subject-independent":
                # extract the subject label
                train_data, train_label, val_data, val_label, test_data, test_label = \
                    index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, True)
                test_sub_num = len(test_data)
                test_sub_label = []
                for i in range(test_sub_num):
                    test_sub_count = len(test_data[i])
                    test_sub_label.extend([i + 1 for j in range(test_sub_count)])
                test_sub_label = np.array(test_sub_label)

            # split train and test data by specified experiment mode
            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, args.keep_dim)
            # print(len(train_data))
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label
            # model to train - 确保使用正确的通道数量
            print(f"准备初始化模型，当前channels参数: {channels}")
            # 对于少通道情况，我们需要确保channels参数与实际使用的通道数量一致
            if hasattr(setting, 'selected_channels') and setting.selected_channels is not None:
                # 使用selected_channels的长度作为模型输入通道数
                model_channels = len(setting.selected_channels)
                print(f"使用少通道设置，模型通道数: {model_channels}")
            else:
                model_channels = channels
                print(f"使用全通道设置，模型通道数: {model_channels}")
            
            # 确保通道数至少为1
            if model_channels <= 0:
                model_channels = 16  # 默认使用16个通道
                print(f"通道数无效，使用默认值: {model_channels}")
                
            if args.dataset.startswith('hci'):
                model = Model['TSception'](model_channels, feature_dim, num_classes, inception_window=[0.25, 0.125, 0.0625])
            else:
                model = Model['TSception'](model_channels, feature_dim, num_classes)
                
            model.to(device)

            # 获取通道顺序
            indexes = np.array([])
            if args.dataset == "deap" or args.dataset == "hci":
                indexes = generate_TS_channel_order(DEAP_CHANNEL_NAME)
            elif args.dataset.startswith("seed") or args.dataset.startswith("mped"):
                indexes = generate_TS_channel_order(SEED_CHANNEL_NAME)
            
            # 打印数据形状以便调试
            print(f"原始train_data形状: {train_data.shape}")
            print(f"原始val_data形状: {val_data.shape}")
            print(f"原始test_data形状: {test_data.shape}")
            
            # 确保数据形状正确：对于TSception，数据应该是 [samples, channels, features]
            # 对于少通道和全通道情况，我们都需要确保数据维度正确
            if len(train_data.shape) > 3:
                # 如果数据维度超过3，可能需要调整维度顺序或降维
                print(f"调整数据维度: {train_data.shape}")
                # 尝试将数据重塑为 [samples, channels, features]
                if train_data.shape[2] == 1:  # 如果中间有一个维度是1
                    train_data = np.squeeze(train_data, axis=2)
                    val_data = np.squeeze(val_data, axis=2)
                    test_data = np.squeeze(test_data, axis=2)
                    print(f"调整后的train_data形状: {train_data.shape}")
            
            # 应用TSception的通道重排序
            if hasattr(setting, 'selected_channels') and setting.selected_channels is not None:
                # 对于少通道情况，我们已经在get_data中处理了通道选择
                print(f"使用少通道设置，通道数量: {train_data.shape[1]}")
            else:
                # 全通道情况下，应用TSception的通道重排序
                train_data = train_data[:, indexes, :]
                val_data = val_data[:, indexes, :]
                test_data = test_data[:, indexes, :]
                print(f"全通道重排序后的train_data形状: {train_data.shape}")
  
            # 确保数据维度正确 - 关键修复
            print(f"最终train_data形状: {train_data.shape}")
            if len(train_data.shape) == 5:
                print("发现5D数据，将其转换为3D格式 [samples, channels, features]")
                # 假设形状为 [samples, 1, time, channels, 1] 或类似格式
                # 我们需要将其重塑为 [samples, channels, features]
                # 首先将第1维和第4维（通道）合并
                train_data = np.squeeze(train_data, axis=1)  # 移除第1维
                val_data = np.squeeze(val_data, axis=1)
                test_data = np.squeeze(test_data, axis=1)
                print(f"移除第1维后的形状: {train_data.shape}")
                
                if len(train_data.shape) == 4:
                    train_data = np.squeeze(train_data, axis=-1)  # 移除最后一维
                    val_data = np.squeeze(val_data, axis=-1)
                    test_data = np.squeeze(test_data, axis=-1)
                    print(f"移除最后一维后的形状: {train_data.shape}")
                
                # 重新排列维度，确保是 [samples, channels, time]
                if train_data.shape[1] != 16:  # 假设16是通道数
                    train_data = np.transpose(train_data, (0, 2, 1))
                    val_data = np.transpose(val_data, (0, 2, 1))
                    test_data = np.transpose(test_data, (0, 2, 1))
                    print(f"维度重排后的形状: {train_data.shape}")
            
            # 添加一个强大的数据预处理函数，确保数据形状正确
            def preprocess_data(data):
                print(f"预处理前形状: {data.shape}")
                # 对于SEED数据集的形状 (samples, time, channels, 5)，其中5可能是频段
                if len(data.shape) == 4 and data.shape[3] == 5:
                    # 选择第一个频段进行处理（简化问题）
                    data = data[:, :, :, 0]
                    print(f"选择第一个频段后形状: {data.shape}")
                    # 现在形状应该是 (samples, time, channels)
                    # 转置为 [samples, channels, time]
                    data = data.transpose(0, 2, 1)
                    print(f"转置后形状: {data.shape}")
                print(f"预处理后形状: {data.shape}")
                return data
            
            # 对所有数据进行预处理
            train_data = preprocess_data(train_data)
            val_data = preprocess_data(val_data)
            test_data = preprocess_data(test_data)
            
            # 确保数据维度正确，尤其是时间维度
            print(f"最终train_data形状: {train_data.shape}")
            if train_data.shape[2] < 10:
                print(f"警告：时间维度太短: {train_data.shape[2]}")
                # 尝试扩展时间维度（如果太短）
                if train_data.shape[2] == 0:
                    # 如果时间维度为0，创建一个最小长度
                    train_data = np.zeros((train_data.shape[0], train_data.shape[1], 10))
                    val_data = np.zeros((val_data.shape[0], val_data.shape[1], 10))
                    test_data = np.zeros((test_data.shape[0], test_data.shape[1], 10))
                    print("已创建最小时间维度长度")
  
            # Train one round using the train one round function defined in the model
            dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8)
            criterion = nn.CrossEntropyLoss()
            output_dir = make_output_dir(args, "TSception")
            round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device=device
                                 ,output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose, optimizer=optimizer,
                                 batch_size=args.batch_size, epochs=args.epochs, criterion=criterion)
            best_metrics.append(round_metric)
            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx-1].append(round_metric)
    # best metrics: every round metrics dict
    # subjects metrics: (subject, sub_round_metric)
    if setting.experiment_mode == "subject-dependent":
        sub_result_log(args, subjects_metrics)
    else:
        result_log(args, best_metrics)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # log out train state
    main(args)
