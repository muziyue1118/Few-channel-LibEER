import numpy as np
from data_utils.preprocess import normalize
from models.Models import Model
from config.setting import seed_sub_dependent_front_back_setting, preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from utils.channel_selector import apply_channel_name_selection
from Trainer.FBSTCTraining import train as fbstc_train
from Trainer.training import train as general_train
import torch
import torch.optim as optim
import torch.nn as nn


def main(args):
    # 应用通道名称选择转换
    apply_channel_name_selection(args)
    
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)
    setting.onehot = False
    data, label, channels, feature_dim, num_classes = get_data(setting)
    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)
    best_metrics = []
    torch.set_default_tensor_type('torch.FloatTensor')
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
                # extract the subject label for evaluation
                subj_train_data, subj_train_label, subj_val_data, subj_val_label, subj_test_data, subj_test_label = \
                    index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, True)
                test_sub_num = len(subj_test_data)
                test_sub_label = []
                for i in range(test_sub_num):
                    test_sub_count = len(subj_test_data[i])
                    test_sub_label.extend([i + 1 for j in range(test_sub_count)])
                test_sub_label = np.array(test_sub_label)

            # split train and test data by specified experiment mode
            # 对于跨个体实验，我们需要将数据合并为numpy数组
            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, False)
            
            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label
                
            # 确保数据是numpy数组
            if not isinstance(train_data, np.ndarray):
                train_data = np.array(train_data)
                val_data = np.array(val_data)
                test_data = np.array(test_data)
            
            # 通道选择已经在get_data函数中执行过，这里不需要再次执行
            # 打印通道信息
            print(f"当前使用的通道数量: {channels}")
            
            train_data, val_data, test_data = normalize(train_data, val_data, test_data, dim='sample', method="z-score")
            # model to train
            filterRange = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40),
                           (40, 44), (44, 48), (48, 52)]
            freq = 200
            if setting.dataset.startswith("seed") or setting.dataset.startswith("mped"):
                freq = 200
            elif setting.dataset.startswith("hci") or setting.dataset.startswith("deap"):
                freq = 128

            # 根据输入窗口大小动态调整池化参数
            # 确保所有参数都是正数
            pool_time_length = max(2, min(80, feature_dim // 2))
            pool_time_stride = max(1, min(5, pool_time_length // 2))
            final_conv_length = max(2, min(35, feature_dim // 4))
            final_conv_stride = max(1, min(25, final_conv_length // 2))
            
            model = Model[args.model](channels, num_classes, fs=freq, filterRange=filterRange,
                                      input_window_samples=feature_dim, same_filters_for_features=False,
                                      pool_time_length=pool_time_length, pool_time_stride=pool_time_stride,
                                      final_conv_length=final_conv_length, final_conv_stride=final_conv_stride).to(device)
            
            # 将数据转换为float32类型，目标转换为long类型，与模型期望的类型一致
            dataset_train = torch.utils.data.TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_label, dtype=torch.long))
            dataset_val = torch.utils.data.TensorDataset(torch.tensor(val_data, dtype=torch.float32), torch.tensor(val_label, dtype=torch.long))
            dataset_test = torch.utils.data.TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_label, dtype=torch.long))
            lr = args.lr
            weight_decay = 1e-4
            output_dir = make_output_dir(args, args.model)
            
            # 使用FBSTC特定的训练器
            round_metric = fbstc_train(model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device=device
                                 ,output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose, lr = lr, weight_decay=weight_decay,
                                 batch_size=args.batch_size, epochs=args.epochs, n_classes=num_classes, test_sub_label=test_sub_label)
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
    parser = get_args_parser()
    # 设置默认设备为cuda:1
    parser.set_defaults(device='cuda:1')
    # 设置默认SEED数据集路径
    parser.set_defaults(dataset_path='/data/mzy/SEED/')
    args = parser.parse_args()
    # log out train state
    main(args)
