from models.Models import Model
from config.setting import seed_sub_independent_leave_one_out_setting, preset_setting, set_setting_by_args
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.args import get_args_parser
from utils.store import make_output_dir
from utils.utils import state_log, result_log, setup_seed, sub_result_log
from Trainer.MsMdaTraining import train
from models.DGCNN import NewSparseL2Regularization
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from data_utils.preprocess import ele_normalize

# run this file with
#    python Msmda_reproduction.py -sessions 1 2 3 -model MS-MDA -batch_size 256 -epochs 200 -lr 0.01 -setting  'seed_sub_independent_leave_one_out_setting' -seed 20 -sr 15
#    0.9397
# python Msmda_train.py -batch_size 256 -epochs 200 -lr 0.01 -setting


def _label_to_index(labels):
    labels = np.asarray(labels)
    if labels.ndim > 1:
        return np.argmax(labels, axis=1)
    return labels


def _as_domain_list(data_parts, label_parts):
    if isinstance(data_parts, np.ndarray):
        data_parts = [data_parts]
        label_parts = [label_parts]
    domains = []
    for data_part, label_part in zip(data_parts, label_parts):
        data_part = np.asarray(data_part)
        label_part = _label_to_index(label_part)
        if data_part.size == 0 or label_part.size == 0:
            continue
        n_samples = min(data_part.shape[0], label_part.shape[0])
        if n_samples == 0:
            continue
        data_part = ele_normalize(data_part[:n_samples]).reshape(n_samples, -1)
        label_part = label_part[:n_samples]
        domains.append((data_part, label_part))
    return domains


def _flatten_domains(domains):
    if len(domains) == 0:
        raise ValueError("MsMda received no non-empty domains after split.")
    data = np.concatenate([domain_data for domain_data, _ in domains], axis=0)
    label = np.concatenate([domain_label for _, domain_label in domains], axis=0)
    return data, label


def main(args):
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
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, keep_dim=True)

            if len(val_data) == 0:
                val_data = test_data.copy()
                val_label = test_label.copy()

            train_domains = _as_domain_list(train_data, train_label)
            val_domains = _as_domain_list(val_data, val_label)
            test_domains = _as_domain_list(test_data, test_label)
            if len(train_domains) == 0:
                raise ValueError("MsMda received no non-empty source domains after split.")
            samples_source = min(domain_data.shape[0] for domain_data, _ in train_domains)
            datasets_train = [
                torch.utils.data.TensorDataset(torch.Tensor(domain_data), torch.Tensor(domain_label))
                for domain_data, domain_label in train_domains
            ]

            val_data, val_label = _flatten_domains(val_domains)
            test_data, test_label = _flatten_domains(test_domains)

            model = Model['MsMda'](channels, feature_dim, num_classes, number_of_source=len(datasets_train))
            # Train one round using the train one round function defined in the model
            dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss()
            output_dir = make_output_dir(args, "MsMda")
            round_metric = train(model=model, datasets_train=datasets_train, dataset_val=dataset_val, dataset_test=dataset_test, output_dir=output_dir, samples_source=samples_source, device=device, metrics=args.metrics, metric_choose=args.metric_choose, optimizer=optimizer,
                                 batch_size=args.batch_size, epochs=args.epochs, criterion=criterion, test_sub_label=test_sub_label)
            best_metrics.append(round_metric)
    # best metrics: every round metrics dict
    # subjects metrics: (subject, sub_round_metric)
    result_log(args, best_metrics)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # log out train state
    main(args)
