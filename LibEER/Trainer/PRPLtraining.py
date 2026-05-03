import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm

from utils.metric import Metric, SubMetric
from utils.store import save_state


def _tensor_dataset(split):
    features = torch.as_tensor(split["feature"], dtype=torch.float32)
    labels = torch.as_tensor(split["label"])
    return TensorDataset(features, labels)


def _target_indices(labels):
    if labels.dim() > 1:
        return torch.argmax(labels, dim=1).long()
    return labels.long().squeeze()


def _target_onehot(labels, num_classes, device):
    if labels.dim() > 1:
        return labels.float().to(device)
    return F.one_hot(labels.long().squeeze(), num_classes=num_classes).float().to(device)


def train_and_test_GAN(model, target_set, validation_set, source_set, test_sub_label=None, device=None,
                       output_dir="result/", metrics=None, metric_choose=None, batch_size=16, epochs=40,
                       lr=0.001, threshold_update=True):
    if metrics is None:
        metrics = ["acc"]
    if metric_choose is None:
        metric_choose = metrics[0]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_train = _tensor_dataset(source_set)
    dataset_val = _tensor_dataset(validation_set)
    dataset_test = _tensor_dataset(target_set)
    data_loader_train = DataLoader(
        dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size, num_workers=4
    )
    data_loader_val = DataLoader(
        dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size, num_workers=4
    )
    data_loader_test = DataLoader(
        dataset_test, sampler=SequentialSampler(dataset_test), batch_size=batch_size, num_workers=4
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    best_metric = {s: -1.0 for s in metrics}

    for epoch in range(epochs):
        model.train()
        metric = Metric(metrics)
        train_bar = tqdm(
            enumerate(data_loader_train),
            total=len(data_loader_train),
            desc=f"Train Epoch {epoch + 1}/{epochs}: lr:{optimizer.param_groups[0]['lr']}",
        )
        for _, (features, labels) in train_bar:
            features = features.to(device)
            labels = labels.to(device)
            label_index = _target_indices(labels)
            label_onehot = _target_onehot(labels, model.num_of_class, device)

            optimizer.zero_grad()
            logits, _, _, _, _ = model(features, features, label_onehot)
            loss = criterion(logits, label_index)
            metric.update(torch.argmax(logits, dim=1), label_index, loss.item())
            train_bar.set_postfix_str(f"loss: {loss.item():.2f}")
            loss.backward()
            optimizer.step()

        print("\033[32m train state: " + metric.value())
        metric_value = evaluate(model, data_loader_val, device, metrics, criterion)
        for m in metrics:
            if metric_value[m] > best_metric[m]:
                best_metric[m] = metric_value[m]
                save_state(output_dir, model, optimizer, epoch + 1, metric=m)
        if threshold_update:
            model.update_threshold(epoch)

    model.load_state_dict(torch.load(f"{output_dir}/checkpoint-best{metric_choose}")["model"])
    metric_value = evaluate(model, data_loader_test, device, metrics, criterion, test_sub_label=test_sub_label)
    for m in metrics:
        print(f"best_val_{m}: {best_metric[m]:.2f}")
        print(f"best_test_{m}: {metric_value[m]:.2f}")
    return metric_value


@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion, test_sub_label=None):
    model.eval()
    metric = Metric(metrics) if test_sub_label is None else SubMetric(metrics)
    sub_labels = None
    if test_sub_label is not None:
        sub_labels = torch.as_tensor(test_sub_label)

    offset = 0
    for _, (features, labels) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating : "):
        features = features.to(device)
        labels = labels.to(device)
        label_index = _target_indices(labels)
        logits = model.predict_logits(features)
        loss = criterion(logits, label_index)
        predictions = torch.argmax(logits, dim=1)
        if sub_labels is None:
            metric.update(predictions, label_index, loss.item())
        else:
            batch_sub_labels = sub_labels[offset:offset + len(features)].to(device)
            offset += len(features)
            metric.update(predictions, label_index, batch_sub_labels, loss.item())

    print("\033[34m eval state: " + metric.value())
    return metric.values
