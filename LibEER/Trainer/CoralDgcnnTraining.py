import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from utils.metric import Metric
from utils.store import save_state


def _run_model(model, samples):
    outputs = model(samples)
    if isinstance(outputs, dict):
        predicts = outputs.get("predicts")
        if predicts is None:
            raise ValueError("CoralDgcnn 模型需要返回 'predicts' 张量")
        coral_loss = outputs.get("coralLoss")
    else:
        predicts = outputs
        coral_loss = None
    return predicts, coral_loss


def _maybe_cast_targets(targets, predicts):
    """
    训练脚本中标签可能是 float（onehot）或 long，这里根据预测张量进行简单适配。
    """
    if targets.dtype == torch.float32 and predicts.dim() == targets.dim():
        return targets.argmax(dim=1)
    if targets.dtype in (torch.float32, torch.float64):
        return targets.long()
    return targets


def train(model, dataset_train, dataset_val, dataset_test, device, output_dir, metrics=None,
          metric_choose=None, optimizer=None, scheduler=None, batch_size=16, epochs=40,
          criterion=None, loss_func=None, loss_param=None, coral_lambda=1.0, num_workers=4):
    if metrics is None:
        metrics = ['acc']
    if metric_choose is None:
        metric_choose = metrics[0]

    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    sampler_test = SequentialSampler(dataset_test)

    data_loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=num_workers)
    data_loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=batch_size, num_workers=num_workers)
    data_loader_test = DataLoader(dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=num_workers)

    model = model.to(device)
    best_metric = {s: 0. for s in metrics}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        metric = Metric(metrics)

        train_bar = tqdm(enumerate(data_loader_train), total=len(data_loader_train),
                         desc=f"Train Epoch {epoch}/{epochs}: lr:{optimizer.param_groups[0]['lr']}")
        for _, (samples, targets) in train_bar:
            samples = samples.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predicts, coral_loss = _run_model(model, samples)
            targets = _maybe_cast_targets(targets, predicts)
            class_loss = criterion(predicts, targets)
            loss = class_loss
            if coral_loss is not None and coral_lambda:
                loss = loss + coral_lambda * coral_loss
            if loss_func is not None:
                loss = loss + loss_func(loss_param)

            metric.update(torch.argmax(predicts, dim=1), targets, loss.item())
            train_bar.set_postfix_str(f"loss: {loss.item():.2f}")

            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        print("\033[32m train state: " + metric.value())

        metric_value = evaluate(model, data_loader_val, device, metrics, criterion, loss_func,
                                loss_param, coral_lambda)
        for m in metrics:
            if metric_value[m] > best_metric[m]:
                best_metric[m] = metric_value[m]
                save_state(output_dir, model, optimizer, epoch + 1, metric=m)

    checkpoint = torch.load(f"{output_dir}/checkpoint-best{metric_choose}")
    model.load_state_dict(checkpoint['model'])
    metric_value = evaluate(model, data_loader_test, device, metrics, criterion, loss_func,
                            loss_param, coral_lambda)
    for m in metrics:
        print(f"best_val_{m}: {best_metric[m]:.2f}")
        print(f"best_test_{m}: {metric_value[m]:.2f}")
    return metric_value


@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion, loss_func, loss_param, coral_lambda):
    model.eval()
    metric = Metric(metrics)
    for _, (samples, targets) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating : "):
        samples = samples.to(device)
        targets = targets.to(device)

        predicts, coral_loss = _run_model(model, samples)
        targets = _maybe_cast_targets(targets, predicts)
        loss = criterion(predicts, targets)
        if coral_loss is not None and coral_lambda:
            loss = loss + coral_lambda * coral_loss
        if loss_func is not None:
            loss = loss + loss_func(loss_param)

        metric.update(torch.argmax(predicts, dim=1), targets, loss.item())

    print("\033[34m eval state: " + metric.value())
    return metric.values

