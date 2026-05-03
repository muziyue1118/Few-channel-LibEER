import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from utils.metric import Metric
from utils.store import save_state


def _unpack_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs["predicts"], outputs.get("coralLoss", 0)
    return outputs, 0


def train(model, dataset_train, dataset_val, dataset_test, device, output_dir="result/", metrics=None,
          metric_choose=None, optimizer=None, scheduler=None, batch_size=16, epochs=40,
          criterion=None, loss_func=None, loss_param=None, test_sub_label=None):
    if metrics is None:
        metrics = ["acc"]
    if metric_choose is None:
        metric_choose = metrics[0]

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
    best_metric = {s: -1.0 for s in metrics}
    for epoch in range(epochs):
        model.train()
        metric = Metric(metrics)
        train_bar = tqdm(
            enumerate(data_loader_train),
            total=len(data_loader_train),
            desc=f"Train Epoch {epoch}/{epochs}: lr:{optimizer.param_groups[0]['lr']}",
        )
        for _, (samples, targets) in train_bar:
            samples = samples.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            outputs, coral_loss = _unpack_outputs(model(samples))
            sparse_loss = 0 if loss_func is None else loss_func(loss_param)
            loss = criterion(outputs, targets) + coral_loss + sparse_loss
            metric.update(torch.argmax(outputs, dim=1), targets, loss.item())
            train_bar.set_postfix_str(f"loss: {loss.item():.2f}")

            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        print("\033[32m train state: " + metric.value())
        metric_value = evaluate(model, data_loader_val, device, metrics, criterion, loss_func, loss_param)
        for m in metrics:
            if metric_value[m] > best_metric[m]:
                best_metric[m] = metric_value[m]
                save_state(output_dir, model, optimizer, epoch + 1, metric=m)

    model.load_state_dict(torch.load(f"{output_dir}/checkpoint-best{metric_choose}")["model"])
    metric_value = evaluate(model, data_loader_test, device, metrics, criterion, loss_func, loss_param)
    for m in metrics:
        print(f"best_val_{m}: {best_metric[m]:.2f}")
        print(f"best_test_{m}: {metric_value[m]:.2f}")
    return metric_value


@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion, loss_func, loss_param):
    model.eval()
    metric = Metric(metrics)
    for _, (samples, targets) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating : "):
        samples = samples.to(device)
        targets = targets.to(device)
        outputs, coral_loss = _unpack_outputs(model(samples))
        sparse_loss = 0 if loss_func is None else loss_func(loss_param)
        loss = criterion(outputs, targets) + coral_loss + sparse_loss
        metric.update(torch.argmax(outputs, dim=1), targets, loss.item())

    print("\033[34m eval state: " + metric.value())
    return metric.values
