from importlib import import_module


_TRAINER_SPECS = {
    "dbnTrain": ("Trainer.DBNTraining", "train"),
    "graphTrain": ("Trainer.graphTraining", "train"),
    "msmdaTrain": ("Trainer.MsMdaTraining", "train"),
    "train": ("Trainer.training", "train"),
    "r2gstnnTrain": ("Trainer.R2GSTNNTraing", "train"),
}


def __getattr__(name):
    if name in _TRAINER_SPECS:
        module_name, attr_name = _TRAINER_SPECS[name]
        return getattr(import_module(module_name), attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_TRAINER_SPECS)
