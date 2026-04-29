from importlib import import_module
from collections.abc import Mapping


_MODEL_SPECS = {
    "DGCNN": ("models.DGCNN", "DGCNN"),
    "CoralDgcnn": ("models.CoralDgcnn", "CoralDgcnn"),
    "DannDgcnn": ("models.DannDgcnn", "DannDgcnn"),
    "R2GSTNN": ("models.R2GSTNN", "R2GSTNN"),
    "BiDANN": ("models.BiDANN", "BiDANN"),
    "RGNN_official": ("models.RGNN_official", "SymSimGCNNet"),
    "GCBNet": ("models.GCBNet", "GCBNet"),
    "GCBNet_BLS": ("models.GCBNet_BLS", "GCBNet_BLS"),
    "CDCN": ("models.CDCN", "CDCN"),
    "DBN": ("models.DBN", "DBN"),
    "STRNN": ("models.STRNN", "STRNN"),
    "EEGNet": ("models.EEGNet", "EEGNet"),
    "HSLT": ("models.HSLT", "HSLT"),
    "ACRNN": ("models.ACRNN", "ACRNN"),
    "TSception": ("models.TSception", "TSception"),
    "MsMda": ("models.MsMda", "MSMDA"),
    "FBSTCNet": ("models.FBSTCNet", "PowerAndConneMixedNet"),
    "NSAL_DGAT": ("models.NSAL_DGAT", "Domain_adaption_model"),
    "PRRL": ("models.PRRL", "PRRL"),
    "svm": ("models.SVM", "SVM"),
}


class LazyModelRegistry(Mapping):
    def __init__(self, specs):
        self._specs = dict(specs)
        self._cache = {}

    def __getitem__(self, name):
        if name not in self._specs:
            raise KeyError(name)
        if name not in self._cache:
            module_name, attr_name = self._specs[name]
            module = import_module(module_name)
            self._cache[name] = getattr(module, attr_name)
        return self._cache[name]

    def __iter__(self):
        return iter(self._specs)

    def __len__(self):
        return len(self._specs)

    def __repr__(self):
        names = ", ".join(self._specs)
        return f"{self.__class__.__name__}([{names}])"


Model = LazyModelRegistry(_MODEL_SPECS)
