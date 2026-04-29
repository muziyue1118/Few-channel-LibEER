from .Models import Model


_EXPORT_NAMES = {
    "DGCNN": "DGCNN",
    "RGNN": "RGNN_official",
    "EEGNet": "EEGNet",
    "STRNN": "STRNN",
    "GCBNet": "GCBNet",
    "DBN": "DBN",
    "TSception": "TSception",
    "SVM": "svm",
    "CDCN": "CDCN",
    "HSLT": "HSLT",
    "ACRNN": "ACRNN",
    "GCBNet_BLS": "GCBNet_BLS",
    "MSMDA": "MsMda",
}


def __getattr__(name):
    if name in _EXPORT_NAMES:
        return Model[_EXPORT_NAMES[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Model", *_EXPORT_NAMES]
