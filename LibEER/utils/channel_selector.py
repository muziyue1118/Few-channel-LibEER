from __future__ import annotations

"""
将命令行传入的通道名称转换为索引，以复用 selected_channels 逻辑。
"""

from typing import List, Optional

from data_utils.constants.seed import SEED_CHANNEL_NAME
from data_utils.constants.deap import DEAP_CHANNEL_NAME


def _get_channel_name_list(dataset: Optional[str]) -> Optional[List[str]]:
    dataset = (dataset or "").lower()
    if dataset.startswith(("seed", "mped", "faced")):
        return SEED_CHANNEL_NAME
    if dataset.startswith(("deap", "hci")):
        return DEAP_CHANNEL_NAME
    return None


def apply_channel_name_selection(args) -> None:
    """
    根据 -selected_channel_names 参数更新 args.selected_channels。
    CDCN/CoralDGCNN/DBN 等训练脚本共用。
    """
    if not getattr(args, "selected_channel_names", None):
        return
    if getattr(args, "selected_channels", None) is not None:
        raise ValueError("请勿同时使用 -selected_channels 与 -selected_channel_names 参数")
    channel_name_list = _get_channel_name_list(getattr(args, "dataset", None))
    if channel_name_list is None:
        raise ValueError(f"当前数据集 {args.dataset} 暂不支持通道名选择")
    name_to_index = {name.upper(): idx for idx, name in enumerate(channel_name_list)}
    selected_indices: List[int] = []
    for raw_name in args.selected_channel_names:
        normalized = raw_name.upper()
        if normalized not in name_to_index:
            raise ValueError(f"通道 {raw_name} 不存在于数据集 {args.dataset}，可选通道：{channel_name_list}")
        selected_indices.append(name_to_index[normalized])
    args.selected_channels = selected_indices
    print(f"使用通道名称 {args.selected_channel_names} => 索引 {args.selected_channels}")


