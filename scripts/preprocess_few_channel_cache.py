#!/usr/bin/env python
"""Create reusable LibEER few-channel caches from full-channel source data."""

import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBEER_DIR = PROJECT_ROOT / "LibEER"
sys.path.insert(0, str(LIBEER_DIR))

from config.setting import preset_setting, set_setting_by_args  # noqa: E402
from data_utils.load_data import extract_dataset, get_uniform_data  # noqa: E402
from data_utils.preprocess import label_process, preprocess  # noqa: E402
from utils.args import get_args_parser  # noqa: E402


CACHE_FILE_NAME = "libeer_cache.pkl"
METADATA_FILE_NAME = "metadata.json"

SEED_CHANNEL_NAME = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4",
    "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3",
    "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ",
    "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
]

SEEDV_18_CHANNEL_NAME = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F3", "FZ", "F4", "F8",
    "FT7", "FC3", "FCZ", "FC4", "FT8", "T7", "CZ", "T8",
]

FACED_CHANNEL_NAME = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4",
    "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
]

STANDARD_2CH = ["FP1", "FP2"]
STANDARD_4CH = STANDARD_2CH + ["T7", "T8"]
STANDARD_8CH_BASE = STANDARD_4CH + ["F7", "F8"]


def build_parser():
    parser = argparse.ArgumentParser(
        "Preprocess full-channel LibEER data and export 2/4/8-channel caches",
        parents=[get_args_parser()],
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["2ch", "4ch", "8ch"],
        choices=["2ch", "4ch", "8ch"],
        help="Few-channel cache variants to export.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root directory for this dataset's cache output.",
    )
    parser.add_argument(
        "--seedv_8ch_fallback",
        default="FC3_FC4",
        choices=["FC3_FC4"],
        help="Fallback pair for SEEDV 8ch when CP5/CP6 and FC5/FC6 are unavailable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing libeer_cache.pkl files.",
    )
    return parser


def dataset_key(dataset):
    dataset_l = dataset.lower()
    if dataset_l.startswith("seediv"):
        return "SEEDIV"
    if dataset_l.startswith("seedv"):
        return "SEEDV"
    if dataset_l.startswith("seed"):
        return "SEED"
    if dataset_l.startswith("faced"):
        return "FACED"
    raise ValueError(
        f"Unsupported dataset '{dataset}'. Few-channel cache generation supports SEED, SEEDIV, SEEDV, and FACED."
    )


def first_array(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (list, tuple)):
        for item in data:
            found = first_array(item)
            if found is not None:
                return found
    return None


def infer_channels_from_processed(data, fallback):
    sample = first_array(data)
    if sample is None:
        return fallback
    sample = np.asarray(sample)
    if sample.ndim < 2:
        return fallback
    return int(sample.shape[-2])


def channel_names_for_dataset(dataset, full_channels):
    key = dataset_key(dataset)
    if key in {"SEED", "SEEDIV"} and full_channels == len(SEED_CHANNEL_NAME):
        return SEED_CHANNEL_NAME
    if key == "SEEDV":
        if full_channels == len(SEED_CHANNEL_NAME):
            return SEED_CHANNEL_NAME
        if full_channels == len(SEEDV_18_CHANNEL_NAME):
            return SEEDV_18_CHANNEL_NAME
    if key == "FACED" and full_channels == len(FACED_CHANNEL_NAME):
        return FACED_CHANNEL_NAME
    raise ValueError(
        f"Cannot infer channel names for dataset={dataset}, channels={full_channels}. "
        "Add its channel order before exporting few-channel caches."
    )


def require_channels(source_names, wanted_names, dataset, channel):
    missing = [name for name in wanted_names if name not in source_names]
    if missing:
        raise ValueError(
            f"{dataset} {channel} cannot be exported; missing channels: {missing}. "
            f"Available channels: {source_names}"
        )
    return [source_names.index(name) for name in wanted_names]


def names_for_channel_set(dataset, source_names, channel, seedv_8ch_fallback):
    key = dataset_key(dataset)
    if channel == "2ch":
        return STANDARD_2CH
    if channel == "4ch":
        return STANDARD_4CH
    if channel != "8ch":
        raise ValueError(f"Unsupported channel set '{channel}'")

    if key == "FACED":
        preferred_pairs = [["FC5", "FC6"], ["CP5", "CP6"]]
    elif key == "SEEDV":
        preferred_pairs = [["CP5", "CP6"], ["FC5", "FC6"]]
        if seedv_8ch_fallback == "FC3_FC4":
            preferred_pairs.append(["FC3", "FC4"])
    else:
        preferred_pairs = [["CP5", "CP6"], ["FC5", "FC6"]]

    for pair in preferred_pairs:
        wanted = STANDARD_8CH_BASE + pair
        if all(name in source_names for name in wanted):
            return wanted

    raise ValueError(
        f"{dataset} 8ch cannot be exported from this source. Need {STANDARD_8CH_BASE} plus "
        "CP5/CP6, FC5/FC6, or the configured SEEDV FC3/FC4 fallback."
    )


def crop_processed(data, channel_indices):
    if isinstance(data, np.ndarray):
        arr = np.asarray(data)
        if arr.ndim < 2:
            raise ValueError(f"Cannot crop channel axis from sample with shape {arr.shape}")
        if max(channel_indices) >= arr.shape[-2]:
            raise ValueError(
                f"Channel index {max(channel_indices)} is out of bounds for sample shape {arr.shape}"
            )
        return np.take(arr, channel_indices, axis=-2)
    if isinstance(data, list):
        return [crop_processed(item, channel_indices) for item in data]
    if isinstance(data, tuple):
        return tuple(crop_processed(item, channel_indices) for item in data)
    return data


def setting_name(args):
    return args.setting if args.setting is not None else "custom_setting"


def output_dir_for(args, channel):
    return Path(args.output_root) / args.dataset / setting_name(args) / channel


def metadata_for(args, setting, dataset_key_value, full_channels, source_names, selected_names, selected_indices):
    return {
        "format": "libeer_few_channel_cache",
        "version": 1,
        "dataset_key": dataset_key_value,
        "dataset": args.dataset,
        "setting": setting_name(args),
        "source_dataset_path": args.dataset_path,
        "full_channels": int(full_channels),
        "source_channel_names": source_names,
        "selected_channels": selected_names,
        "selected_indices": [int(idx) for idx in selected_indices],
        "preprocess": {
            "pass_band": setting.pass_band,
            "extract_bands": setting.extract_bands,
            "sample_length": setting.sample_length,
            "stride": setting.stride,
            "time_window": setting.time_window,
            "overlap": setting.overlap,
            "only_seg": setting.only_seg if setting.dataset not in extract_dataset else True,
            "feature_type": setting.feature_type,
            "eog_clean": setting.eog_clean,
            "bounds": setting.bounds,
            "onehot": setting.onehot,
            "label_used": setting.label_used,
        },
    }


def write_cache(cache_dir, cache_payload, metadata, overwrite):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / CACHE_FILE_NAME
    metadata_path = cache_dir / METADATA_FILE_NAME
    if cache_path.exists() and not overwrite:
        raise FileExistsError(f"{cache_path} already exists. Pass --overwrite to replace it.")

    with cache_path.open("wb") as cache_file:
        pickle.dump(cache_payload, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)
        metadata_file.write("\n")
    return cache_path


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    print(f"Reading full-channel source: dataset={setting.dataset} path={setting.dataset_path}")
    data, baseline, label, sample_rate, full_channels = get_uniform_data(setting.dataset, setting.dataset_path)
    print("Running LibEER preprocessing once on the full-channel source...")
    all_data, feature_dim = preprocess(
        data=data,
        baseline=baseline,
        sample_rate=sample_rate,
        pass_band=setting.pass_band,
        extract_bands=setting.extract_bands,
        sample_length=setting.sample_length,
        stride=setting.stride,
        time_window=setting.time_window,
        overlap=setting.overlap,
        only_seg=setting.only_seg if setting.dataset not in extract_dataset else True,
        feature_type=setting.feature_type,
        eog_clean=setting.eog_clean,
    )

    full_channels = infer_channels_from_processed(all_data, full_channels)
    source_names = channel_names_for_dataset(setting.dataset, full_channels)
    key = dataset_key(setting.dataset)
    exported = []

    for channel in args.channels:
        selected_names = names_for_channel_set(
            setting.dataset,
            source_names,
            channel,
            args.seedv_8ch_fallback,
        )
        selected_indices = require_channels(source_names, selected_names, setting.dataset, channel)
        print(f"Exporting {channel}: {selected_names} -> indices {selected_indices}")

        cropped_data = crop_processed(all_data, selected_indices)
        cropped_channels = infer_channels_from_processed(cropped_data, len(selected_indices))
        if cropped_channels != len(selected_indices):
            raise RuntimeError(
                f"{channel} crop produced {cropped_channels} channels, expected {len(selected_indices)}"
            )

        cropped_data, cropped_label, num_classes = label_process(
            data=cropped_data,
            label=copy.deepcopy(label),
            bounds=setting.bounds,
            onehot=setting.onehot,
            label_used=setting.label_used,
        )
        metadata = metadata_for(
            args=args,
            setting=setting,
            dataset_key_value=key,
            full_channels=full_channels,
            source_names=source_names,
            selected_names=selected_names,
            selected_indices=selected_indices,
        )
        cache_payload = {
            "format": "libeer_few_channel_cache",
            "version": 1,
            "all_data": cropped_data,
            "all_label": cropped_label,
            "channels": len(selected_indices),
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "metadata": metadata,
        }
        cache_path = write_cache(output_dir_for(args, channel), cache_payload, metadata, args.overwrite)
        exported.append(str(cache_path))

    print("Finished exporting LibEER few-channel caches:")
    for cache_path in exported:
        print(f"  {cache_path}")


if __name__ == "__main__":
    main()
