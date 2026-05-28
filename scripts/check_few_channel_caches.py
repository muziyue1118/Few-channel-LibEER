#!/usr/bin/env python
"""Validate LibEER few-channel cache files and report their sample shapes."""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np


DEFAULT_CACHE_ROOT = os.environ.get("CACHE_ROOT", "/data/mzy/libeer_few_channel_cache")
SEEDV_CACHE_SETTING = os.environ.get("SEEDV_CACHE_SETTING", "seedv_sub_dependent_train_val_test_setting")

DATASET_CONFIG = {
    ("SEED", "feature_de_lds"): ("seed_de_lds", "seed_sub_dependent_train_val_test_setting"),
    ("SEED", "raw128"): ("seed_raw", "seed_sub_dependent_train_val_test_setting"),
    ("SEEDV", "feature_de_lds"): ("seedv_raw", SEEDV_CACHE_SETTING),
    ("SEEDV", "raw128"): ("seedv_raw", SEEDV_CACHE_SETTING),
    ("FACED", "feature_de_lds"): ("faced_de_lds", "faced_sub_independent_train_val_test_setting"),
}

EXPECTED_CHANNELS = {
    "2ch": 2,
    "4ch": 4,
    "8ch": 8,
}


def parse_args():
    parser = argparse.ArgumentParser("Check LibEER few-channel cache files")
    parser.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--datasets", nargs="+", default=["SEED", "SEEDV", "FACED"])
    parser.add_argument("--profiles", nargs="+", default=["feature_de_lds", "raw128"])
    parser.add_argument("--channels", nargs="+", default=["2ch", "4ch", "8ch"])
    parser.add_argument("--skip_missing", action="store_true")
    return parser.parse_args()


def first_array(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (list, tuple)):
        for item in data:
            found = first_array(item)
            if found is not None:
                return found
    return None


def cache_path(cache_root, dataset_key, profile, channel):
    config = DATASET_CONFIG.get((dataset_key, profile))
    if config is None:
        return None
    dataset, setting = config
    root = Path(cache_root) / dataset_key / dataset / setting
    profiled = root / profile / channel / "libeer_cache.pkl"
    legacy = root / channel / "libeer_cache.pkl"
    if profiled.exists() or profile == "raw128":
        return profiled
    return legacy if legacy.exists() else profiled


def load_metadata(cache_file):
    metadata_file = cache_file.with_name("metadata.json")
    if not metadata_file.exists():
        return {}
    with metadata_file.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def check_one(cache_file, expected_channels):
    with cache_file.open("rb") as fp:
        payload = pickle.load(fp)
    sample = first_array(payload["all_data"])
    if sample is None:
        raise ValueError("no ndarray sample found")
    sample = np.asarray(sample)
    inferred_channels = int(sample.shape[-2]) if sample.ndim >= 2 else None
    payload_channels = int(payload["channels"])
    if payload_channels != expected_channels:
        raise ValueError(f"payload channels={payload_channels}, expected={expected_channels}")
    if inferred_channels != expected_channels:
        raise ValueError(f"sample shape={sample.shape}, shape[-2]={inferred_channels}, expected={expected_channels}")
    return payload, sample, load_metadata(cache_file)


def main():
    args = parse_args()
    failures = []
    print("dataset\tprofile\tchannel\tchannels\tfeature_dim\tsample_shape\tselected_channels\tpath")
    for dataset_key in args.datasets:
        for profile in args.profiles:
            for channel in args.channels:
                expected = EXPECTED_CHANNELS[channel]
                path = cache_path(args.cache_root, dataset_key, profile, channel)
                if path is None:
                    continue
                if not path.exists():
                    message = f"missing {dataset_key}/{profile}/{channel}: {path}"
                    if args.skip_missing:
                        print(f"{dataset_key}\t{profile}\t{channel}\tSKIP\t-\t-\t-\t{path}")
                        continue
                    failures.append(message)
                    print(f"{dataset_key}\t{profile}\t{channel}\tMISSING\t-\t-\t-\t{path}")
                    continue
                try:
                    payload, sample, metadata = check_one(path, expected)
                    selected = ",".join(metadata.get("selected_channels", []))
                    print(
                        f"{dataset_key}\t{profile}\t{channel}\t{payload['channels']}\t"
                        f"{payload['feature_dim']}\t{tuple(sample.shape)}\t{selected}\t{path}"
                    )
                except Exception as exc:
                    failures.append(f"{dataset_key}/{profile}/{channel}: {exc}")
                    print(f"{dataset_key}\t{profile}\t{channel}\tFAIL\t-\t-\t-\t{path}")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
