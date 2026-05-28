#!/usr/bin/env python3
"""Export few-channel LibEER experiment results into flat tables.

The script is intentionally log-based: it does not need the original EEG data.
It reads runs_all_datasets_few_channels/{master_summary.tsv,worker_runs} and
emits complete job/unit matrices. Failed or not-yet-run jobs are kept with empty
metric cells so downstream analysis can see exactly what is missing.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


FEATURE_NETWORKS = [
    "DGCNN",
    "CoralDgcnn",
    "GCBNet",
    "GCBNet_BLS",
    "CDCN",
    "DBN",
    "HSLT",
    "BiDANN",
    "R2GSTNN",
    "NSAL_DGAT",
    "RGNN_official",
    "MsMda",
    "PRRL",
]
RAW_NETWORKS = ["EEGNet", "TSception", "ACRNN", "FBSTCNet"]
DEFAULT_CHANNELS = ["2ch", "4ch", "8ch"]
DEFAULT_SEEDS = ["2024"]

DATASET_PROFILES = {
    "SEED": {
        "feature_de_lds": ("seed_de_lds", "seed_sub_dependent_train_val_test_setting"),
        "raw128": ("seed_raw", "seed_sub_dependent_train_val_test_setting"),
    },
    "SEEDV": {
        "feature_de_lds": ("seedv_raw", "seedv_sub_independent_loso_train_val_test_setting"),
        "raw128": ("seedv_raw", "seedv_sub_independent_loso_train_val_test_setting"),
    },
    "FACED": {
        "feature_de_lds": ("faced_de_lds", "faced_sub_independent_train_val_test_setting"),
    },
}

# Current few-channel settings keep SEED as subject-dependent rounds across
# sessions, while SEED-V uses 16 LOSO folds. FACED is a single subject-
# independent split.
DEFAULT_UNIT_COUNTS = {"SEED": 45, "SEEDV": 16, "FACED": 1}

METRIC_NAME_MAP = {
    "macro-f1": "macro_f1",
    "macro_f1": "macro_f1",
    "acc": "acc",
}

ERROR_PATTERNS = [
    "Traceback",
    "ModuleNotFoundError",
    "ImportError",
    "RuntimeError",
    "ValueError",
    "IndexError",
    "AssertionError",
    "FileNotFoundError",
    "CUDA out of memory",
    "OutOfMemoryError",
]


@dataclass(frozen=True)
class JobKey:
    dataset_key: str
    profile: str
    channel: str
    seed: str
    network: str


@dataclass
class JobRecord:
    key: JobKey
    dataset: str = ""
    setting: str = ""
    status: str = "pending"
    exit_code: str = ""
    seconds: str = ""
    gpu: str = ""
    time: str = ""
    cache_path: str = ""
    launcher_log: str = ""
    worker_run_root: str = ""
    failure_hint: str = ""
    source: str = "expected"


@dataclass
class UnitResult:
    unit_index: int
    unit_label: str
    unit_type: str
    metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    source: str = ""


def parse_words(value: str | None, default: list[str]) -> list[str]:
    if value is None or not value.strip():
        return list(default)
    return [item for item in value.split() if item]


def norm_metric_name(name: str) -> str:
    return METRIC_NAME_MAP.get(name.strip(), name.strip().replace("-", "_"))


def parse_unit_counts(items: Iterable[str]) -> dict[str, int]:
    result = dict(DEFAULT_UNIT_COUNTS)
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--unit-count expects DATASET=N, got: {item}")
        key, value = item.split("=", 1)
        result[key.strip().upper()] = int(value)
    return result


def parse_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fp:
        return list(csv.DictReader(fp, delimiter="\t"))


def infer_profile(row: dict[str, str]) -> str:
    profile = row.get("cache_profile") or row.get("profile") or ""
    if profile:
        return profile

    worker_root = row.get("worker_run_root", "")
    parts = Path(worker_root).parts
    if "worker_runs" in parts:
        idx = parts.index("worker_runs")
        # New layout: worker_runs/DATASET/profile/channel/network/seed
        if len(parts) > idx + 5 and parts[idx + 2] in {"feature_de_lds", "raw128"}:
            return parts[idx + 2]

    network = row.get("network", "")
    dataset = row.get("dataset", "")
    if network in RAW_NETWORKS or dataset in {"seed_raw"}:
        return "raw128"
    return "feature_de_lds"


def latest_master_rows(summary_path: Path) -> dict[JobKey, JobRecord]:
    latest: dict[JobKey, JobRecord] = {}
    for row in parse_tsv(summary_path):
        dataset_key = row.get("dataset_key", "")
        channel = row.get("channel", "")
        seed = row.get("seed", "")
        network = row.get("network", "")
        if not (dataset_key and channel and seed and network):
            continue
        profile = infer_profile(row)
        key = JobKey(dataset_key, profile, channel, seed, network)
        dataset = row.get("dataset", "")
        setting = DATASET_PROFILES.get(dataset_key, {}).get(profile, ("", ""))[1]
        latest[key] = JobRecord(
            key=key,
            dataset=dataset,
            setting=setting,
            status=row.get("status", ""),
            exit_code=row.get("exit_code", ""),
            seconds=row.get("seconds", ""),
            gpu=row.get("gpu", ""),
            time=row.get("time", ""),
            cache_path=row.get("cache_path", ""),
            launcher_log=row.get("launcher_log", ""),
            worker_run_root=row.get("worker_run_root", ""),
            source="master_summary",
        )
    return latest


def expected_jobs(
    run_root: Path,
    dataset_keys: list[str],
    channels: list[str],
    seeds: list[str],
    networks: list[str],
    include_designed_skips: bool = False,
) -> dict[JobKey, JobRecord]:
    jobs: dict[JobKey, JobRecord] = {}
    for dataset_key in dataset_keys:
        profiles = DATASET_PROFILES.get(dataset_key, {})
        for profile, (dataset, setting) in profiles.items():
            profile_networks = RAW_NETWORKS if profile == "raw128" else FEATURE_NETWORKS
            for network in networks:
                if network not in profile_networks:
                    continue
                for channel in channels:
                    for seed in seeds:
                        key = JobKey(dataset_key, profile, channel, seed, network)
                        worker = run_root / "worker_runs" / dataset_key / profile / channel / network / f"seed{seed}"
                        jobs[key] = JobRecord(
                            key=key,
                            dataset=dataset,
                            setting=setting,
                            worker_run_root=str(worker),
                        )
        if include_designed_skips and dataset_key == "FACED":
            for network in networks:
                if network not in RAW_NETWORKS:
                    continue
                for channel in channels:
                    for seed in seeds:
                        key = JobKey(dataset_key, "raw128", channel, seed, network)
                        worker = run_root / "worker_runs" / dataset_key / "raw128" / channel / network / f"seed{seed}"
                        jobs[key] = JobRecord(
                            key=key,
                            dataset="",
                            setting="faced_sub_independent_train_val_test_setting",
                            status="designed_skip",
                            worker_run_root=str(worker),
                            failure_hint="FACED raw128 cache/profile was intentionally skipped by the runner.",
                        )
    return jobs


def read_text(path: Path, max_bytes: int | None = None) -> str:
    try:
        if max_bytes is None:
            return path.read_text(encoding="utf-8", errors="replace")
        with path.open("rb") as fp:
            data = fp.read(max_bytes)
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def find_failure_hint(job: JobRecord) -> str:
    candidates: list[Path] = []
    if job.launcher_log:
        candidates.append(Path(job.launcher_log))
    worker = Path(job.worker_run_root)
    candidates.extend(sorted((worker / "logs").glob("*.log")))

    for path in candidates:
        text = read_text(path)
        if not text:
            continue
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if any(pattern in line for pattern in ERROR_PATTERNS):
                detail = line.strip()
                if idx + 1 < len(lines) and "Traceback" not in detail:
                    return detail[:500]
                if idx + 1 < len(lines):
                    return (detail + " " + lines[idx + 1].strip())[:500]
                return detail[:500]
    return ""


def parse_state_logs(worker_root: Path) -> list[UnitResult]:
    results: list[UnitResult] = []
    state_dir = worker_root / "state_logs"
    if not state_dir.exists():
        return results

    def unit_sort_key(item: tuple[object, object]) -> tuple[int, str]:
        label = str(item[0])
        match = re.search(r"\d+", label)
        if match:
            return int(match.group(0)), label
        return 10**9, label

    for path in sorted(state_dir.glob("*"), key=lambda p: p.stat().st_mtime if p.exists() else 0):
        if not path.is_file():
            continue
        for raw_line in read_text(path).splitlines():
            line = raw_line.strip()
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                value = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            if all(isinstance(v, dict) for v in value.values()):
                for idx, (unit_label, metrics) in enumerate(sorted(value.items(), key=unit_sort_key), 1):
                    clean = {
                        norm_metric_name(k): float(v)
                        for k, v in metrics.items()
                        if isinstance(v, (int, float))
                    }
                    if clean:
                        results.append(
                            UnitResult(
                                unit_index=idx,
                                unit_label=str(unit_label),
                                unit_type="subject",
                                metrics=clean,
                                source=str(path),
                            )
                        )
    return results


def parse_result_table(text: str, source: str) -> list[UnitResult]:
    lines = text.splitlines()
    results: list[UnitResult] = []
    for idx, line in enumerate(lines):
        if "Result" not in line or "|" not in line:
            continue
        headers = [part.strip() for part in line.split("|") if part.strip()]
        if not headers or headers[0].lower() != "result":
            continue
        metric_names = [norm_metric_name(name.replace("_mean", "")) for name in headers[1:]]
        for row_line in lines[idx + 1 :]:
            if not row_line.strip().startswith("|"):
                break
            parts = [part.strip() for part in row_line.split("|") if part.strip()]
            if not parts:
                continue
            if not re.fullmatch(r"\d+", parts[0]):
                break
            metrics: dict[str, float] = {}
            for name, value in zip(metric_names, parts[1:]):
                try:
                    metrics[name] = float(value)
                except ValueError:
                    pass
            if metrics:
                unit_index = int(parts[0])
                results.append(
                    UnitResult(
                        unit_index=unit_index,
                        unit_label=str(unit_index),
                        unit_type="fold",
                        metrics=metrics,
                        source=source,
                    )
                )
    return results


def parse_best_metric_blocks(text: str, source: str) -> list[UnitResult]:
    results: list[UnitResult] = []
    current_test: dict[str, float] = {}
    current_val: dict[str, float] = {}

    def flush() -> None:
        nonlocal current_test, current_val
        if current_test or current_val:
            idx = len(results) + 1
            results.append(
                UnitResult(
                    unit_index=idx,
                    unit_label=str(idx),
                    unit_type="fold",
                    metrics=current_test,
                    val_metrics=current_val,
                    source=source,
                )
            )
        current_test = {}
        current_val = {}

    for line in text.splitlines():
        if line.startswith("train indexes:") or line.startswith("Command:"):
            flush()
            continue
        match = re.search(r"best_(val|test)_([A-Za-z0-9_\-]+):\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", line)
        if not match:
            continue
        split, metric, value = match.groups()
        metric_name = norm_metric_name(metric)
        if split == "val":
            current_val[metric_name] = float(value)
        else:
            current_test[metric_name] = float(value)
    flush()
    return results


def parse_worker_results(worker_root: Path) -> tuple[list[UnitResult], str]:
    state_results = parse_state_logs(worker_root)
    if state_results:
        return state_results, "state_logs"

    log_paths = sorted((worker_root / "logs").glob("*.log"))
    if not log_paths:
        return [], "missing_logs"

    # Prefer the newest log; retries overwrite the worker summary but may leave
    # multiple log files if command names changed.
    log_path = max(log_paths, key=lambda p: p.stat().st_mtime)
    text = read_text(log_path)
    table_results = parse_result_table(text, str(log_path))
    if table_results:
        return table_results, "result_table"
    block_results = parse_best_metric_blocks(text, str(log_path))
    if block_results:
        return block_results, "best_metric_blocks"
    return [], "no_metrics_found"


def unit_type_for_job(key: JobKey) -> str:
    if key.dataset_key == "SEED":
        return "subject"
    return "fold"


def complete_units(
    job: JobRecord,
    parsed: list[UnitResult],
    unit_count: int,
    parse_source: str,
) -> list[UnitResult]:
    if parsed:
        target_unit_type = unit_type_for_job(job.key)
        for unit in parsed:
            unit.unit_type = target_unit_type
        by_index = {unit.unit_index: unit for unit in parsed}
        complete = []
        for idx in range(1, max(unit_count, max(by_index)) + 1):
            if idx in by_index:
                complete.append(by_index[idx])
            else:
                complete.append(
                    UnitResult(
                        unit_index=idx,
                        unit_label=str(idx),
                        unit_type=unit_type_for_job(job.key),
                        source=f"placeholder_missing_from_{parse_source}",
                    )
                )
        return complete

    return [
        UnitResult(
            unit_index=idx,
            unit_label=str(idx),
            unit_type=unit_type_for_job(job.key),
            source=parse_source,
        )
        for idx in range(1, unit_count + 1)
    ]


def metric_values(units: list[UnitResult], metric: str) -> list[float]:
    values = []
    for unit in units:
        value = unit.metrics.get(metric)
        if isinstance(value, (int, float)) and not math.isnan(value):
            values.append(float(value))
    return values


def sanity_warning_for_job(
    job: JobRecord,
    units: list[UnitResult],
    expected_units: int,
    parse_source: str,
) -> str:
    if job.key.dataset_key != "SEEDV":
        return ""
    warnings = []
    found_units = len([unit for unit in units if unit.metrics])
    if found_units != expected_units:
        warnings.append(f"found {found_units}/{expected_units} metric units")
    acc_values = metric_values(units, "acc")
    if acc_values:
        acc_mean = mean(acc_values)
        zero_units = sum(1 for value in acc_values if abs(value) < 1e-12)
        if acc_mean < 0.2:
            warnings.append(f"mean acc {acc_mean:.3f} below 5-class chance 0.200")
        if zero_units > max(1, len(acc_values) // 2):
            warnings.append(f"{zero_units}/{len(acc_values)} zero-accuracy units")
    if job.status == "ok" and parse_source in {"missing_logs", "no_metrics_found"}:
        warnings.append(f"ok job but metrics were not parsed from {parse_source}")
    return "; ".join(warnings)


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_write_xlsx(path: Path, sheets: dict[str, tuple[list[dict[str, str]], list[str]]]) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("pandas is not installed; skip xlsx export.")
        return
    with pd.ExcelWriter(path) as writer:
        for sheet_name, (rows, fieldnames) in sheets.items():
            pd.DataFrame(rows, columns=fieldnames).to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs_all_datasets_few_channels")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--datasets", default="SEED SEEDV FACED")
    parser.add_argument("--channels", default="2ch 4ch 8ch")
    parser.add_argument("--seeds", default="2024")
    parser.add_argument("--networks", default=" ".join(FEATURE_NETWORKS + RAW_NETWORKS))
    parser.add_argument("--metrics", default="acc macro-f1")
    parser.add_argument("--unit-count", action="append", default=[], help="Override expected rows, e.g. SEED=15")
    parser.add_argument("--include-designed-skips", action="store_true", help="Also include intentionally skipped FACED raw/time jobs.")
    parser.add_argument("--xlsx", action="store_true", help="Also write few_channel_results.xlsx if pandas is available.")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_root / "extracted_results"
    dataset_keys = parse_words(args.datasets, ["SEED", "SEEDV", "FACED"])
    channels = parse_words(args.channels, DEFAULT_CHANNELS)
    seeds = parse_words(args.seeds, DEFAULT_SEEDS)
    networks = parse_words(args.networks, FEATURE_NETWORKS + RAW_NETWORKS)
    metrics = [norm_metric_name(metric) for metric in parse_words(args.metrics, ["acc", "macro-f1"])]
    unit_counts = parse_unit_counts(args.unit_count)

    jobs = expected_jobs(run_root, dataset_keys, channels, seeds, networks, args.include_designed_skips)
    latest = latest_master_rows(run_root / "master_summary.tsv")
    for key, record in latest.items():
        if key in jobs:
            jobs[key] = record

    job_rows: list[dict[str, str]] = []
    unit_rows: list[dict[str, str]] = []
    status_counts = defaultdict(int)

    for key in sorted(jobs, key=lambda k: (k.dataset_key, k.profile, k.channel, k.network, k.seed)):
        job = jobs[key]
        if job.status in {"failed", "pending", ""}:
            job.failure_hint = find_failure_hint(job)
        worker_root = Path(job.worker_run_root) if job.worker_run_root else (
            run_root / "worker_runs" / key.dataset_key / key.profile / key.channel / key.network / f"seed{key.seed}"
        )
        parsed_units: list[UnitResult] = []
        parse_source = "not_run"
        if job.status == "ok":
            parsed_units, parse_source = parse_worker_results(worker_root)
        elif job.status == "":
            job.status = "pending"

        expected_units = unit_counts.get(key.dataset_key, 1)
        units = complete_units(job, parsed_units, expected_units, parse_source)
        sanity_warning = sanity_warning_for_job(job, units, expected_units, parse_source)
        status_counts[job.status] += 1

        job_row = {
            "dataset_key": key.dataset_key,
            "dataset": job.dataset,
            "profile": key.profile,
            "channel": key.channel,
            "seed": key.seed,
            "network": key.network,
            "status": job.status,
            "exit_code": job.exit_code,
            "seconds": job.seconds,
            "gpu": job.gpu,
            "time": job.time,
            "unit_count_expected": str(expected_units),
            "unit_count_found": str(len([u for u in units if u.metrics])),
            "parse_source": parse_source,
            "sanity_warning": sanity_warning,
            "failure_hint": job.failure_hint,
            "launcher_log": job.launcher_log,
            "worker_run_root": str(worker_root),
        }
        for metric in metrics:
            values = metric_values(units, metric)
            job_row[f"{metric}_mean"] = fmt_float(mean(values) if values else None)
            job_row[f"{metric}_std"] = fmt_float(pstdev(values) if len(values) > 1 else (0.0 if len(values) == 1 else None))
        job_rows.append(job_row)

        for unit in units:
            row = {
                "dataset_key": key.dataset_key,
                "dataset": job.dataset,
                "profile": key.profile,
                "channel": key.channel,
                "seed": key.seed,
                "network": key.network,
                "status": job.status,
                "exit_code": job.exit_code,
                "unit_type": unit.unit_type,
                "unit_index": str(unit.unit_index),
                "unit_label": unit.unit_label,
                "parse_source": unit.source or parse_source,
                "failure_hint": job.failure_hint,
                "worker_run_root": str(worker_root),
            }
            for metric in metrics:
                row[f"test_{metric}"] = fmt_float(unit.metrics.get(metric))
                row[f"val_{metric}"] = fmt_float(unit.val_metrics.get(metric))
            unit_rows.append(row)

    metric_summary_rows: list[dict[str, str]] = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in unit_rows:
        grouped[(row["dataset_key"], row["profile"], row["channel"], row["network"])].append(row)
    for (dataset_key, profile, channel, network), rows in sorted(grouped.items()):
        ok_rows = [row for row in rows if row["status"] == "ok"]
        summary = {
            "dataset_key": dataset_key,
            "profile": profile,
            "channel": channel,
            "network": network,
            "ok_unit_rows": str(len(ok_rows)),
            "total_unit_rows": str(len(rows)),
        }
        for metric in metrics:
            values = [float(row[f"test_{metric}"]) for row in ok_rows if row.get(f"test_{metric}")]
            summary[f"{metric}_mean"] = fmt_float(mean(values) if values else None)
            summary[f"{metric}_std"] = fmt_float(pstdev(values) if len(values) > 1 else (0.0 if len(values) == 1 else None))
        metric_summary_rows.append(summary)

    job_fields = [
        "dataset_key", "dataset", "profile", "channel", "seed", "network",
        "status", "exit_code", "seconds", "gpu", "time",
        "unit_count_expected", "unit_count_found", "parse_source",
        *[f"{metric}_mean" for metric in metrics],
        *[f"{metric}_std" for metric in metrics],
        "sanity_warning", "failure_hint", "launcher_log", "worker_run_root",
    ]
    unit_fields = [
        "dataset_key", "dataset", "profile", "channel", "seed", "network",
        "status", "exit_code", "unit_type", "unit_index", "unit_label",
        *[f"test_{metric}" for metric in metrics],
        *[f"val_{metric}" for metric in metrics],
        "parse_source", "failure_hint", "worker_run_root",
    ]
    summary_fields = [
        "dataset_key", "profile", "channel", "network", "ok_unit_rows", "total_unit_rows",
        *[f"{metric}_mean" for metric in metrics],
        *[f"{metric}_std" for metric in metrics],
    ]

    write_tsv(output_dir / "few_channel_job_summary.tsv", job_rows, job_fields)
    write_tsv(output_dir / "few_channel_unit_results.tsv", unit_rows, unit_fields)
    write_tsv(output_dir / "few_channel_metric_summary.tsv", metric_summary_rows, summary_fields)
    write_csv(output_dir / "few_channel_job_summary.csv", job_rows, job_fields)
    write_csv(output_dir / "few_channel_unit_results.csv", unit_rows, unit_fields)
    write_csv(output_dir / "few_channel_metric_summary.csv", metric_summary_rows, summary_fields)
    if args.xlsx:
        maybe_write_xlsx(
            output_dir / "few_channel_results.xlsx",
            {
                "job_summary": (job_rows, job_fields),
                "unit_results": (unit_rows, unit_fields),
                "metric_summary": (metric_summary_rows, summary_fields),
            },
        )

    print(f"Exported results to: {output_dir}")
    print(f"Jobs: {len(job_rows)}")
    print(f"Unit rows: {len(unit_rows)}")
    print("Status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
