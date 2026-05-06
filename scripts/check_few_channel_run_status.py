#!/usr/bin/env python3
"""Report progress for the few-channel LibEER batch runner.

The script mirrors scripts/run_all_datasets_few_channel_networks.sh so it can
answer three questions while a nohup batch run is still active:

  * how many runnable jobs exist in the current matrix
  * how many are ok / failed / running / pending
  * which jobs failed and where to inspect their logs

It uses run_all_libeer.log to infer the start time of the current nohup run.
That keeps old dry-run or smoke-test rows in master_summary.tsv from polluting
the current progress report.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DATASETS = "SEED SEEDV FACED"
DEFAULT_CHANNELS = "2ch 4ch 8ch"
DEFAULT_SEEDS = "2024"
DEFAULT_GPUS = "0"
DEFAULT_MAX_JOBS_PER_GPU = 1
DEFAULT_NETWORKS = (
    "DGCNN CoralDgcnn GCBNet GCBNet_BLS CDCN DBN HSLT BiDANN R2GSTNN "
    "NSAL_DGAT EEGNet TSception ACRNN RGNN_official MsMda FBSTCNet PRRL"
)

RAW_NETWORKS = {"EEGNet", "TSception", "ACRNN", "FBSTCNet"}
RAW_SUPPORTED_DATASETS = {"SEED", "SEEDV"}
ERROR_PATTERNS = (
    "Traceback",
    "ModuleNotFoundError",
    "ImportError",
    "RuntimeError",
    "ValueError",
    "IndexError",
    "AssertionError",
    "CUDA out of memory",
    "Killed",
    "No such file",
)


@dataclass(frozen=True, order=True)
class JobKey:
    dataset_key: str
    cache_profile: str
    channel: str
    seed: str
    network: str

    def label(self) -> str:
        return (
            f"{self.dataset_key}/{self.cache_profile}/"
            f"{self.channel}/{self.network}/seed{self.seed}"
        )


@dataclass
class SummaryRow:
    time: dt.datetime | None
    key: JobKey
    dataset: str
    gpu: str
    status: str
    exit_code: str
    seconds: int | None
    cache_path: str
    launcher_log: str
    worker_run_root: str
    line_no: int


@dataclass
class LogEvent:
    time: dt.datetime | None
    key: JobKey
    gpu: str
    status: str
    launcher_log: str
    line_no: int


def split_env(name: str, default: str) -> list[str]:
    return [part for part in os.environ.get(name, default).split() if part]


def cache_profile_for_network(network: str) -> str:
    return "raw128" if network in RAW_NETWORKS else "feature_de_lds"


def dataset_name_for_key_profile(dataset_key: str, cache_profile: str) -> str:
    if dataset_key == "SEED" and cache_profile == "feature_de_lds":
        return os.environ.get("SEED_FEATURE_DATASET") or os.environ.get("SEED_DATASET", "seed_de_lds")
    if dataset_key == "SEED" and cache_profile == "raw128":
        return os.environ.get("SEED_RAW_DATASET", "seed_raw")
    if dataset_key == "SEEDV" and cache_profile == "feature_de_lds":
        return os.environ.get("SEEDV_FEATURE_DATASET") or os.environ.get("SEEDV_DATASET", "seedv_raw")
    if dataset_key == "SEEDV" and cache_profile == "raw128":
        return os.environ.get("SEEDV_RAW_DATASET", "seedv_raw")
    if dataset_key == "FACED" and cache_profile == "feature_de_lds":
        return os.environ.get("FACED_FEATURE_DATASET") or os.environ.get("FACED_DATASET", "faced_de_lds")
    if dataset_key == "SEEDIV" and cache_profile == "feature_de_lds":
        return os.environ.get("SEEDIV_DATASET", "seediv_de_lds")
    return ""


def parse_time(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%F %T"):
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def fmt_time(value: dt.datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value else "-"


def parse_seconds(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def parse_since(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise SystemExit(f"Could not parse --since '{value}'. Use 'YYYY-MM-DD HH:MM:SS'.")


def first_timestamp_in_log(run_log: Path) -> dt.datetime | None:
    if not run_log.exists():
        return None
    pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
    with run_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                return parse_time(match.group(1))
    return None


def build_expected_jobs(
    datasets: Iterable[str],
    channels: Iterable[str],
    seeds: Iterable[str],
    networks: Iterable[str],
) -> tuple[list[JobKey], list[JobKey]]:
    expected: list[JobKey] = []
    designed_skips: list[JobKey] = []
    for dataset_key in datasets:
        for channel in channels:
            for seed in seeds:
                for network in networks:
                    cache_profile = cache_profile_for_network(network)
                    key = JobKey(dataset_key, cache_profile, channel, seed, network)
                    if cache_profile == "raw128" and dataset_key not in RAW_SUPPORTED_DATASETS:
                        designed_skips.append(key)
                        continue
                    expected.append(key)
    return expected, designed_skips


def read_master_summary(summary_path: Path, since: dt.datetime | None) -> tuple[dict[JobKey, SummaryRow], list[SummaryRow]]:
    latest: dict[JobKey, SummaryRow] = {}
    unexpected: list[SummaryRow] = []
    if not summary_path.exists():
        return latest, unexpected

    with summary_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            return latest, unexpected

        for line_no, row in enumerate(reader, start=2):
            dataset_key = (row.get("dataset_key") or "").strip()
            channel = (row.get("channel") or "").strip()
            seed = (row.get("seed") or "").strip()
            network = (row.get("network") or "").strip()
            if not (dataset_key and channel and seed and network):
                continue
            cache_profile = (row.get("cache_profile") or "").strip() or cache_profile_for_network(network)
            row_time = parse_time(row.get("time"))
            if since and row_time and row_time < since:
                continue

            key = JobKey(dataset_key, cache_profile, channel, seed, network)
            item = SummaryRow(
                time=row_time,
                key=key,
                dataset=(row.get("dataset") or dataset_name_for_key_profile(dataset_key, cache_profile)).strip(),
                gpu=(row.get("gpu") or "").strip(),
                status=(row.get("status") or "").strip(),
                exit_code=(row.get("exit_code") or "").strip(),
                seconds=parse_seconds(row.get("seconds")),
                cache_path=(row.get("cache_path") or "").strip(),
                launcher_log=(row.get("launcher_log") or row.get("log_file") or "").strip(),
                worker_run_root=(row.get("worker_run_root") or "").strip(),
                line_no=line_no,
            )
            old = latest.get(key)
            if old is None or ((item.time or dt.datetime.min), item.line_no) >= (
                (old.time or dt.datetime.min),
                old.line_no,
            ):
                latest[key] = item
    return latest, unexpected


EVENT_RE = re.compile(
    r"\[(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+"
    r"(?P<kind>START|FINISH)\s+"
    r".*?dataset=(?P<dataset_key>\S+)\s+"
    r".*?profile=(?P<cache_profile>\S+)\s+"
    r".*?channel=(?P<channel>\S+)\s+"
    r".*?seed=(?P<seed>\S+)\s+"
    r".*?network=(?P<network>\S+)"
    r"(?:\s+gpu=(?P<gpu>\S+))?"
    r"(?:.*?status=(?P<status>\S+))?"
    r"(?:.*?log=(?P<log>\S+))?"
)


def read_run_log_events(run_log: Path, since: dt.datetime | None) -> list[LogEvent]:
    events: list[LogEvent] = []
    if not run_log.exists():
        return events
    with run_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            match = EVENT_RE.search(line)
            if not match:
                continue
            event_time = parse_time(match.group("time"))
            if since and event_time and event_time < since:
                continue
            key = JobKey(
                match.group("dataset_key"),
                match.group("cache_profile"),
                match.group("channel"),
                match.group("seed"),
                match.group("network"),
            )
            events.append(
                LogEvent(
                    time=event_time,
                    key=key,
                    gpu=match.group("gpu") or "",
                    status=match.group("status") or match.group("kind").lower(),
                    launcher_log=match.group("log") or "",
                    line_no=line_no,
                )
            )
    return events


def scan_processes() -> tuple[bool, list[str]]:
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid,etime,cmd"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False, []
    lines = []
    launcher_alive = False
    for line in completed.stdout.splitlines():
        if "run_all_datasets_few_channel_networks.sh" in line or "run_all_few_channel_networks.sh" in line:
            if "check_few_channel_run_status.py" in line:
                continue
            lines.append(line.strip())
            if "run_all_datasets_few_channel_networks.sh" in line:
                launcher_alive = True
    return launcher_alive, lines


def read_failure_hint(log_path: str, max_lines: int = 500) -> str:
    if not log_path:
        return ""
    path = Path(log_path)
    if not path.exists():
        return "log file not found"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"could not read log: {exc}"
    tail = lines[-max_lines:]
    matches = [line.strip() for line in tail if any(pattern in line for pattern in ERROR_PATTERNS)]
    if matches:
        return matches[-1][:240]
    nonempty = [line.strip() for line in tail if line.strip()]
    return nonempty[-1][:240] if nonempty else ""


def read_failure_hint_for_row(row: SummaryRow) -> str:
    log_candidates: list[Path] = []
    if row.worker_run_root:
        worker_root = Path(row.worker_run_root)
        if worker_root.exists():
            log_candidates.extend(sorted((worker_root / "logs").glob("*.log")))
            log_candidates.extend(sorted(worker_root.glob("**/*.log")))
    if row.launcher_log:
        log_candidates.append(Path(row.launcher_log))

    seen: set[Path] = set()
    for path in log_candidates:
        if path in seen:
            continue
        seen.add(path)
        hint = read_failure_hint(str(path))
        if hint and hint != "log file not found":
            return hint
    return read_failure_hint(row.launcher_log)


def fmt_duration(seconds: int | float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    widths = [len(header) for header in headers]
    str_rows = [[str(value) for value in row] for row in rows]
    for row in str_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    print("  " + "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  " + "  ".join("-" * widths[idx] for idx in range(len(headers))))
    for row in str_rows:
        print("  " + "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Check LibEER few-channel batch run progress.")
    parser.add_argument("--run-root", default=os.environ.get("RUN_ROOT", str(project_root / "runs_all_datasets_few_channels")))
    parser.add_argument("--run-log", default=str(project_root / "run_all_libeer.log"))
    parser.add_argument("--summary", default=None, help="Defaults to RUN_ROOT/master_summary.tsv.")
    parser.add_argument("--since", default=None, help="Ignore summary/events before this time: 'YYYY-MM-DD HH:MM:SS'.")
    parser.add_argument("--show-failed", type=int, default=30)
    parser.add_argument("--show-running", type=int, default=30)
    parser.add_argument("--show-pending", type=int, default=20)
    parser.add_argument("--tail", type=int, default=12, help="Show last N lines of run log.")
    parser.add_argument(
        "--fail-exit-code",
        action="store_true",
        help="Exit with code 1 when failed jobs are found. By default this report exits 0.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    run_log = Path(args.run_log)
    summary_path = Path(args.summary) if args.summary else run_root / "master_summary.tsv"

    since = parse_since(args.since)
    if since is None:
        since = first_timestamp_in_log(run_log)

    datasets = split_env("DATASETS", DEFAULT_DATASETS)
    channels = split_env("CHANNELS", DEFAULT_CHANNELS)
    seeds = split_env("SEEDS", DEFAULT_SEEDS)
    networks = split_env("NETWORKS", DEFAULT_NETWORKS)
    max_jobs_per_gpu = int(os.environ.get("MAX_JOBS_PER_GPU", DEFAULT_MAX_JOBS_PER_GPU))

    expected, designed_skips = build_expected_jobs(datasets, channels, seeds, networks)
    expected_set = set(expected)

    latest_rows, _ = read_master_summary(summary_path, since)
    events = read_run_log_events(run_log, since)
    launcher_alive, process_lines = scan_processes()
    observed_gpus = sorted({event.gpu for event in events if event.gpu}, key=lambda value: (len(value), value))
    if os.environ.get("GPUS"):
        gpus = split_env("GPUS", DEFAULT_GPUS)
    else:
        gpus = observed_gpus or split_env("GPUS", DEFAULT_GPUS)
    max_parallel = max(1, len(gpus) * max_jobs_per_gpu)

    log_started: dict[JobKey, LogEvent] = {}
    log_finished: dict[JobKey, LogEvent] = {}
    for event in events:
        if event.status == "start":
            log_started[event.key] = event
        else:
            log_finished[event.key] = event

    ok: dict[JobKey, SummaryRow] = {}
    failed: dict[JobKey, SummaryRow] = {}
    other_finished: dict[JobKey, SummaryRow] = {}
    for key, row in latest_rows.items():
        if key not in expected_set:
            continue
        status = row.status.lower()
        if status == "ok" and row.exit_code == "0":
            ok[key] = row
        elif status in {"failed", "error"} or (row.exit_code and row.exit_code != "0"):
            failed[key] = row
        else:
            other_finished[key] = row

    running_keys = {
        key
        for key in log_started
        if key in expected_set and key not in ok and key not in failed and key not in log_finished
    }
    pending_keys = expected_set - set(ok) - set(failed) - running_keys

    finished_count = len(ok) + len(failed) + len(other_finished)
    unfinished_count = len(running_keys) + len(pending_keys)
    progress = (finished_count / len(expected) * 100.0) if expected else 0.0

    ok_durations = [row.seconds for row in ok.values() if row.seconds and row.seconds > 1]
    avg_seconds = sum(ok_durations) / len(ok_durations) if ok_durations else None
    eta_seconds = None
    if avg_seconds and pending_keys:
        eta_seconds = (len(pending_keys) * avg_seconds) / max_parallel

    print("Few-channel LibEER run status")
    print(f"  run_root: {run_root}")
    print(f"  summary : {summary_path} {'(missing)' if not summary_path.exists() else ''}")
    print(f"  run_log : {run_log} {'(missing)' if not run_log.exists() else ''}")
    print(f"  since   : {fmt_time(since)}")
    print(f"  launcher_alive: {'yes' if launcher_alive else 'no'}")
    print()

    print("Overall")
    print(f"  runnable_total       : {len(expected)}")
    print(f"  finished             : {finished_count} ({progress:.1f}%)")
    print(f"  ok                   : {len(ok)}")
    print(f"  failed               : {len(failed)}")
    print(f"  running              : {len(running_keys)}")
    print(f"  pending_not_started  : {len(pending_keys)}")
    print(f"  unfinished           : {unfinished_count}")
    print(f"  designed_skips       : {len(designed_skips)}")
    print(f"  max_parallel         : {max_parallel}")
    if avg_seconds:
        print(f"  avg_ok_job_time      : {fmt_duration(avg_seconds)}")
    if eta_seconds:
        print(f"  rough_pending_eta    : {fmt_duration(eta_seconds)}")
    print()

    grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for key in expected:
        group = (key.dataset_key, key.cache_profile, key.channel)
        if key in ok:
            grouped[group]["ok"] += 1
        elif key in failed:
            grouped[group]["failed"] += 1
        elif key in running_keys:
            grouped[group]["running"] += 1
        else:
            grouped[group]["pending"] += 1
        grouped[group]["total"] += 1

    print("By Dataset/Profile/Channel")
    table_rows = []
    for group in sorted(grouped):
        values = grouped[group]
        table_rows.append(
            [
                group[0],
                group[1],
                group[2],
                values["total"],
                values["ok"],
                values["failed"],
                values["running"],
                values["pending"],
            ]
        )
    print_table(["dataset", "profile", "channel", "total", "ok", "fail", "run", "pend"], table_rows)
    print()

    if failed:
        print(f"Failed Jobs (showing up to {args.show_failed})")
        rows = []
        for key, row in sorted(failed.items(), key=lambda item: (item[1].time or dt.datetime.min, item[0])):
            rows.append(
                [
                    fmt_time(row.time),
                    key.label(),
                    row.gpu,
                    row.exit_code,
                    fmt_duration(row.seconds),
                    read_failure_hint_for_row(row),
                    row.launcher_log,
                ]
            )
            if len(rows) >= args.show_failed:
                break
        print_table(["time", "job", "gpu", "exit", "seconds", "hint", "launcher_log"], rows)
        print()

    if running_keys:
        title = "Running Jobs"
        if not launcher_alive:
            title += " (launcher not detected; these may be stale if the nohup process exited)"
        print(f"{title} (showing up to {args.show_running})")
        rows = []
        for key in sorted(running_keys):
            event = log_started.get(key)
            rows.append([fmt_time(event.time if event else None), key.label(), event.gpu if event else "", event.launcher_log if event else ""])
            if len(rows) >= args.show_running:
                break
        print_table(["start", "job", "gpu", "launcher_log"], rows)
        print()

    if pending_keys:
        print(f"Pending Jobs (showing up to {args.show_pending})")
        rows = []
        for key in sorted(pending_keys):
            rows.append([key.dataset_key, key.cache_profile, key.channel, key.seed, key.network])
            if len(rows) >= args.show_pending:
                break
        print_table(["dataset", "profile", "channel", "seed", "network"], rows)
        print()

    if process_lines:
        print("Matching Processes")
        for line in process_lines[:12]:
            print(f"  {line}")
        if len(process_lines) > 12:
            print(f"  ... {len(process_lines) - 12} more")
        print()

    if run_log.exists() and args.tail > 0:
        print(f"Last {args.tail} Lines Of Run Log")
        try:
            lines = run_log.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-args.tail :]:
                print(f"  {line}")
        except OSError as exc:
            print(f"  could not read run log: {exc}")

    return 1 if failed and args.fail_exit_code else 0


if __name__ == "__main__":
    sys.exit(main())
