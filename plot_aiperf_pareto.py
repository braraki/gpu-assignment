#!/usr/bin/env python3
"""Gather AIPerf runs and save one or more Pareto curves."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


STAT_PRIORITY = {
    "avg": 0,
    "mean": 1,
    "value": 2,
    "median": 3,
    "p50": 4,
    "max": 5,
    "min": 6,
}


@dataclass
class RunSummary:
    series_name: str
    run_name: str
    concurrency: int
    tokens_per_s_per_user: float
    tokens_per_s_per_gpu: float
    output_token_throughput: float
    ttft_ms: float | None
    itl_ms: float | None
    request_latency_ms: float | None
    json_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect AIPerf run directories, write a summary CSV, "
            "and save a Pareto plot."
        )
    )
    parser.add_argument(
        "--results-root",
        required=True,
        type=Path,
        help="Directory that contains baseline_c* run directories.",
    )
    parser.add_argument(
        "--pattern",
        default="baseline_c*",
        help=(
            "Glob used to find run directories under --results-root for "
            "single-series plots."
        ),
    )
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        metavar="LABEL=PATTERN",
        help=(
            "Add a named series to the plot. May be passed multiple times, "
            "for example: --series baseline=baseline_c* "
            "--series decoder-residual-fusion=decoder_residual_fusion_c*. "
            "When provided, all series are plotted together and written to "
            "the same CSV."
        ),
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        type=Path,
        help="Where to write the summary CSV.",
    )
    parser.add_argument(
        "--output-figure",
        required=True,
        type=Path,
        help="Where to write the Pareto plot image.",
    )
    parser.add_argument(
        "--gpu-count",
        default=1.0,
        type=float,
        help="Number of GPUs used for the benchmark. Defaults to 1.",
    )
    parser.add_argument(
        "--title",
        default="Baseline Pareto Curve",
        help="Plot title.",
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def iter_numeric_paths(obj: object, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], float]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from iter_numeric_paths(value, path + (str(key),))
        return
    if isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from iter_numeric_paths(value, path + (str(index),))
        return
    if is_number(obj):
        yield path, float(obj)


def path_score(path: tuple[str, ...], include_terms: tuple[str, ...], exclude_terms: tuple[str, ...]) -> tuple[int, int, str] | None:
    normalized_parts = tuple(normalize(part) for part in path if normalize(part))
    joined = "_".join(normalized_parts)
    if not all(term in joined for term in include_terms):
        return None
    if any(term in joined for term in exclude_terms):
        return None

    stat_rank = 99
    for part in reversed(normalized_parts):
        if part in STAT_PRIORITY:
            stat_rank = STAT_PRIORITY[part]
            break

    return (stat_rank, len(normalized_parts), joined)


def extract_metric(flat_values: list[tuple[tuple[str, ...], float]], candidate_specs: list[tuple[tuple[str, ...], tuple[str, ...]]]) -> float | None:
    candidates: list[tuple[tuple[int, int, str], float]] = []
    for path, value in flat_values:
        for include_terms, exclude_terms in candidate_specs:
            score = path_score(path, include_terms, exclude_terms)
            if score is not None:
                candidates.append((score, value))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def extract_concurrency(run_dir: Path) -> int:
    match = re.search(r"_c(\d+)$", run_dir.name)
    if not match:
        raise ValueError(f"Could not parse concurrency from directory name: {run_dir}")
    return int(match.group(1))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_run(run_dir: Path, gpu_count: float, series_name: str) -> RunSummary:
    json_path = run_dir / "profile_export_aiperf.json"
    if not json_path.exists():
        matches = sorted(run_dir.rglob("profile_export_aiperf.json"))
        if not matches:
            raise FileNotFoundError(f"Missing profile_export_aiperf.json under {run_dir}")
        json_path = matches[0]

    payload = load_json(json_path)
    flat_values = list(iter_numeric_paths(payload))

    tokens_per_user = extract_metric(
        flat_values,
        [
            (("output", "token", "throughput", "per", "user"), ()),
            (("output_token_throughput_per_user",), ()),
        ],
    )
    output_token_throughput = extract_metric(
        flat_values,
        [
            (("output_token_throughput",), ("per_user",)),
            (("output", "token", "throughput"), ("per_user", "request")),
        ],
    )
    ttft_ms = extract_metric(
        flat_values,
        [
            (("ttft",), ()),
            (("time", "first", "token"), ()),
        ],
    )
    itl_ms = extract_metric(
        flat_values,
        [
            (("itl",), ()),
            (("inter", "token", "latency"), ()),
        ],
    )
    request_latency_ms = extract_metric(
        flat_values,
        [
            (("request", "latency"), ()),
        ],
    )

    if tokens_per_user is None:
        raise ValueError(f"Could not find output throughput per user in {json_path}")
    if output_token_throughput is None:
        raise ValueError(f"Could not find output token throughput in {json_path}")
    if gpu_count <= 0:
        raise ValueError("--gpu-count must be positive")

    concurrency = extract_concurrency(run_dir)
    return RunSummary(
        series_name=series_name,
        run_name=run_dir.name,
        concurrency=concurrency,
        tokens_per_s_per_user=tokens_per_user,
        tokens_per_s_per_gpu=output_token_throughput / gpu_count,
        output_token_throughput=output_token_throughput,
        ttft_ms=ttft_ms,
        itl_ms=itl_ms,
        request_latency_ms=request_latency_ms,
        json_path=json_path,
    )


def write_csv(rows: list[RunSummary], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "series_name",
                "run_name",
                "concurrency",
                "tokens_per_s_per_user",
                "tokens_per_s_per_gpu",
                "output_token_throughput",
                "ttft_ms",
                "itl_ms",
                "request_latency_ms",
                "json_path",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.series_name,
                    row.run_name,
                    row.concurrency,
                    f"{row.tokens_per_s_per_user:.6f}",
                    f"{row.tokens_per_s_per_gpu:.6f}",
                    f"{row.output_token_throughput:.6f}",
                    "" if row.ttft_ms is None else f"{row.ttft_ms:.6f}",
                    "" if row.itl_ms is None else f"{row.itl_ms:.6f}",
                    "" if row.request_latency_ms is None else f"{row.request_latency_ms:.6f}",
                    str(row.json_path),
                ]
            )


def split_series_arg(series_arg: str) -> tuple[str, str]:
    label, separator, pattern = series_arg.partition("=")
    if not separator or not label.strip() or not pattern.strip():
        raise ValueError(
            f"Invalid --series value {series_arg!r}. Expected LABEL=PATTERN."
        )
    return label.strip(), pattern.strip()


def collect_rows(
    results_root: Path,
    pattern: str,
    gpu_count: float,
    series_name: str,
) -> list[RunSummary]:
    run_dirs = sorted(
        [path for path in results_root.glob(pattern) if path.is_dir()],
        key=extract_concurrency,
    )
    if not run_dirs:
        raise SystemExit(f"No run directories matched {pattern!r} under {results_root}")

    rows = [
        summarize_run(run_dir, gpu_count=gpu_count, series_name=series_name)
        for run_dir in run_dirs
    ]
    rows.sort(key=lambda row: row.concurrency)
    return rows


def plot_rows(rows: list[RunSummary], output_figure: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - runtime environment issue
        raise SystemExit(
            "matplotlib is required to save the figure. "
            "Install it with: python3 -m pip install matplotlib"
        ) from exc

    output_figure.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    series_names = list(dict.fromkeys(row.series_name for row in rows))
    for series_name in series_names:
        series_rows = [row for row in rows if row.series_name == series_name]
        xs = [row.tokens_per_s_per_user for row in series_rows]
        ys = [row.tokens_per_s_per_gpu for row in series_rows]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=series_name)
        for row in series_rows:
            plt.annotate(
                f"c{row.concurrency}",
                (row.tokens_per_s_per_user, row.tokens_per_s_per_gpu),
                textcoords="offset points",
                xytext=(6, 6),
            )
    plt.xlabel("token/s/user")
    plt.ylabel("tokens/s/gpu")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    if len(series_names) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_figure, dpi=200)
    plt.close()


def print_summary(rows: list[RunSummary], output_csv: Path, output_figure: Path) -> None:
    print("Collected runs:")
    for row in rows:
        ttft = "n/a" if row.ttft_ms is None else f"{row.ttft_ms:.2f}"
        itl = "n/a" if row.itl_ms is None else f"{row.itl_ms:.2f}"
        req = "n/a" if row.request_latency_ms is None else f"{row.request_latency_ms:.2f}"
        print(
            f"  {row.series_name} / {row.run_name}: "
            f"x={row.tokens_per_s_per_user:.3f} token/s/user, "
            f"y={row.tokens_per_s_per_gpu:.3f} tokens/s/gpu, "
            f"ttft_ms={ttft}, itl_ms={itl}, request_latency_ms={req}"
        )
    print(f"Summary CSV: {output_csv}")
    print(f"Figure: {output_figure}")


def main() -> int:
    args = parse_args()
    rows: list[RunSummary] = []
    if args.series:
        for series_arg in args.series:
            series_name, pattern = split_series_arg(series_arg)
            rows.extend(
                collect_rows(
                    args.results_root,
                    pattern=pattern,
                    gpu_count=args.gpu_count,
                    series_name=series_name,
                )
            )
    else:
        rows = collect_rows(
            args.results_root,
            pattern=args.pattern,
            gpu_count=args.gpu_count,
            series_name="default",
        )

    write_csv(rows, args.output_csv)
    plot_rows(rows, args.output_figure, args.title)
    print_summary(rows, args.output_csv, args.output_figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
