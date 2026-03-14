"""
Clean research-style plotting for RLM experiment results.

Principles:
- One figure per insight.
- Minimal clutter.
- Comparison-focused, not decorative.
- Use the actual experiment schema directly.

Outputs are written into `experiments/plots/` by default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

LENGTH_BUCKETS = [4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000]
LENGTH_LABELS = {
    4_000: "4K",
    8_000: "8K",
    16_000: "16K",
    32_000: "32K",
    64_000: "64K",
    128_000: "128K",
    256_000: "256K",
}
POSITIONS = [0.1, 0.5, 0.9]
POSITION_LABELS = {0.1: "10%", 0.5: "50%", 0.9: "90%"}
METHODS = ["vanilla", "rlm"]
METHOD_LABELS = {"vanilla": "Vanilla", "rlm": "RLM"}
METHOD_COLORS = {"vanilla": "#1f4e79", "rlm": "#b23a48"}
METHOD_MARKERS = {"vanilla": "o", "rlm": "s"}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#d7d7d7",
            "grid.linestyle": "--",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.85,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#cfcfcf",
            "legend.fancybox": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bucket_length(context_length: int) -> int:
    return min(LENGTH_BUCKETS, key=lambda bucket: abs(context_length - bucket))


def infer_niah_positions(rows: list[dict]) -> dict[str, float]:
    try:
        from benchmarks.niah import generate_niah_suite
    except Exception:
        return {}

    niah_rows = [row for row in rows if row.get("task_type") == "niah"]
    if not niah_rows:
        return {}

    bucket_set = sorted({bucket_length(row["context_length"]) for row in niah_rows})
    tasks = generate_niah_suite(context_lengths=bucket_set, positions=POSITIONS)
    mapping: dict[str, float] = {}
    for task in tasks:
        mapping[f"{task.context_length}:{task.query}"] = task.needle_position_pct
    return mapping


def enrich_rows(rows: list[dict]) -> list[dict]:
    pos_map = infer_niah_positions(rows)
    enriched: list[dict] = []
    for row in rows:
        item = dict(row)
        item["bucket_length"] = bucket_length(row["context_length"])
        item["correct"] = bool(row.get("scores", {}).get("contains_answer", False))
        item["timeout"] = bool(row.get("timed_out", False))
        item["truncated"] = bool(row.get("truncated", False))
        item["iterations"] = int(row.get("iterations", row.get("rlm_iterations", 0) or 0))
        item["sub_calls"] = int(row.get("sub_calls", 0) or 0)
        item["repl_error_count"] = len(row.get("repl_errors", []))
        key = f"{row['context_length']}:{row['query']}"
        item["needle_position"] = pos_map.get(key)
        enriched.append(item)

    if pos_map:
        return enriched

    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for item in enriched:
        if item.get("task_type") == "niah":
            grouped[(item["bucket_length"], item["method"])].append(item)

    for group in grouped.values():
        group.sort(key=lambda item: item["context_length"])
        for index, item in enumerate(group[: len(POSITIONS)]):
            item["needle_position"] = POSITIONS[index]
    return enriched


def merge_rows(rows: list[dict]) -> list[dict]:
    unique: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row.get("run_name"),
            row.get("method"),
            row.get("task_type"),
            row.get("context_length"),
            row.get("query"),
        )
        unique[key] = row
    return list(unique.values())


def active_buckets(rows: list[dict]) -> list[int]:
    present = {row["bucket_length"] for row in rows}
    return [bucket for bucket in LENGTH_BUCKETS if bucket in present]


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + (z * z) / total
    center = (p + (z * z) / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + (z * z) / (4 * total)) / total) / denom
    return (center - margin, center + margin)


def summarize_by_method_bucket(rows: list[dict]) -> dict[tuple[str, int], dict]:
    summary: dict[tuple[str, int], dict] = {}
    for method in METHODS:
        for bucket in active_buckets(rows):
            subset = [row for row in rows if row["method"] == method and row["bucket_length"] == bucket]
            if not subset:
                continue
            successes = sum(1 for row in subset if row["correct"])
            total = len(subset)
            lo, hi = wilson_interval(successes, total)
            latencies = [float(row.get("latency_s", 0.0)) for row in subset]
            summary[(method, bucket)] = {
                "n": total,
                "accuracy": successes / total,
                "accuracy_lo": lo,
                "accuracy_hi": hi,
                "latency_median": percentile(latencies, 50),
                "latency_p25": percentile(latencies, 25),
                "latency_p75": percentile(latencies, 75),
                "timeout_rate": mean([1.0 if row["timeout"] else 0.0 for row in subset]),
                "truncation_rate": mean([1.0 if row["truncated"] else 0.0 for row in subset]),
            }
    return summary


def summarize_task_types(rows: list[dict]) -> dict[tuple[str, str], dict]:
    summary: dict[tuple[str, str], dict] = {}
    task_types = sorted({row.get("task_type", "unknown") for row in rows})
    for task_type in task_types:
        for method in METHODS:
            subset = [
                row for row in rows if row.get("task_type") == task_type and row["method"] == method
            ]
            if not subset:
                continue
            latencies = [float(row.get("latency_s", 0.0)) for row in subset]
            summary[(task_type, method)] = {
                "n": len(subset),
                "accuracy": mean([1.0 if row["correct"] else 0.0 for row in subset]),
                "latency_median": percentile(latencies, 50),
            }
    return summary


def summarize_rlm(rows: list[dict]) -> dict[int, dict]:
    rlm_rows = [row for row in rows if row["method"] == "rlm"]
    summary: dict[int, dict] = {}
    for bucket in active_buckets(rlm_rows):
        subset = [row for row in rlm_rows if row["bucket_length"] == bucket]
        if not subset:
            continue
        summary[bucket] = {
            "iterations": mean([float(row.get("iterations", 0.0)) for row in subset]),
            "sub_calls": mean([float(row.get("sub_calls", 0.0)) for row in subset]),
            "timeout_rate": mean([1.0 if row["timeout"] else 0.0 for row in subset]),
            "error_rate": mean([1.0 if row.get("repl_error_count", 0) > 0 else 0.0 for row in subset]),
        }
    return summary


def summarize_pairing(rows: list[dict]) -> dict[int, dict[str, float]]:
    paired: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        pair_key = (
            row.get("task_type"),
            row.get("context_length"),
            row.get("query"),
            row.get("answer"),
        )
        paired[pair_key][row["method"]] = row

    summary: dict[int, dict[str, float]] = defaultdict(
        lambda: {"both_correct": 0, "rlm_only": 0, "vanilla_only": 0, "both_wrong": 0, "n": 0}
    )
    for methods in paired.values():
        vanilla = methods.get("vanilla")
        rlm = methods.get("rlm")
        if not vanilla or not rlm:
            continue
        bucket = vanilla["bucket_length"]
        summary[bucket]["n"] += 1
        if vanilla["correct"] and rlm["correct"]:
            summary[bucket]["both_correct"] += 1
        elif rlm["correct"] and not vanilla["correct"]:
            summary[bucket]["rlm_only"] += 1
        elif vanilla["correct"] and not rlm["correct"]:
            summary[bucket]["vanilla_only"] += 1
        else:
            summary[bucket]["both_wrong"] += 1
    return summary


def summarize_niah_position(rows: list[dict], position: float) -> dict[tuple[str, int], dict]:
    subset = [row for row in rows if row.get("task_type") == "niah" and row.get("needle_position") == position]
    return summarize_by_method_bucket(subset)


def setup_context_axis(ax: plt.Axes, buckets: list[int]) -> np.ndarray:
    positions = np.arange(len(buckets), dtype=float)
    ax.set_xticks(positions)
    ax.set_xticklabels([LENGTH_LABELS[bucket] for bucket in buckets])
    ax.set_xlim(-0.35, len(buckets) - 0.65)
    return positions


def plot_accuracy_by_context(rows: list[dict], out_path: Path) -> None:
    buckets = active_buckets(rows)
    summary = summarize_by_method_bucket(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = setup_context_axis(ax, buckets)

    for method in METHODS:
        acc = []
        lo = []
        hi = []
        for bucket in buckets:
            stats = summary.get((method, bucket))
            acc.append((stats["accuracy"] * 100) if stats else np.nan)
            lo.append((stats["accuracy_lo"] * 100) if stats else np.nan)
            hi.append((stats["accuracy_hi"] * 100) if stats else np.nan)
        ax.fill_between(x, lo, hi, color=METHOD_COLORS[method], alpha=0.12, linewidth=0)
        ax.plot(x, acc, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], label=METHOD_LABELS[method])

    for idx, bucket in enumerate(buckets):
        n = summary.get(("vanilla", bucket), {}).get("n", 0)
        if n:
            ax.annotate(f"n={n}", (x[idx], 102), ha="center", va="bottom", fontsize=8, color="#666666")

    ax.set_title("Accuracy by context length")
    ax.set_ylabel("Contains-answer accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_latency_by_context(rows: list[dict], out_path: Path) -> None:
    buckets = active_buckets(rows)
    summary = summarize_by_method_bucket(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = setup_context_axis(ax, buckets)

    for method in METHODS:
        median = []
        p25 = []
        p75 = []
        for bucket in buckets:
            stats = summary.get((method, bucket))
            median.append(stats["latency_median"] if stats else np.nan)
            p25.append(stats["latency_p25"] if stats else np.nan)
            p75.append(stats["latency_p75"] if stats else np.nan)
        ax.fill_between(x, p25, p75, color=METHOD_COLORS[method], alpha=0.12, linewidth=0)
        ax.plot(x, median, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], label=METHOD_LABELS[method])

    ax.set_title("Latency by context length")
    ax.set_ylabel("Median latency (s)")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_advantage_by_context(rows: list[dict], out_path: Path) -> None:
    buckets = active_buckets(rows)
    summary = summarize_by_method_bucket(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = setup_context_axis(ax, buckets)
    delta = []
    for bucket in buckets:
        vanilla = summary.get(("vanilla", bucket))
        rlm = summary.get(("rlm", bucket))
        if vanilla and rlm:
            delta.append((rlm["accuracy"] - vanilla["accuracy"]) * 100)
        else:
            delta.append(np.nan)

    colors = ["#4c956c" if value >= 0 else "#8da9c4" for value in delta]
    ax.bar(x, delta, color=colors, width=0.62, edgecolor="none")
    ax.axhline(0, color="#444444", linewidth=1.0)
    ax.set_title("RLM advantage by context length")
    ax.set_ylabel("Accuracy gap (RLM - Vanilla)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _pos: f"{value:+.0f} pp"))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_outcome_decomposition(rows: list[dict], out_path: Path) -> None:
    buckets = active_buckets(rows)
    summary = summarize_pairing(rows)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = setup_context_axis(ax, buckets)

    outcome_order = ["both_correct", "rlm_only", "vanilla_only", "both_wrong"]
    outcome_labels = {
        "both_correct": "Both correct",
        "rlm_only": "RLM-only win",
        "vanilla_only": "Vanilla-only win",
        "both_wrong": "Both wrong",
    }
    outcome_colors = {
        "both_correct": "#6ba368",
        "rlm_only": "#d9a441",
        "vanilla_only": "#7f9ccf",
        "both_wrong": "#c9c9c9",
    }

    bottom = np.zeros(len(buckets), dtype=float)
    for outcome in outcome_order:
        values = [summary.get(bucket, {}).get(outcome, 0) for bucket in buckets]
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.62,
            color=outcome_colors[outcome],
            edgecolor="none",
            label=outcome_labels[outcome],
        )
        bottom += np.asarray(values, dtype=float)

    for idx, bucket in enumerate(buckets):
        total = int(summary.get(bucket, {}).get("n", 0))
        if total:
            ax.annotate(f"n={total}", (x[idx], total + 0.05), ha="center", va="bottom", fontsize=8, color="#666666")

    ax.set_title("Paired outcome decomposition")
    ax.set_ylabel("Number of paired tasks")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_task_type_accuracy(rows: list[dict], out_path: Path) -> bool:
    task_types = sorted({row.get("task_type", "unknown") for row in rows})
    if len(task_types) <= 1:
        return False

    summary = summarize_task_types(rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = np.arange(len(task_types), dtype=float)
    width = 0.34

    for offset, method in [(-width / 2, "vanilla"), (width / 2, "rlm")]:
        values = []
        for task_type in task_types:
            stats = summary.get((task_type, method))
            values.append((stats["accuracy"] * 100) if stats else np.nan)
        ax.bar(x + offset, values, width=width, color=METHOD_COLORS[method], label=METHOD_LABELS[method])

    ax.set_xticks(x)
    ax.set_xticklabels(task_types)
    ax.set_title("Accuracy by task family")
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_task_type_latency(rows: list[dict], out_path: Path) -> bool:
    task_types = sorted({row.get("task_type", "unknown") for row in rows})
    if len(task_types) <= 1:
        return False

    summary = summarize_task_types(rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = np.arange(len(task_types), dtype=float)
    width = 0.34

    for offset, method in [(-width / 2, "vanilla"), (width / 2, "rlm")]:
        values = []
        for task_type in task_types:
            stats = summary.get((task_type, method))
            values.append(stats["latency_median"] if stats else np.nan)
        ax.bar(x + offset, values, width=width, color=METHOD_COLORS[method], label=METHOD_LABELS[method])

    ax.set_xticks(x)
    ax.set_xticklabels(task_types)
    ax.set_title("Latency by task family")
    ax.set_ylabel("Median latency (s)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_niah_position_accuracy(rows: list[dict], position: float, out_path: Path) -> bool:
    niah_rows = [row for row in rows if row.get("task_type") == "niah" and row.get("needle_position") == position]
    if not niah_rows:
        return False

    buckets = active_buckets(niah_rows)
    summary = summarize_niah_position(rows, position)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = setup_context_axis(ax, buckets)

    for method in METHODS:
        values = []
        for bucket in buckets:
            stats = summary.get((method, bucket))
            values.append((stats["accuracy"] * 100) if stats else np.nan)
        ax.plot(x, values, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], label=METHOD_LABELS[method])

    ax.set_title(f"NIAH accuracy at needle position {POSITION_LABELS[position]}")
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_rlm_iterations(rows: list[dict], out_path: Path) -> bool:
    summary = summarize_rlm(rows)
    if not summary:
        return False

    buckets = sorted(summary)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = setup_context_axis(ax, buckets)
    values = [summary[bucket]["iterations"] for bucket in buckets]
    ax.plot(x, values, color="#7a3e9d", marker="o")
    ax.set_title("RLM iterations by context length")
    ax.set_ylabel("Average iterations")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_rlm_subcalls(rows: list[dict], out_path: Path) -> bool:
    summary = summarize_rlm(rows)
    if not summary:
        return False

    buckets = sorted(summary)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = setup_context_axis(ax, buckets)
    values = [summary[bucket]["sub_calls"] for bucket in buckets]
    ax.plot(x, values, color="#c46210", marker="s")
    ax.set_title("RLM sub-calls by context length")
    ax.set_ylabel("Average sub-calls")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_rlm_failure_rates(rows: list[dict], out_path: Path) -> bool:
    summary = summarize_rlm(rows)
    if not summary:
        return False

    buckets = sorted(summary)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = setup_context_axis(ax, buckets)
    timeout_rate = [summary[bucket]["timeout_rate"] * 100 for bucket in buckets]
    error_rate = [summary[bucket]["error_rate"] * 100 for bucket in buckets]
    width = 0.32

    ax.bar(x - width / 2, timeout_rate, width=width, color="#7f9ccf", label="Timeout rate")
    ax.bar(x + width / 2, error_rate, width=width, color="#c97b7b", label="REPL error rate")
    ax.set_title("RLM failure rates by context length")
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def clean_output_dir(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    for path in out_dir.glob("*.png"):
        path.unlink()


def generate_plots(prefix: str, rows: list[dict], out_dir: Path) -> None:
    plot_accuracy_by_context(rows, out_dir / f"{prefix}_accuracy_by_context.png")
    plot_latency_by_context(rows, out_dir / f"{prefix}_latency_by_context.png")
    plot_advantage_by_context(rows, out_dir / f"{prefix}_advantage_by_context.png")
    plot_outcome_decomposition(rows, out_dir / f"{prefix}_outcome_decomposition.png")
    plot_task_type_accuracy(rows, out_dir / f"{prefix}_task_type_accuracy.png")
    plot_task_type_latency(rows, out_dir / f"{prefix}_task_type_latency.png")
    plot_niah_position_accuracy(rows, 0.1, out_dir / f"{prefix}_niah_accuracy_pos_10.png")
    plot_niah_position_accuracy(rows, 0.5, out_dir / f"{prefix}_niah_accuracy_pos_50.png")
    plot_niah_position_accuracy(rows, 0.9, out_dir / f"{prefix}_niah_accuracy_pos_90.png")
    plot_rlm_iterations(rows, out_dir / f"{prefix}_rlm_iterations.png")
    plot_rlm_subcalls(rows, out_dir / f"{prefix}_rlm_subcalls.png")
    plot_rlm_failure_rates(rows, out_dir / f"{prefix}_rlm_failure_rates.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clean research-style plots for RLM experiments")
    parser.add_argument("files", nargs="+", help="One or more result JSONL files")
    parser.add_argument("--merge", action="store_true", help="Merge inputs into one combined output set")
    parser.add_argument("--task-type", default=None, help="Optional task type filter")
    parser.add_argument("--out", default=None, help="Output directory (default: experiments/plots)")
    parser.add_argument("--keep-old", action="store_true", help="Do not clear old PNG files in the output directory")
    args = parser.parse_args()

    apply_style()

    out_dir = Path(args.out) if args.out else Path("experiments") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_old:
        clean_output_dir(out_dir)

    loaded: list[tuple[str, list[dict]]] = []
    merged_input: list[dict] = []
    for file_name in args.files:
        path = Path(file_name)
        if not path.exists():
            print(f"ERROR: file not found: {path}")
            sys.exit(1)
        rows = load_jsonl(path)
        if args.task_type:
            rows = [row for row in rows if row.get("task_type") == args.task_type]
        enriched = enrich_rows(rows)
        loaded.append((path.stem, enriched))
        merged_input.extend(enriched)
        print(f"Loaded {len(enriched)} rows from {path.name}")

    if not loaded:
        print("No results loaded")
        sys.exit(1)

    if args.merge:
        merged = enrich_rows(merge_rows(merged_input))
        generate_plots("combined", merged, out_dir)
    else:
        for prefix, rows in loaded:
            generate_plots(prefix, rows, out_dir)

    print(f"Plot files saved to {out_dir}/")


if __name__ == "__main__":
    main()
