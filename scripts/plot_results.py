"""
Publication-quality plots for RLM vs Vanilla comparison.
Style: DeepSeek-R1-Zero paper.

Each (method x position) is its own line on the chart.
X-axis is numeric context length. Multiple lines tell different stories.

Usage:
    uv run python scripts/plot_results.py experiments/niah_full_v2.jsonl
    uv run python scripts/plot_results.py experiments/niah_full_v2.jsonl experiments/niah_long_v1.jsonl --merge
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Colors & styles per line
# ---------------------------------------------------------------------------
_BLUE  = "#1f77b4"
_RED   = "#d62728"
_GREEN = "#2ca02c"
_GRAY  = "#7f7f7f"

# Vanilla = blue shades, RLM = red shades
_LINE_STYLES = {
    # (method, position) -> (color, linestyle, marker)
    ("vanilla", 0.1): ("#1f77b4", "-",  "o"),   # blue solid
    ("vanilla", 0.5): ("#4a9fdb", "--", "s"),    # lighter blue dashed
    ("vanilla", 0.9): ("#7ec8f0", ":",  "^"),    # lightest blue dotted
    ("rlm",     0.1): ("#d62728", "-",  "o"),    # red solid
    ("rlm",     0.5): ("#e8686a", "--", "s"),    # lighter red dashed
    ("rlm",     0.9): ("#f5a3a4", ":",  "^"),    # lightest red dotted
}

_POS_NAMES = {0.1: "10%", 0.5: "50%", 0.9: "90%"}

# Length buckets
_LENGTH_BUCKETS = [4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000]
_LENGTH_LABELS  = {4000: "4K", 8000: "8K", 16000: "16K", 32000: "32K",
                   64000: "64K", 128000: "128K", 256000: "256K"}
_POSITIONS = [0.1, 0.5, 0.9]


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "#CCCCCC",
        "axes.linewidth":     0.8,
        "axes.grid":          True,
        "grid.color":         "#E0E0E0",
        "grid.linewidth":     0.5,
        "grid.alpha":         0.8,
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.labelsize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    9,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "#CCCCCC",
        "legend.fancybox":    False,
        "figure.dpi":         150,
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
        "lines.linewidth":    1.8,
        "lines.markersize":   7,
    })


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _bucket(ctx_len: int) -> int:
    return min(_LENGTH_BUCKETS, key=lambda b: abs(ctx_len - b))


def _infer_positions(results: list[dict]) -> dict[str, float]:
    try:
        from benchmarks.niah import generate_niah_suite
        all_lengths = set(r["context_length"] for r in results)
        bucket_set = set(_bucket(l) for l in all_lengths)
        tasks = generate_niah_suite(context_lengths=sorted(bucket_set), positions=_POSITIONS)
        mapping = {}
        for t in tasks:
            mapping[f"{t.context_length}:{t.query}"] = t.needle_position_pct
        return mapping
    except Exception:
        return {}


def enrich(results: list[dict]) -> list[dict]:
    pos_map = _infer_positions(results)
    for r in results:
        r["bucket_length"] = _bucket(r["context_length"])
        key = f"{r['context_length']}:{r['query']}"
        r["needle_position"] = pos_map.get(key)
    if not pos_map:
        groups = defaultdict(list)
        for r in results:
            groups[(r["bucket_length"], r["method"])].append(r)
        for group in groups.values():
            group.sort(key=lambda r: r["context_length"])
            for i, r in enumerate(group):
                if i < len(_POSITIONS):
                    r["needle_position"] = _POSITIONS[i]
    return results


def _active_buckets(results: list[dict]) -> list[int]:
    present = sorted(set(r["bucket_length"] for r in results))
    return [b for b in _LENGTH_BUCKETS if b in present]


def _merge_results(all_results: list[dict]) -> list[dict]:
    seen = {}
    for r in all_results:
        key = (r["method"], r["context_length"], r.get("query", ""))
        seen[key] = r
    return list(seen.values())


def _fmt_k(val, _pos=None):
    if val >= 1000:
        return f"{val / 1000:.0f}K"
    return f"{val:.0f}"


def _setup_xaxis(ax, buckets):
    """Set up a numeric X-axis with K-formatted ticks."""
    x_arr = np.array(buckets, dtype=float)
    ax.set_xlim(x_arr[0] * 0.8, x_arr[-1] * 1.1)
    ax.set_xticks(x_arr)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))


# ---------------------------------------------------------------------------
# Plot 1: Accuracy — one line per (method x position)
# ---------------------------------------------------------------------------

def plot_accuracy_multiline(results: list[dict], out: Path):
    """6 lines: (vanilla|rlm) x (10%|50%|90%), X=context length, Y=accuracy."""
    buckets = _active_buckets(results)
    x_arr = np.array(buckets, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in ["vanilla", "rlm"]:
        for pos in _POSITIONS:
            color, ls, marker = _LINE_STYLES[(method, pos)]
            label = f"{method} @ {_POS_NAMES[pos]}"

            ys = []
            for b in buckets:
                hits = [r for r in results
                        if r["method"] == method
                        and r["bucket_length"] == b
                        and r.get("needle_position") == pos]
                if hits:
                    ys.append(float(hits[0]["scores"]["contains_answer"]) * 100)
                else:
                    ys.append(np.nan)

            ax.plot(x_arr, ys, marker=marker, linestyle=ls, color=color,
                    markersize=7, linewidth=1.8, label=label, zorder=4)

    # Perfect baseline
    ax.axhline(100, color=_GREEN, linestyle="--", linewidth=1.0, alpha=0.4,
               label="perfect score")

    ax.set_xlabel("Context Length")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-5, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    _setup_xaxis(ax, buckets)

    # Split legend: vanilla group + rlm group
    ax.legend(loc="lower left", ncol=2, columnspacing=1.5)
    ax.set_title("NIAH accuracy by method and needle position")

    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Plot 2: Latency — one line per (method x position) with bands
# ---------------------------------------------------------------------------

def plot_latency_multiline(results: list[dict], out: Path):
    """6 lines: (vanilla|rlm) x (10%|50%|90%), X=context length, Y=latency."""
    buckets = _active_buckets(results)
    x_arr = np.array(buckets, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in ["vanilla", "rlm"]:
        for pos in _POSITIONS:
            color, ls, marker = _LINE_STYLES[(method, pos)]
            label = f"{method} @ {_POS_NAMES[pos]}"

            ys = []
            for b in buckets:
                hits = [r for r in results
                        if r["method"] == method
                        and r["bucket_length"] == b
                        and r.get("needle_position") == pos]
                if hits:
                    ys.append(hits[0]["latency_s"])
                else:
                    ys.append(np.nan)

            ax.plot(x_arr, ys, marker=marker, linestyle=ls, color=color,
                    markersize=7, linewidth=1.8, label=label, zorder=4)

    ax.set_xlabel("Context Length")
    ax.set_ylabel("Latency (seconds)")
    _setup_xaxis(ax, buckets)
    ax.legend(loc="upper left", ncol=2, columnspacing=1.5)
    ax.set_title("NIAH latency by method and needle position")

    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Plot 3: Aggregated accuracy (one line per method, averaged over positions)
# ---------------------------------------------------------------------------

def plot_accuracy_aggregated(results: list[dict], out: Path):
    """2 lines: vanilla vs rlm, averaged across all positions."""
    buckets = _active_buckets(results)
    x_arr = np.array(buckets, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))

    for method, color, marker in [("vanilla", _BLUE, "o"), ("rlm", _RED, "o")]:
        mr = [r for r in results if r["method"] == method]
        accs = []
        for b in buckets:
            br = [r for r in mr if r["bucket_length"] == b]
            if br:
                accs.append(np.mean([float(r["scores"]["contains_answer"]) for r in br]) * 100)
            else:
                accs.append(np.nan)

        ax.plot(x_arr, accs, f"{marker}-", color=color, markersize=8,
                linewidth=2, label=method, zorder=4)

        # Data labels
        for xi, yi in zip(x_arr, accs):
            if not np.isnan(yi):
                ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points",
                            xytext=(0, 12), ha="center", fontsize=9, color=color,
                            fontweight="bold")

    ax.axhline(100, color=_GREEN, linestyle="--", linewidth=1.0, alpha=0.4,
               label="perfect score")

    ax.set_xlabel("Context Length")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-5, 120)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    _setup_xaxis(ax, buckets)
    ax.legend(loc="upper right")
    ax.set_title("RLM vs Vanilla accuracy (averaged across positions)")

    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Plot 4: Accuracy by position (lines per method x length)
# ---------------------------------------------------------------------------

def plot_accuracy_by_position(results: list[dict], out: Path):
    """Lines per (method x context_length), X=position."""
    buckets = _active_buckets(results)
    x_pos = np.array([10, 50, 90], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Color gradient: short=light, long=dark
    v_colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(buckets)))
    r_colors = plt.cm.Reds(np.linspace(0.35, 0.9, len(buckets)))

    markers = ["o", "s", "^", "D", "v", "P", "*"]

    for i, b in enumerate(buckets):
        lbl = _LENGTH_LABELS[b]
        mk = markers[i % len(markers)]

        for method, cmap_colors, ls in [("vanilla", v_colors, "-"), ("rlm", r_colors, "--")]:
            mr = [r for r in results if r["method"] == method and r["bucket_length"] == b]
            ys = []
            for pos in _POSITIONS:
                hits = [r for r in mr if r.get("needle_position") == pos]
                if hits:
                    ys.append(float(hits[0]["scores"]["contains_answer"]) * 100)
                else:
                    ys.append(np.nan)

            if all(np.isnan(y) for y in ys):
                continue

            ax.plot(x_pos, ys, marker=mk, linestyle=ls,
                    color=cmap_colors[i], markersize=7, linewidth=1.6,
                    label=f"{method} {lbl}", zorder=4)

    ax.set_xlabel("Needle Position (% into document)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-5, 115)
    ax.set_xlim(0, 100)
    ax.set_xticks([10, 50, 90])
    ax.set_xticklabels(["10% (start)", "50% (middle)", "90% (end)"])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(loc="lower left", ncol=2, fontsize=8, columnspacing=1)
    ax.set_title("Accuracy by needle position (per method and context length)")

    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Plot 5: RLM Advantage (delta per position)
# ---------------------------------------------------------------------------

def plot_advantage(results: list[dict], out: Path):
    """3 lines (one per position) showing RLM% - Vanilla% across lengths."""
    buckets = _active_buckets(results)
    x_arr = np.array(buckets, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))

    pos_colors = {0.1: "#2ca02c", 0.5: "#ff7f0e", 0.9: "#9467bd"}
    pos_markers = {0.1: "o", 0.5: "s", 0.9: "^"}

    for pos in _POSITIONS:
        deltas = []
        for b in buckets:
            v_hits = [r for r in results if r["method"] == "vanilla"
                      and r["bucket_length"] == b and r.get("needle_position") == pos]
            r_hits = [r for r in results if r["method"] == "rlm"
                      and r["bucket_length"] == b and r.get("needle_position") == pos]
            if v_hits and r_hits:
                v_acc = float(v_hits[0]["scores"]["contains_answer"]) * 100
                r_acc = float(r_hits[0]["scores"]["contains_answer"]) * 100
                deltas.append(r_acc - v_acc)
            else:
                deltas.append(np.nan)

        ax.plot(x_arr, deltas, marker=pos_markers[pos], linestyle="-",
                color=pos_colors[pos], markersize=8, linewidth=1.8,
                label=f"position {_POS_NAMES[pos]}", zorder=4)

    # Zero baseline
    ax.axhline(0, color="#333333", linewidth=1.2, linestyle="-", zorder=2)

    # Shade regions
    ylims = ax.get_ylim()
    ax.fill_between(x_arr, 0, 150, alpha=0.03, color=_GREEN, zorder=1)
    ax.fill_between(x_arr, -150, 0, alpha=0.03, color=_RED, zorder=1)
    ax.text(x_arr[-1] * 0.95, 5, "RLM wins", ha="right", fontsize=9,
            color=_GREEN, fontstyle="italic", alpha=0.7)
    ax.text(x_arr[-1] * 0.95, -10, "Vanilla wins", ha="right", fontsize=9,
            color=_RED, fontstyle="italic", alpha=0.7)

    ax.set_xlabel("Context Length")
    ax.set_ylabel("Accuracy gap (RLM% - Vanilla%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}pp"))
    _setup_xaxis(ax, buckets)
    ax.legend(loc="upper left")
    ax.set_title("RLM advantage over Vanilla (per needle position)")

    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Plot 6: NIAH Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(results: list[dict], out: Path):
    buckets = _active_buckets(results)
    labels = [_LENGTH_LABELS[b] for b in buckets]
    n_cols = len(buckets)
    pos_labels = ["90%\n(end)", "50%\n(middle)", "10%\n(start)"]

    fig, (ax_v, ax_r) = plt.subplots(1, 2, figsize=(max(8, 2.5 * n_cols), 4.5))

    for ax, method, panel_title in [(ax_v, "vanilla", "Vanilla LLM"),
                                     (ax_r, "rlm", "RLM (REPL)")]:
        mr = [r for r in results if r["method"] == method]
        grid = np.full((len(_POSITIONS), n_cols), np.nan)
        annot = [[" " for _ in range(n_cols)] for _ in range(len(_POSITIONS))]

        for r in mr:
            pos = r.get("needle_position")
            bkt = r["bucket_length"]
            if pos in _POSITIONS and bkt in buckets:
                row = _POSITIONS.index(pos)
                col = buckets.index(bkt)
                correct = r["scores"].get("contains_answer", False)
                timed_out = r.get("timed_out", False)
                if timed_out:
                    grid[row, col] = 0.5; annot[row][col] = "TIMEOUT"
                elif correct:
                    grid[row, col] = 1.0; annot[row][col] = "PASS"
                else:
                    grid[row, col] = 0.0; annot[row][col] = "FAIL"

        grid = grid[::-1]; annot = annot[::-1]
        cmap = mcolors.ListedColormap(["#E85454", "#B0B0B0", "#4CAF73"])
        norm = mcolors.BoundaryNorm([-0.25, 0.25, 0.75, 1.25], cmap.N)

        ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto",
                  extent=[-0.5, n_cols - 0.5, -0.5, len(_POSITIONS) - 0.5])

        for i in range(len(_POSITIONS)):
            for j in range(n_cols):
                txt = annot[i][j]
                c = "white" if txt in ("PASS", "FAIL") else "#333333"
                ax.text(j, len(_POSITIONS) - 1 - i, txt,
                        ha="center", va="center", fontsize=11,
                        fontweight="bold", color=c)

        ax.set_xticks(range(n_cols)); ax.set_xticklabels(labels)
        ax.set_yticks(range(len(_POSITIONS))); ax.set_yticklabels(pos_labels)
        ax.set_xlabel("Context Length")
        ax.set_title(panel_title, fontweight="bold", pad=10)
        ax.tick_params(length=0)
        for i in range(len(_POSITIONS) + 1):
            ax.axhline(i - 0.5, color="white", linewidth=2)
        for j in range(n_cols + 1):
            ax.axvline(j - 0.5, color="white", linewidth=2)

    ax_v.set_ylabel("Needle Position")
    fig.suptitle("NIAH Accuracy Grid: Vanilla vs RLM",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Plot 7: Dashboard (two-panel DeepSeek layout)
# ---------------------------------------------------------------------------

def plot_dashboard(results: list[dict], out: Path):
    """Left: multiline accuracy. Right: multiline latency."""
    buckets = _active_buckets(results)
    x_arr = np.array(buckets, dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Both panels: 6 lines each
    for method in ["vanilla", "rlm"]:
        for pos in _POSITIONS:
            color, ls, marker = _LINE_STYLES[(method, pos)]
            label = f"{method} @ {_POS_NAMES[pos]}"

            accs, lats = [], []
            for b in buckets:
                hits = [r for r in results
                        if r["method"] == method
                        and r["bucket_length"] == b
                        and r.get("needle_position") == pos]
                if hits:
                    accs.append(float(hits[0]["scores"]["contains_answer"]) * 100)
                    lats.append(hits[0]["latency_s"])
                else:
                    accs.append(np.nan)
                    lats.append(np.nan)

            ax1.plot(x_arr, accs, marker=marker, linestyle=ls, color=color,
                     markersize=6, linewidth=1.5, label=label, zorder=4)
            ax2.plot(x_arr, lats, marker=marker, linestyle=ls, color=color,
                     markersize=6, linewidth=1.5, label=label, zorder=4)

    # Left panel config
    ax1.axhline(100, color=_GREEN, linestyle="--", linewidth=1, alpha=0.4)
    ax1.set_xlabel("Context Length")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(-5, 115)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    _setup_xaxis(ax1, buckets)
    ax1.legend(loc="lower left", ncol=2, fontsize=8, columnspacing=1)
    ax1.set_title("NIAH accuracy by method and needle position")

    # Right panel config
    ax2.set_xlabel("Context Length")
    ax2.set_ylabel("Latency per response (seconds)")
    _setup_xaxis(ax2, buckets)
    ax2.legend(loc="upper left", ncol=2, fontsize=8, columnspacing=1)
    ax2.set_title("NIAH latency by method and needle position")

    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Plot 8: RLM Behaviour
# ---------------------------------------------------------------------------

def plot_rlm_behaviour(results: list[dict], out: Path):
    rlm = [r for r in results if r["method"] == "rlm"]
    if not rlm:
        return

    buckets = _active_buckets(results)
    x_arr = np.array(buckets, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))

    # One line per position showing iterations
    for pos in _POSITIONS:
        color = {"0.1": "#2ca02c", "0.5": "#ff7f0e", "0.9": "#9467bd"}[str(pos)]
        marker = {"0.1": "o", "0.5": "s", "0.9": "^"}[str(pos)]

        iters = []
        for b in buckets:
            hits = [r for r in rlm if r["bucket_length"] == b
                    and r.get("needle_position") == pos]
            if hits:
                iters.append(hits[0].get("iterations", hits[0].get("rlm_iterations", 1)))
            else:
                iters.append(np.nan)

        ax.plot(x_arr, iters, marker=marker, linestyle="-", color=color,
                markersize=8, linewidth=1.8,
                label=f"position {_POS_NAMES[pos]}", zorder=4)

    # Scatter overlay: color by outcome
    for r in rlm:
        b = r["bucket_length"]
        iters_val = r.get("iterations", r.get("rlm_iterations", 1))
        correct = r["scores"].get("contains_answer", False)
        timed_out = r.get("timed_out", False)
        if timed_out:
            ec, fc = _GRAY, _GRAY
        elif correct:
            ec, fc = _GREEN, _GREEN
        else:
            ec, fc = _RED, _RED

        ax.scatter(b, iters_val, s=100, color=fc, edgecolor="white",
                   linewidth=1.5, zorder=6, alpha=0.8)

    ax.axhline(20, color=_GRAY, linestyle="--", linewidth=1, alpha=0.7,
               label="max iterations")

    # Custom legend for outcomes
    extra = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_GREEN,
               markersize=9, label="Correct"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_RED,
               markersize=9, label="Wrong"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_GRAY,
               markersize=9, label="Timeout"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + extra, loc="upper left", ncol=2, fontsize=9)

    ax.set_xlabel("Context Length")
    ax.set_ylabel("REPL Iterations")
    _setup_xaxis(ax, buckets)
    ax.set_title("RLM REPL iterations by context length and position")

    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate DeepSeek-R1-Zero style plots for RLM experiments")
    parser.add_argument("files", nargs="+", help="One or more .jsonl result files")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all files into one combined dataset")
    parser.add_argument("--task-type", default=None, help="Filter: niah | long_doc_qa")
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args()

    _apply_style()

    out_dir = Path(args.out) if args.out else Path("experiments") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, list[dict]]] = []
    all_raw: list[dict] = []
    for fpath in args.files:
        path = Path(fpath)
        if not path.exists():
            print(f"ERROR: {path} not found"); sys.exit(1)
        raw = load_jsonl(path)
        if args.task_type:
            raw = [r for r in raw if r.get("task_type") == args.task_type]
        enriched = enrich(raw)
        runs.append((path.stem, enriched))
        all_raw.extend(enriched)
        print(f"Loaded {len(enriched)} results from {path.name}")

    if not runs:
        print("No results."); sys.exit(1)

    if args.merge and len(runs) > 1:
        merged = enrich(_merge_results(all_raw))
        prefix = "combined"
        data = merged
        print(f"Merged into {len(merged)} unique results")
    else:
        prefix, data = runs[0]

    print(f"\nGenerating plots -> {out_dir}/\n")

    plot_accuracy_multiline(data,     out_dir / f"{prefix}_accuracy_multiline.png")
    plot_latency_multiline(data,      out_dir / f"{prefix}_latency_multiline.png")
    plot_accuracy_aggregated(data,    out_dir / f"{prefix}_accuracy_aggregated.png")
    plot_accuracy_by_position(data,   out_dir / f"{prefix}_acc_by_position.png")
    plot_advantage(data,              out_dir / f"{prefix}_advantage.png")
    plot_heatmap(data,                out_dir / f"{prefix}_heatmap.png")
    plot_dashboard(data,              out_dir / f"{prefix}_dashboard.png")
    plot_rlm_behaviour(data,          out_dir / f"{prefix}_rlm_behaviour.png")

    print(f"\nDone. 8 plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
