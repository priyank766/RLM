"""
View and analyse results from a JSONL experiment file.

Usage:
    uv run python scripts/view_results.py experiments/niah_full.jsonl
    uv run python scripts/view_results.py experiments/niah_full.jsonl --by-length
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from comparison.eval import aggregate_results

console = Console()


def load_results(path: Path) -> list[dict]:
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="View RLM experiment results")
    parser.add_argument("file", help="Path to .jsonl results file")
    parser.add_argument("--by-length", action="store_true", help="Break down by context length")
    parser.add_argument("--show-errors", action="store_true", help="Print REPL error samples")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        sys.exit(1)

    results = load_results(path)
    console.print(f"\n[bold cyan]Results: {path.name}[/bold cyan]")
    console.print(f"Total entries: {len(results)}\n")

    # Overall summary
    agg = aggregate_results(results)
    _print_summary_table(agg, "Overall")

    # Breakdown by context length
    if args.by_length:
        lengths = sorted({r["context_length"] for r in results})
        for length in lengths:
            subset = [r for r in results if r["context_length"] == length]
            sub_agg = aggregate_results(subset)
            _print_summary_table(sub_agg, f"Context {length:,} chars")

    # Show REPL errors
    if args.show_errors:
        rlm_results = [r for r in results if r.get("method") == "rlm" and r.get("repl_errors")]
        if rlm_results:
            console.print("\n[bold yellow]REPL Error Samples:[/bold yellow]")
            for r in rlm_results[:5]:
                console.print(f"  ctx={r['context_length']:,}  iters={r.get('iterations')}")
                for err in r["repl_errors"][:2]:
                    console.print(f"    [red]{err[:200]}[/red]")
        else:
            console.print("\n[green]No REPL errors found.[/green]")


def _print_summary_table(agg: dict, title: str) -> None:
    v = agg.get("vanilla", {})
    r = agg.get("rlm", {})

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Vanilla", justify="right")
    table.add_column("RLM", justify="right")

    def pct(val):
        return f"{val * 100:.1f}%" if val is not None else "—"

    def fmt(val):
        return f"{val:.2f}" if val is not None else "—"

    table.add_row("n", str(v.get("n", 0)), str(r.get("n", 0)))
    table.add_row("Contains Answer", pct(v.get("contains_answer_rate")), pct(r.get("contains_answer_rate")))
    table.add_row("Exact Match", pct(v.get("exact_match_rate")), pct(r.get("exact_match_rate")))
    table.add_row("Numeric Match", pct(v.get("numeric_match_rate")), pct(r.get("numeric_match_rate")))
    table.add_row("Avg Latency (s)", fmt(v.get("avg_latency_s")), fmt(r.get("avg_latency_s")))
    table.add_row("Timeouts", str(v.get("timeout_count", 0)), str(r.get("timeout_count", 0)))

    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
