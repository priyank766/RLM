"""
Comparison Runner.

Runs Vanilla LLM and RLM on the same benchmark tasks and saves
results to JSONL in experiments/. Prints a live summary table.

Key features:
  - Resumable: re-running with the same run_name skips already-done tasks
  - Trajectory saving: --save-trajectories writes full RLM code traces to a
    separate JSONL for RL/SFT training data collection
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from rich.console import Console
from rich.table import Table
from rich.progress import track

from rlm.clients.ollama import OllamaClient
from rlm.rlm_repl import RLM_REPL
from baseline.vanilla_llm import VanillaLLM
from comparison.eval import score_result, aggregate_results

console = Console()

EXPERIMENTS_DIR = Path(__file__).parent.parent.parent / "experiments"


class Task(Protocol):
    context: str
    query: str
    answer: str
    context_length: int
    task_type: str


def run_comparison(
    tasks: list[Any],
    model: str = "qwen3.5:2b",
    run_name: str | None = None,
    max_rlm_iterations: int = 20,
    skip_vanilla: bool = False,
    skip_rlm: bool = False,
    save_trajectories: bool = False,
) -> list[dict]:
    """
    Run both Vanilla LLM and RLM on all tasks. Save results to JSONL.

    Resumable: if run_name already has a partial JSONL, completed tasks are
    skipped automatically. Safe to Ctrl+C and restart.

    Args:
        tasks: List of task objects (NIAHTask, LongDocTask, etc.)
        model: Ollama model tag.
        run_name: Experiment name (auto-generated if None).
        max_rlm_iterations: Max REPL iterations per RLM task.
        skip_vanilla / skip_rlm: Run only one method.
        save_trajectories: Save full RLM code traces to a separate JSONL.

    Returns:
        List of result dicts (also saved to JSONL).
    """
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = EXPERIMENTS_DIR / f"{run_name}.jsonl"
    traj_path = EXPERIMENTS_DIR / f"{run_name}_trajectories.jsonl" if save_trajectories else None

    # ── Resume: load already-completed task fingerprints ─────────────────────
    completed = _load_completed_fingerprints(output_path)
    if completed:
        console.print(f"[yellow]Resuming — {len(completed)} tasks already done, skipping them.[/yellow]")

    client = OllamaClient(model=model)

    if not client.is_available():
        console.print(f"[red]ERROR: Ollama not available at {client.base_url}[/red]")
        console.print("[yellow]Start Ollama with: ollama serve[/yellow]")
        console.print(f"[yellow]Pull model: ollama pull {model}[/yellow]")
        return []

    console.print(f"\n[bold cyan]RLM vs Vanilla Comparison[/bold cyan]")
    console.print(f"Model      : [yellow]{model}[/yellow]")
    console.print(f"Tasks      : [yellow]{len(tasks)}[/yellow]")
    console.print(f"Run name   : [yellow]{run_name}[/yellow]")
    console.print(f"Output     : [yellow]{output_path}[/yellow]")
    if traj_path:
        console.print(f"Trajectories: [yellow]{traj_path}[/yellow]")
    console.print()

    vanilla = VanillaLLM(client)
    rlm = RLM_REPL(root_client=client, max_iterations=max_rlm_iterations)

    results: list[dict] = []

    for i, task in enumerate(track(tasks, description="Running tasks...")):
        task_meta = {
            "context_length": task.context_length,
            "query": task.query,
            "answer": task.answer,
            "task_type": task.task_type,
        }

        # ── Vanilla ──────────────────────────────────────────────────────────
        if not skip_vanilla:
            fp = _fingerprint("vanilla", task)
            if fp in completed:
                console.print(f"  [{i+1}/{len(tasks)}] VANILLA  [dim](skipped — already done)[/dim]")
            else:
                vanilla_entry = _run_vanilla(vanilla, task, task_meta, run_name)
                results.append(vanilla_entry)
                _save_result(output_path, vanilla_entry)
                _print_row(i + 1, vanilla_entry, len(tasks))

        # ── RLM ──────────────────────────────────────────────────────────────
        if not skip_rlm:
            fp = _fingerprint("rlm", task)
            if fp in completed:
                console.print(f"  [{i+1}/{len(tasks)}] RLM      [dim](skipped — already done)[/dim]")
            else:
                rlm_entry, trajectory = _run_rlm(rlm, task, task_meta, run_name)
                results.append(rlm_entry)
                _save_result(output_path, rlm_entry)
                _print_row(i + 1, rlm_entry, len(tasks))

                if traj_path and trajectory:
                    _save_trajectory(traj_path, rlm_entry, trajectory)

    _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _fingerprint(method: str, task: Any) -> str:
    """Stable ID for one (method, task) pair — used to detect already-done work."""
    key = f"{method}:{getattr(task, 'task_type', '')}:{task.context_length}:{task.query}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _load_completed_fingerprints(path: Path) -> set[str]:
    """Load fingerprints of tasks already saved in an existing JSONL."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = f"{r['method']}:{r.get('task_type','')}:{r['context_length']}:{r['query']}"
                done.add(hashlib.md5(key.encode()).hexdigest()[:16])
            except Exception:
                pass
    return done


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def _run_vanilla(vanilla: VanillaLLM, task: Any, task_meta: dict, run_name: str) -> dict:
    try:
        result = vanilla.completion(task.context, task.query)
        return {
            "method": "vanilla",
            "run_name": run_name,
            **task_meta,
            "prediction": result["answer"],
            "scores": score_result(result["answer"], task.answer),
            "latency_s": result["latency_s"],
            "truncated": result["truncated"],
            "timed_out": False,
        }
    except Exception as exc:
        return _error_entry("vanilla", run_name, task_meta, exc)


def _run_rlm(
    rlm: RLM_REPL, task: Any, task_meta: dict, run_name: str
) -> tuple[dict, list[dict]]:
    try:
        result = rlm.completion(task.context, task.query)
        entry = {
            "method": "rlm",
            "run_name": run_name,
            **task_meta,
            "prediction": result["answer"],
            "scores": score_result(result["answer"], task.answer),
            "latency_s": result["latency_s"],
            "iterations": result["iterations"],
            "sub_calls": result["sub_calls"],
            "repl_errors": result["repl_errors"],
            "timed_out": result["timed_out"],
        }
        return entry, result.get("trajectory", [])
    except Exception as exc:
        return _error_entry("rlm", run_name, task_meta, exc), []


def _error_entry(method: str, run_name: str, task_meta: dict, exc: Exception) -> dict:
    return {
        "method": method,
        "run_name": run_name,
        **task_meta,
        "prediction": None,
        "scores": {"exact_match": False, "contains_answer": False, "numeric_match": False},
        "error": str(exc),
        "latency_s": 0.0,
        "timed_out": False,
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_result(path: Path, result: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def _save_trajectory(path: Path, result: dict, trajectory: list[dict]) -> None:
    """Save full RLM trajectory alongside the result summary — used for RL training data."""
    record = {
        "task_type": result["task_type"],
        "context_length": result["context_length"],
        "query": result["query"],
        "answer": result["answer"],
        "prediction": result["prediction"],
        "correct": result["scores"]["contains_answer"],
        "timed_out": result.get("timed_out", False),
        "iterations": result.get("iterations", 0),
        "sub_calls": result.get("sub_calls", 0),
        "repl_errors": result.get("repl_errors", []),
        "trajectory": trajectory,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_row(task_num: int, result: dict, total: int) -> None:
    method = result["method"].upper()
    correct = result["scores"].get("contains_answer", False)
    mark = "[green]PASS[/green]" if correct else "[red]FAIL[/red]"
    timeout_tag = " [yellow](timeout)[/yellow]" if result.get("timed_out") else ""
    error_tag = " [red](error)[/red]" if result.get("error") else ""
    extra = ""
    if result["method"] == "rlm":
        extra = f"  iters={result.get('iterations','?')}  sub_calls={result.get('sub_calls','?')}"
    console.print(
        f"  [{task_num}/{total}] {method:7s} {mark}  "
        f"ctx={result['context_length']:,}ch  "
        f"lat={result['latency_s']:.1f}s"
        f"{extra}{timeout_tag}{error_tag}"
    )


def _print_summary(results: list[dict]) -> None:
    if not results:
        return
    agg = aggregate_results(results)
    v = agg.get("vanilla", {})
    r = agg.get("rlm", {})

    table = Table(title="\nComparison Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Vanilla", justify="right")
    table.add_column("RLM", justify="right")

    def pct(val):
        return f"{val * 100:.1f}%" if val is not None else "—"
    def fmt(val):
        return f"{val:.2f}" if val is not None else "—"

    table.add_row("Contains Answer", pct(v.get("contains_answer_rate")), pct(r.get("contains_answer_rate")))
    table.add_row("Exact Match",     pct(v.get("exact_match_rate")),     pct(r.get("exact_match_rate")))
    table.add_row("Numeric Match",   pct(v.get("numeric_match_rate")),   pct(r.get("numeric_match_rate")))
    table.add_row("Avg Latency (s)", fmt(v.get("avg_latency_s")),        fmt(r.get("avg_latency_s")))
    table.add_row("Timeouts",        str(v.get("timeout_count", 0)),     str(r.get("timeout_count", 0)))
    table.add_row("Errors",          str(v.get("error_count",   0)),     str(r.get("error_count",   0)))

    console.print(table)
