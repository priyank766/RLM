"""
Smoke test: run one tiny NIAH task through both Vanilla and RLM.

Usage:
    uv run python scripts/smoke_test.py

This is the fastest way to verify:
  1. Ollama is running and the model is loaded
  2. The REPL executes correctly
  3. The RLM loop terminates with FINAL()
  4. Both methods return a prediction
"""

import sys
from pathlib import Path

# Add src/ to path so we can import our packages
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

from rlm.clients.ollama import OllamaClient
from rlm.rlm_repl import RLM_REPL
from baseline.vanilla_llm import VanillaLLM
from benchmarks.niah import generate_niah_task

console = Console()

MODEL = "qwen3.5:2b"
CONTEXT_CHARS = 4_000  # Keep small for a quick smoke test


def main() -> None:
    console.print("\n[bold cyan]RLM Smoke Test[/bold cyan]")
    console.print(f"Model: [yellow]{MODEL}[/yellow]")
    console.print(f"Context size: [yellow]{CONTEXT_CHARS:,} chars[/yellow]\n")

    # Check Ollama
    client = OllamaClient(model=MODEL)
    if not client.is_available():
        console.print("[red]Ollama is not running![/red]")
        console.print(f"[yellow]Start it with: ollama serve[/yellow]")
        console.print(f"[yellow]Pull model with: ollama pull {MODEL}[/yellow]")
        sys.exit(1)
    console.print("[green]Ollama is running[/green]")

    # Generate task
    task = generate_niah_task(target_chars=CONTEXT_CHARS, needle_position_pct=0.5, seed=42)
    console.print(f"\nTask:")
    console.print(f"  Query  : [cyan]{task.query}[/cyan]")
    console.print(f"  Answer : [green]{task.answer}[/green]")
    console.print(f"  Context: {task.context_length:,} chars\n")

    # ── Vanilla ──────────────────────────────────────────────────────────────
    console.print("[bold]Running Vanilla LLM...[/bold]")
    vanilla = VanillaLLM(client)
    v_result = vanilla.completion(task.context, task.query)
    v_correct = task.answer in (v_result["answer"] or "")
    console.print(f"  Prediction : {v_result['answer'][:200]}")
    console.print(f"  Correct    : {'[green]YES[/green]' if v_correct else '[red]NO[/red]'}")
    console.print(f"  Latency    : {v_result['latency_s']:.1f}s\n")

    # ── RLM ──────────────────────────────────────────────────────────────────
    console.print("[bold]Running RLM...[/bold]")
    rlm = RLM_REPL(root_client=client, max_iterations=10)
    r_result = rlm.completion(task.context, task.query)
    r_correct = task.answer in (r_result["answer"] or "")

    console.print(f"  Prediction : {str(r_result['answer'])[:200]}")
    console.print(f"  Correct    : {'[green]YES[/green]' if r_correct else '[red]NO[/red]'}")
    console.print(f"  Iterations : {r_result['iterations']}")
    console.print(f"  Sub-calls  : {r_result['sub_calls']}")
    console.print(f"  Timed out  : {r_result['timed_out']}")
    console.print(f"  REPL errors: {len(r_result['repl_errors'])}")
    console.print(f"  Latency    : {r_result['latency_s']:.1f}s\n")

    if r_result["repl_errors"]:
        console.print("[yellow]REPL errors encountered:[/yellow]")
        for err in r_result["repl_errors"]:
            console.print(f"  [red]{err[:300]}[/red]")

    # Print trajectory summary
    console.print(f"\n[bold]RLM Trajectory ({r_result['iterations']} iterations):[/bold]")
    for entry in r_result["trajectory"]:
        i = entry["iteration"]
        code_preview = entry["code"][:80].replace("\n", " ")
        has_err = bool(entry["stderr"])
        final = "[green](FINAL)[/green]" if entry["final_set"] else ""
        err_tag = "[red](ERR)[/red]" if has_err else ""
        console.print(f"  [{i}] {code_preview}... {final}{err_tag}")

    console.print("\n[bold green]Smoke test complete.[/bold green]")


if __name__ == "__main__":
    main()
