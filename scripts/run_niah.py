"""
Run the full NIAH benchmark suite: Vanilla vs RLM.

Usage:
    uv run python scripts/run_niah.py
    uv run python scripts/run_niah.py --small          # 4K/8K only (faster)
    uv run python scripts/run_niah.py --rlm-only
    uv run python scripts/run_niah.py --vanilla-only
    uv run python scripts/run_niah.py --save-trajectories   # save RLM traces for RL training
    uv run python scripts/run_niah.py --run-name niah_v2    # resume a named run

Resumable: re-running with the same --run-name skips already-completed tasks.
Results saved to: experiments/<run_name>.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarks.niah import generate_niah_suite
from comparison.runner import run_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="NIAH benchmark: Vanilla vs RLM")
    parser.add_argument("--small",              action="store_true", help="Only 4K and 8K contexts")
    parser.add_argument("--long",               action="store_true", help="32K-128K (where RLM shines)")
    parser.add_argument("--lengths",            type=str, default=None, help="Custom lengths: '32000,64000,128000'")
    parser.add_argument("--rlm-only",           action="store_true", help="Skip vanilla baseline")
    parser.add_argument("--vanilla-only",       action="store_true", help="Skip RLM")
    parser.add_argument("--save-trajectories",  action="store_true", help="Save full RLM traces for RL training")
    parser.add_argument("--model",    default="qwen3.5:2b", help="Ollama model tag")
    parser.add_argument("--max-iter", type=int, default=20, help="Max RLM REPL iterations per task")
    parser.add_argument("--run-name", default=None, help="Experiment name (use same name to resume)")
    args = parser.parse_args()

    if args.lengths:
        context_lengths = [int(x.strip()) for x in args.lengths.split(",")]
    elif args.small:
        context_lengths = [4_000, 8_000]
    elif args.long:
        context_lengths = [32_000, 64_000, 128_000]
    else:
        context_lengths = [4_000, 8_000, 16_000, 32_000]
    positions = [0.1, 0.5, 0.9]

    tasks = generate_niah_suite(context_lengths=context_lengths, positions=positions)
    if args.run_name:
        run_name = args.run_name
    elif args.long:
        run_name = "niah_long"
    elif args.small:
        run_name = "niah_small"
    else:
        run_name = "niah_full"

    run_comparison(
        tasks=tasks,
        model=args.model,
        run_name=run_name,
        max_rlm_iterations=args.max_iter,
        skip_vanilla=args.rlm_only,
        skip_rlm=args.vanilla_only,
        save_trajectories=args.save_trajectories,
    )


if __name__ == "__main__":
    main()
