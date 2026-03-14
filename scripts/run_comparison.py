"""
Full comparison run: NIAH + Long-Doc QA, Vanilla vs RLM.

Usage:
    uv run python scripts/run_comparison.py
    uv run python scripts/run_comparison.py --small
    uv run python scripts/run_comparison.py --save-trajectories
    uv run python scripts/run_comparison.py --run-name my_run   # resume

Resumable: re-running with the same --run-name skips already-completed tasks.
Results saved to: experiments/<run_name>.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarks.niah import generate_niah_suite
from benchmarks.long_doc_qa import generate_long_doc_suite
from comparison.runner import run_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Full comparison: Vanilla vs RLM")
    parser.add_argument("--small",              action="store_true", help="Only 4K/8K contexts (faster)")
    parser.add_argument("--rlm-only",           action="store_true")
    parser.add_argument("--vanilla-only",       action="store_true")
    parser.add_argument("--save-trajectories",  action="store_true", help="Save full RLM traces for RL training")
    parser.add_argument("--model",    default="qwen3.5:2b")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--run-name", default=None, help="Experiment name (use same name to resume)")
    args = parser.parse_args()

    lengths   = [4_000, 8_000] if args.small else [4_000, 8_000, 16_000, 32_000]
    positions = [0.1, 0.5, 0.9]

    niah_tasks = generate_niah_suite(context_lengths=lengths, positions=positions)
    doc_tasks  = generate_long_doc_suite(context_lengths=lengths)
    all_tasks  = niah_tasks + doc_tasks
    run_name   = args.run_name or f"comparison_{'small' if args.small else 'full'}"

    print(f"Total tasks: {len(all_tasks)} ({len(niah_tasks)} NIAH + {len(doc_tasks)} LongDocQA)")

    run_comparison(
        tasks=all_tasks,
        model=args.model,
        run_name=run_name,
        max_rlm_iterations=args.max_iter,
        skip_vanilla=args.rlm_only,
        skip_rlm=args.vanilla_only,
        save_trajectories=args.save_trajectories,
    )


if __name__ == "__main__":
    main()
