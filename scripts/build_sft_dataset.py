"""
Build SFT dataset from RLM trajectories.

Filters trajectories: correct answer + FINAL called + no REPL errors.
Formats as chat-style SFT data compatible with TRL's SFTTrainer.

Usage:
    uv run python scripts/build_sft_dataset.py
    uv run python scripts/build_sft_dataset.py --min-trajectories 10
    uv run python scripts/build_sft_dataset.py --out experiments/sft_dataset.jsonl

Output format (one JSON per line):
    {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Context info: ... Query: ..."},
        {"role": "assistant", "content": "```python\n...\n```"},
        {"role": "user", "content": "[stdout] ..."},
        {"role": "assistant", "content": "```python\nFINAL('answer')\n```"}
    ]}
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _load_system_prompt() -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / "rlm_system.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return "You are an RLM. Write Python code to examine the context variable and answer queries. Call FINAL(answer) when done."


def _build_initial_user_msg(context_length: int, query: str, preview: str = "") -> str:
    """Build the same initial user message the RLM_REPL sends to the root LM."""
    return (
        f"Context info:\n"
        f"  Total length : {context_length:,} characters\n"
        f"  Preview      : {preview}...\n\n"
        f"Query: {query}\n\n"
        f"Write Python code to examine the context and answer the query. "
        f"Call FINAL(answer) when you have the answer."
    )


def _format_code_as_assistant(code: str) -> str:
    """Wrap code in markdown fence like LLM would output."""
    return f"```python\n{code}\n```"


def _format_feedback(stdout: str, stderr: str) -> str:
    """Reproduce the feedback format from rlm_repl._format_feedback."""
    MAX_CHARS = 600
    parts = []
    if stdout:
        truncated = len(stdout) > MAX_CHARS
        preview = stdout[:MAX_CHARS]
        label = f"[stdout | {len(stdout)} chars" + (" | truncated]" if truncated else "]")
        parts.append(f"{label}\n{preview}")
    if stderr:
        error_lines = stderr.strip().splitlines()
        short_error = "\n".join(error_lines[-3:])
        parts.append(f"[REPL ERROR]\n{short_error}\nYour previous code DID NOT WORK. Write different code.")
    elif not stdout:
        parts.append("[no output -- code ran silently, no errors]")
    parts.append("Write new Python code. Call FINAL(answer) when you have the answer.")
    return "\n\n".join(parts)


def trajectory_to_messages(record: dict, system_prompt: str) -> list[dict] | None:
    """
    Convert a trajectory record to SFT chat messages.

    Returns None if the trajectory is not suitable for training.
    """
    # Filter: must be correct, have FINAL called, no REPL errors
    if not record.get("correct", False):
        return None

    trajectory = record.get("trajectory", [])
    if not trajectory:
        return None

    # Must have called FINAL
    if not any(step.get("final_set") for step in trajectory):
        return None

    # Must have no REPL errors
    repl_errors = record.get("repl_errors", [])
    if repl_errors:
        return None

    context_length = record["context_length"]
    query = record["query"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_initial_user_msg(context_length, query)},
    ]

    for i, step in enumerate(trajectory):
        code = step.get("code", "")
        stdout = step.get("stdout", "")
        stderr = step.get("stderr", "")

        # Assistant writes code
        messages.append({"role": "assistant", "content": _format_code_as_assistant(code)})

        # If this is the final step (FINAL was called), stop here
        if step.get("final_set"):
            break

        # Otherwise, user provides feedback for next iteration
        messages.append({"role": "user", "content": _format_feedback(stdout, stderr)})

    return messages


def build_dataset(trajectory_files: list[Path], system_prompt: str) -> list[dict]:
    """Load all trajectories, filter, and convert to SFT format."""
    dataset = []
    total = 0
    seen_queries = set()  # Deduplicate by query

    for fpath in trajectory_files:
        if not fpath.exists():
            print(f"  SKIP (not found): {fpath}")
            continue

        with open(fpath, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                total += 1

                messages = trajectory_to_messages(record, system_prompt)
                if messages is None:
                    continue

                # Deduplicate by (context_length, query)
                dedup_key = f"{record['context_length']}:{record['query']}"
                if dedup_key in seen_queries:
                    continue
                seen_queries.add(dedup_key)

                dataset.append({
                    "messages": messages,
                    "metadata": {
                        "context_length": record["context_length"],
                        "query": record["query"],
                        "answer": record.get("answer"),
                        "iterations": len(record.get("trajectory", [])),
                        "source_file": fpath.name,
                    }
                })

    print(f"  Processed {total} trajectories -> {len(dataset)} SFT examples")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Build SFT dataset from RLM trajectories")
    parser.add_argument("--out", default="experiments/sft_dataset.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--min-trajectories", type=int, default=5,
                        help="Minimum examples required (warn if below)")
    args = parser.parse_args()

    # Find all trajectory files
    exp_dir = Path("experiments")
    traj_files = sorted(exp_dir.glob("*_trajectories.jsonl"))
    print(f"Found {len(traj_files)} trajectory files:")
    for f in traj_files:
        print(f"  {f}")

    system_prompt = _load_system_prompt()
    dataset = build_dataset(traj_files, system_prompt)

    if len(dataset) < args.min_trajectories:
        print(f"\nWARNING: Only {len(dataset)} examples (need {args.min_trajectories}+)")
        print("Run more benchmarks with --save-trajectories to collect more data.")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(dataset)} SFT examples to {out_path}")

    # Print stats
    if dataset:
        iters = [d["metadata"]["iterations"] for d in dataset]
        lens = [d["metadata"]["context_length"] for d in dataset]
        print(f"  Context lengths: {min(lens):,} - {max(lens):,} chars")
        print(f"  Iterations: {min(iters)}-{max(iters)} (avg {sum(iters)/len(iters):.1f})")
        print(f"  Avg messages per example: {sum(len(d['messages']) for d in dataset) / len(dataset):.1f}")


if __name__ == "__main__":
    main()
