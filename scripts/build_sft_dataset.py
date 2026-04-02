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

Changes (2026-03-20):
    - Fixed: handle missing/malformed trajectory fields gracefully
    - Fixed: handle empty/whitespace-only JSONL lines
    - Fixed: validate messages have minimum required structure
    - Added: --allow-errors flag to include trajectories with recoverable errors
    - Added: statistics on rejection reasons
    - Added: context preview in initial user msg for better SFT quality
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
    lines = [
        "Context info:",
        f"  Total length : {context_length:,} characters",
    ]
    if preview:
        lines.append(f"  Preview      : {preview}...")
    lines.extend([
        "",
        f"Query: {query}",
        "",
        "Write Python code to examine the context and answer the query. "
        "Call FINAL(answer) when you have the answer.",
    ])
    return "\n".join(lines)


def _format_code_as_assistant(code: str) -> str:
    """Wrap code in markdown fence like LLM would output."""
    if not code:
        return "```python\n# no code\n```"
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


def trajectory_to_messages(
    record: dict,
    system_prompt: str,
    allow_errors: bool = False,
) -> tuple[list[dict] | None, str]:
    """
    Convert a trajectory record to SFT chat messages.

    Returns:
        (messages, rejection_reason) — messages is None if rejected, with reason.
    """
    # Validate required fields
    if "trajectory" not in record:
        return None, "missing_trajectory_field"

    trajectory = record.get("trajectory")
    if not trajectory or not isinstance(trajectory, list):
        return None, "empty_trajectory"

    # Must be marked correct
    if not record.get("correct", False):
        return None, "incorrect_answer"

    # Must have called FINAL
    has_final = False
    for step in trajectory:
        if isinstance(step, dict) and step.get("final_set"):
            has_final = True
            break
    if not has_final:
        return None, "no_final_called"

    # Check REPL errors (unless allow_errors is True)
    if not allow_errors:
        repl_errors = record.get("repl_errors", [])
        if repl_errors:
            return None, "has_repl_errors"

        # Also check per-step stderr
        for step in trajectory:
            if isinstance(step, dict) and step.get("stderr"):
                return None, "has_step_errors"

    # Build context info
    context_length = record.get("context_length", 0)
    query = record.get("query", "")
    if not query:
        return None, "missing_query"

    # Try to get a preview from the record (may not exist)
    preview = record.get("context_preview", "")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_initial_user_msg(context_length, query, preview)},
    ]

    for i, step in enumerate(trajectory):
        if not isinstance(step, dict):
            return None, f"invalid_step_at_{i}"

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

    # Validate message structure: must alternate user/assistant properly
    # Minimum: system + user + assistant = 3 messages
    if len(messages) < 3:
        return None, "too_few_messages"

    # Last message must be from assistant (the FINAL call)
    if messages[-1]["role"] != "assistant":
        return None, "last_message_not_assistant"

    return messages, ""


def build_dataset(
    trajectory_files: list[Path],
    system_prompt: str,
    allow_errors: bool = False,
) -> tuple[list[dict], dict[str, int]]:
    """Load all trajectories, filter, and convert to SFT format."""
    dataset = []
    total = 0
    seen_queries: set[str] = set()  # Deduplicate by query
    rejection_stats: dict[str, int] = {}

    for fpath in trajectory_files:
        if not fpath.exists():
            print(f"  SKIP (not found): {fpath}")
            continue

        with open(fpath, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Handle malformed JSON gracefully
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  WARNING: Skipping malformed JSON at {fpath.name}:{line_num}: {e}")
                    rejection_stats["json_parse_error"] = rejection_stats.get("json_parse_error", 0) + 1
                    continue

                if not isinstance(record, dict):
                    rejection_stats["not_a_dict"] = rejection_stats.get("not_a_dict", 0) + 1
                    continue

                total += 1

                messages, reason = trajectory_to_messages(record, system_prompt, allow_errors)
                if messages is None:
                    rejection_stats[reason] = rejection_stats.get(reason, 0) + 1
                    continue

                # Deduplicate by (context_length, query)
                ctx_len = record.get("context_length", 0)
                query = record.get("query", "")
                dedup_key = f"{ctx_len}:{query}"
                if dedup_key in seen_queries:
                    rejection_stats["duplicate"] = rejection_stats.get("duplicate", 0) + 1
                    continue
                seen_queries.add(dedup_key)

                dataset.append({
                    "messages": messages,
                    "metadata": {
                        "context_length": ctx_len,
                        "query": query,
                        "answer": record.get("answer"),
                        "iterations": len(record.get("trajectory", [])),
                        "source_file": fpath.name,
                    }
                })

    print(f"  Processed {total} trajectories -> {len(dataset)} SFT examples")
    return dataset, rejection_stats


def main():
    parser = argparse.ArgumentParser(description="Build SFT dataset from RLM trajectories")
    parser.add_argument("--out", default="experiments/sft_dataset.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--min-trajectories", type=int, default=5,
                        help="Minimum examples required (warn if below)")
    parser.add_argument("--allow-errors", action="store_true",
                        help="Include trajectories that had recoverable REPL errors")
    args = parser.parse_args()

    # Find all trajectory files
    exp_dir = Path("experiments")
    if not exp_dir.exists():
        print(f"ERROR: experiments/ directory not found. Run benchmarks first.")
        sys.exit(1)

    traj_files = sorted(exp_dir.glob("*_trajectories.jsonl"))
    if not traj_files:
        print("ERROR: No trajectory files found in experiments/")
        print("Run benchmarks with --save-trajectories flag first:")
        print("  uv run python scripts/run_niah.py --save-trajectories")
        sys.exit(1)

    print(f"Found {len(traj_files)} trajectory files:")
    for f in traj_files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")

    system_prompt = _load_system_prompt()
    dataset, rejection_stats = build_dataset(
        traj_files, system_prompt, allow_errors=args.allow_errors
    )

    # Print rejection stats
    if rejection_stats:
        print(f"\n  Rejection breakdown:")
        for reason, count in sorted(rejection_stats.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    if len(dataset) < args.min_trajectories:
        print(f"\nWARNING: Only {len(dataset)} examples (need {args.min_trajectories}+)")
        print("Run more benchmarks with --save-trajectories to collect more data.")
        if len(dataset) == 0:
            print("No usable trajectories found. Exiting without creating file.")
            sys.exit(1)

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
        msg_counts = [len(d["messages"]) for d in dataset]
        print(f"  Context lengths: {min(lens):,} - {max(lens):,} chars")
        print(f"  Iterations: {min(iters)}-{max(iters)} (avg {sum(iters)/len(iters):.1f})")
        print(f"  Messages per example: {min(msg_counts)}-{max(msg_counts)} (avg {sum(msg_counts)/len(msg_counts):.1f})")


if __name__ == "__main__":
    main()
