"""
Evaluation and scoring utilities for comparison runs.

All scoring is purely string-based — no neural models needed.
"""

import re
from typing import Any


def exact_match(prediction: str | None, answer: str) -> bool:
    """True if prediction matches answer exactly (case-insensitive, stripped)."""
    if prediction is None:
        return False
    return prediction.strip().lower() == answer.strip().lower()


def contains_answer(prediction: str | None, answer: str) -> bool:
    """True if prediction contains the answer string (case-insensitive)."""
    if prediction is None:
        return False
    return answer.strip().lower() in prediction.strip().lower()


def numeric_match(prediction: str | None, answer: str) -> bool:
    """
    True if the first number in prediction matches the first number in answer.
    Falls back to contains_answer if no numbers found.
    """
    if prediction is None:
        return False
    pred_nums = re.findall(r"\d+", prediction.replace(",", ""))
    ans_nums = re.findall(r"\d+", answer.replace(",", ""))
    if pred_nums and ans_nums:
        return pred_nums[0] == ans_nums[0]
    return contains_answer(prediction, answer)


def score_result(prediction: str | None, answer: str) -> dict[str, bool]:
    """Score one prediction against ground truth. Returns all metric flags."""
    return {
        "exact_match": exact_match(prediction, answer),
        "contains_answer": contains_answer(prediction, answer),
        "numeric_match": numeric_match(prediction, answer),
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics over a list of result dicts from runner."""
    if not results:
        return {}

    def _mean(vals: list) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def _summarise(subset: list[dict]) -> dict:
        if not subset:
            return {}
        return {
            "n": len(subset),
            "contains_answer_rate": _mean([r["scores"]["contains_answer"] for r in subset]),
            "exact_match_rate": _mean([r["scores"]["exact_match"] for r in subset]),
            "numeric_match_rate": _mean([r["scores"]["numeric_match"] for r in subset]),
            "avg_latency_s": _mean([r.get("latency_s", 0.0) for r in subset]),
            "timeout_count": sum(1 for r in subset if r.get("timed_out", False)),
            "error_count": sum(1 for r in subset if r.get("error")),
        }

    vanilla = [r for r in results if r.get("method") == "vanilla"]
    rlm = [r for r in results if r.get("method") == "rlm"]

    return {
        "total_tasks": len(results),
        "vanilla": _summarise(vanilla),
        "rlm": _summarise(rlm),
    }
