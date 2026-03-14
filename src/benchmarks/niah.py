"""
Needle in a Haystack (NIAH) Benchmark Generator.

A specific fact (the needle) is hidden inside a long document of random
words (the haystack). The model must retrieve that exact fact.

This is the classic long-context stress test — simple to score,
unambiguous ground truth, controllable difficulty via context length
and needle position.
"""

import random
from dataclasses import dataclass, field


@dataclass
class NIAHTask:
    context: str
    query: str
    answer: str
    context_length: int
    needle_position_pct: float  # 0.0 = start, 1.0 = end
    task_type: str = "niah"
    seed: int = 0


# Common English words to build the haystack
_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "great", "between", "need",
    "large", "often", "hand", "high", "place", "hold", "turn", "were", "main",
    "find", "long", "down", "part", "last", "right", "move", "thing", "general",
    "before", "here", "through", "same", "help", "such", "old", "line", "both",
]

# (sentence_template, query_template, answer_template)
_NEEDLE_TEMPLATES = [
    (
        "The secret passcode is {value}.",
        "What is the secret passcode?",
        "{value}",
    ),
    (
        "The hidden reference number is {value}.",
        "What is the hidden reference number?",
        "{value}",
    ),
    (
        "The special activation code is {value}.",
        "What is the special activation code?",
        "{value}",
    ),
    (
        "The unique identifier for this record is {value}.",
        "What is the unique identifier for this record?",
        "{value}",
    ),
    (
        "The verification key has been set to {value}.",
        "What has the verification key been set to?",
        "{value}",
    ),
]


def generate_niah_task(
    target_chars: int,
    needle_position_pct: float = 0.5,
    seed: int = 42,
) -> NIAHTask:
    """
    Generate one NIAH task.

    Args:
        target_chars: Approximate context length in characters.
        needle_position_pct: Where to insert needle (0.0 = start, 1.0 = end).
        seed: Random seed for reproducibility.
    """
    rng = random.Random(seed)

    tmpl_sentence, tmpl_query, tmpl_answer = rng.choice(_NEEDLE_TEMPLATES)
    needle_value = str(rng.randint(10_000, 99_999))
    needle_sentence = tmpl_sentence.format(value=needle_value)
    query = tmpl_query
    answer = tmpl_answer.format(value=needle_value)

    haystack = _generate_haystack(target_chars, rng)

    # Insert needle at the requested fractional position (word boundary)
    insert_char = int(len(haystack) * needle_position_pct)
    snap = haystack.rfind(" ", 0, insert_char)
    if snap == -1:
        snap = 0

    context = haystack[:snap] + f" {needle_sentence} " + haystack[snap:]

    return NIAHTask(
        context=context,
        query=query,
        answer=answer,
        context_length=len(context),
        needle_position_pct=needle_position_pct,
        seed=seed,
    )


def generate_niah_suite(
    context_lengths: list[int] | None = None,
    positions: list[float] | None = None,
    seed: int = 42,
) -> list[NIAHTask]:
    """
    Generate a suite of NIAH tasks across context lengths and positions.

    Default grid: 4 lengths × 3 positions = 12 tasks.
    """
    if context_lengths is None:
        context_lengths = [4_000, 8_000, 16_000, 32_000]
    if positions is None:
        positions = [0.1, 0.5, 0.9]

    tasks: list[NIAHTask] = []
    for length in context_lengths:
        for pos in positions:
            task_seed = seed + int(length / 100) + int(pos * 1000)
            tasks.append(generate_niah_task(length, pos, seed=task_seed))
    return tasks


def _generate_haystack(target_chars: int, rng: random.Random) -> str:
    """Build a random word document of approximately target_chars characters."""
    words: list[str] = []
    current = 0
    while current < target_chars:
        w = rng.choice(_WORDS)
        words.append(w)
        current += len(w) + 1
    return " ".join(words)
