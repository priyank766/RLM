"""
Long Document QA Benchmark Generator.

Generates a synthetic document with multiple facts scattered throughout
padded with filler text. Requires finding a specific fact buried in the
document — harder than NIAH because filler text is more coherent.
"""

import random
from dataclasses import dataclass


@dataclass
class LongDocTask:
    context: str
    query: str
    answer: str
    context_length: int
    task_type: str = "long_doc_qa"
    seed: int = 0


# (fact_type, fact_template, query_template, answer_template)
_FACT_TEMPLATES = [
    (
        "city",
        "The population of {city} is {value} people.",
        "What is the population of {city}?",
        "{value}",
    ),
    (
        "river",
        "The length of the {river} River is {value} kilometers.",
        "How long is the {river} River in kilometers?",
        "{value}",
    ),
    (
        "mountain",
        "Mount {mountain} has an elevation of {value} meters above sea level.",
        "What is the elevation of Mount {mountain} in meters?",
        "{value}",
    ),
    (
        "country",
        "The official capital city of {country} is {value}.",
        "What is the official capital city of {country}?",
        "{value}",
    ),
    (
        "scientist",
        "{scientist} received the award in the year {value}.",
        "In which year did {scientist} receive the award?",
        "{value}",
    ),
    (
        "company",
        "{company} was officially founded in {value}.",
        "In what year was {company} officially founded?",
        "{value}",
    ),
    (
        "element",
        "The atomic number of {element} is {value}.",
        "What is the atomic number of {element}?",
        "{value}",
    ),
]

_ENTITIES: dict[str, list[str]] = {
    "city": ["Springfield", "Riverdale", "Maplewood", "Lakeside", "Hillcrest", "Fairview"],
    "river": ["Azure", "Silver", "Crimson", "Golden", "Amber", "Sapphire"],
    "mountain": ["Ironpeak", "Frosthorn", "Sunridge", "Cloudtop", "Stoneback"],
    "country": ["Valdoria", "Kestria", "Nelmoor", "Branthia", "Selvaan"],
    "scientist": ["Dr. Elena Marsh", "Prof. James Wick", "Dr. Yuki Tanaka", "Prof. Marcus Bell"],
    "company": ["Axiom Systems", "Nexatech", "Brightpath", "CoreLogic"],
    "element": ["Zynthium", "Veridian", "Optalite", "Ferroxide"],
    # Capital city values
    "capital": ["Aurentum", "Veldris", "Calhourn", "Miraste", "Telvana"],
}

_FILLER_SENTENCES = [
    "The weather in this region is generally mild throughout the year.",
    "Researchers have conducted numerous studies on this subject over the decades.",
    "Historical records indicate significant changes in this area over time.",
    "Local authorities have implemented several measures to address various concerns.",
    "According to recent surveys, public opinion on this matter has shifted considerably.",
    "The economic impact of these developments has been widely debated by experts.",
    "Scientific evidence supports the current understanding of this phenomenon.",
    "Traditional practices in this field have evolved significantly in recent years.",
    "The geographical features of this region influence many aspects of daily life.",
    "International cooperation has played a key role in advancing research here.",
    "Statistical analysis reveals interesting patterns in the available data.",
    "Community leaders have expressed various perspectives on these ongoing changes.",
    "Historical documents from this period provide valuable insights into the era.",
    "Technological advancements have transformed how this process works in practice.",
    "Environmental factors play a crucial role in determining long-term outcomes.",
    "The committee reviewed all available evidence before reaching its final decision.",
    "Multiple factors contribute to the complexity of this ongoing situation.",
    "The implementation of new policies has had mixed results across different regions.",
    "Stakeholders from various sectors have provided detailed input on this matter.",
    "Long-term planning remains essential for addressing these multifaceted challenges.",
    "Further investigation is needed to fully understand the broader implications.",
    "The results were subsequently published and reviewed by independent researchers.",
    "New methodologies have been introduced to improve overall operational efficiency.",
    "Experts continue to disagree about the best approach to handling this situation.",
    "The project has been under active development for several years now.",
]


def generate_long_doc_task(
    target_chars: int,
    n_facts: int = 5,
    query_fact_index: int = 2,
    seed: int = 42,
) -> LongDocTask:
    """
    Generate a long document with n_facts embedded, then query about one.

    Args:
        target_chars: Approximate context length in characters.
        n_facts: Number of distinct facts to embed in the document.
        query_fact_index: Which fact (0-indexed) to ask about.
        seed: Random seed.
    """
    rng = random.Random(seed)
    n_facts = min(n_facts, len(_FACT_TEMPLATES))
    query_fact_index = max(0, min(query_fact_index, n_facts - 1))

    # Select and instantiate facts
    templates = rng.sample(_FACT_TEMPLATES, n_facts)
    facts: list[dict[str, str]] = []
    for fact_type, fact_tmpl, q_tmpl, a_tmpl in templates:
        entity = rng.choice(_ENTITIES[fact_type])
        if fact_type == "country":
            value = rng.choice(_ENTITIES["capital"])
        elif fact_type in ("scientist", "company"):
            value = str(rng.randint(1950, 2020))
        else:
            value = str(rng.randint(1_000, 99_999))
        facts.append({
            "sentence": fact_tmpl.format(**{fact_type: entity, "value": value}),
            "query": q_tmpl.format(**{fact_type: entity}),
            "answer": a_tmpl.format(**{fact_type: entity, "value": value}),
        })

    target_fact = facts[query_fact_index]
    filler_per_gap = target_chars // (n_facts + 1)

    # Build document: filler → fact → filler → fact → ...
    sections: list[str] = []
    for fact in facts:
        sections.append(_generate_filler(filler_per_gap, rng))
        sections.append(fact["sentence"])

    sections.append(_generate_filler(filler_per_gap // 2, rng))
    context = "\n\n".join(sections)

    return LongDocTask(
        context=context,
        query=target_fact["query"],
        answer=target_fact["answer"],
        context_length=len(context),
        seed=seed,
    )


def generate_long_doc_suite(
    context_lengths: list[int] | None = None,
    seed: int = 42,
) -> list[LongDocTask]:
    """Generate a suite of long-doc QA tasks at increasing context lengths."""
    if context_lengths is None:
        context_lengths = [4_000, 8_000, 16_000, 32_000]

    tasks: list[LongDocTask] = []
    for i, length in enumerate(context_lengths):
        tasks.append(
            generate_long_doc_task(
                target_chars=length,
                n_facts=5,
                query_fact_index=2,
                seed=seed + i * 17,
            )
        )
    return tasks


def _generate_filler(target_chars: int, rng: random.Random) -> str:
    """Generate coherent-ish filler text of approximately target_chars characters."""
    sentences: list[str] = []
    current = 0
    while current < target_chars:
        s = rng.choice(_FILLER_SENTENCES)
        sentences.append(s)
        current += len(s) + 1
    return " ".join(sentences)
