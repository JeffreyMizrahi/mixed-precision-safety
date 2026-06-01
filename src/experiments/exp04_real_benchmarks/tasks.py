STANDARD = [
    "mmlu",
    "hellaswag",
    "arc_challenge",
    "winogrande",
    "gsm8k",
]

SAFETY = [
    "truthfulqa_mc1",
    "truthfulqa_mc2",
    "bbq",
    "toxigen",
]

REASONING = [
    "bbh",
    "minerva_math",
    "humaneval",
]

PERPLEXITY = [
    "wikitext",
]

SUITES = {
    "standard": STANDARD,
    "safety": SAFETY,
    "reasoning": REASONING,
    "perplexity": PERPLEXITY,
    "all": STANDARD + SAFETY + REASONING,
}


def resolve(suite_names: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for s in suite_names:
        if s in SUITES:
            tasks = SUITES[s]
        else:
            tasks = [s]
        for t in tasks:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out
