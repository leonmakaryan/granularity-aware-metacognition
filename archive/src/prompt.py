"""Location prompts and answer extraction.

The model answers on a `Location:` line, or `Location: I don't know`.
Without the marker the small models tend to answer in long sentences.
"""

import re

# The deployment prompt.
PROMPT_TEMPLATE = (
    "Answer where the following is located. Reply in exactly this format:\n"
    "Location: <place name>\n\n"
    "Give a single place name after 'Location:'. If you are unsure of the precise "
    "place, name a broader region or country you are confident about instead of "
    "guessing. If you do not know even the country, reply exactly:\n"
    "Location: I don't know\n\n"
    "Question: {q}"
)

# Prompt baselines: the bare question, the deployment prompt without "I don't know",
# and the deployment prompt itself. The bare question has no marker, so its answer is
# the first line.
PLAIN_PROMPT_TEMPLATE = "Question: {q}\nAnswer:"

# The deployment prompt without the "I don't know" option.
NOIDK_PROMPT_TEMPLATE = (
    "Answer where the following is located. Reply in exactly this format:\n"
    "Location: <place name>\n\n"
    "Give a single place name after 'Location:'. If you are unsure of the precise "
    "place, name a broader region or country you are confident about instead of "
    "guessing.\n\n"
    "Question: {q}"
)

# Asks only for the country. A town name does not show whether the model knows the
# country, unlike a date, which contains its own year.
COUNTRY_PROMPT_TEMPLATE = (
    "Answer where the following is located, naming only the country. Reply in exactly "
    "this format:\n"
    "Location: <country name>\n\n"
    "Name a single country after 'Location:'. If you do not know the country, reply "
    "exactly:\n"
    "Location: I don't know\n\n"
    "Question: {q}"
)


# Few-shot variant — only used if the base model emits ~0 IDK in calibration
# rollouts (the Formulation C seeding problem). Demonstrates IDK in-context
# WITHOUT weight-SFT, so it does not erode factual knowledge.
FEWSHOT_PROMPT_TEMPLATE = (
    "Answer where the following is located. Reply in exactly this format:\n"
    "Location: <place name>\n\n"
    "Give a single place name after 'Location:'. If you are unsure of the precise "
    "place, name a broader region or country you are confident about instead of "
    "guessing. If you do not know even the country, reply exactly:\n"
    "Location: I don't know\n\n"
    "Question: Where is Paris located?\n"
    "Location: France\n\n"
    "Question: Where is Qbxztronia located?\n"
    "Location: I don't know\n\n"
    "Question: {q}"
)


def make_prompt(question: str, template: str = PROMPT_TEMPLATE) -> str:
    return template.format(q=question)


# Prompts for the multi-task SFT (sft.py): one asks for a given level (city, region
# or country), one asks which level the model can answer reliably.

FORCE_INSTRUCTION = {
    "city": "Give the most specific place you can (the city, town, or local place).",
    "region": "Give the region, state, or province (not just the country).",
    "country": "Give only the country.",
}

DIRECT_TEMPLATE = (
    "Answer where the following is located. Reply in exactly this format:\n"
    "Location: <place name>\n\n"
    "{instruction}\n\n"
    "Question: {q}"
)

META_TEMPLATE = (
    "For the question below, what is the most specific level you can answer "
    "RELIABLY and correctly? Reply with exactly one of: city, region, country, "
    "or 'I don't know' (use that if you cannot reliably name even the country).\n"
    "Reply in exactly this format:\n"
    "Reliable level: <city / region / country / I don't know>\n\n"
    "Question: {q}"
)


def level_label(hierarchy: list[str], idx: int | None) -> str:
    """Collapse a hierarchy index to a coarse bucket label. idx None -> IDK label.
    idx 0 -> city (most specific), last -> country, anything between -> region."""
    if idx is None:
        return "I don't know"
    if idx == len(hierarchy) - 1:
        return "country"
    if idx == 0:
        return "city"
    return "region"


def make_direct_prompt(question: str, label: str) -> str:
    """Level-conditioned prompt. `label` in {city, region, country}."""
    return DIRECT_TEMPLATE.format(instruction=FORCE_INSTRUCTION[label], q=question)


def make_meta_prompt(question: str) -> str:
    return META_TEMPLATE.format(q=question)


_META_RE = re.compile(
    r"Reliable level:\s*(city|region|country|i\s*don.?t\s*know|idk)",
    re.IGNORECASE)


def extract_meta_label(completion: str) -> str | None:
    """Parse a meta-prompt reply to one of {city, region, country, idk} or None."""
    if not completion:
        return None
    m = _META_RE.search(completion)
    if not m:
        return None
    tok = m.group(1).lower().replace(" ", "")
    if tok in ("city", "region", "country"):
        return tok
    return "idk"


# The rest of the "Location:" line. Anything before or after it is ignored.
LOCATION_RE = re.compile(r"Location:\s*([^\n]+)", re.IGNORECASE)

_IDK_PATTERNS = re.compile(
    r"\b(i\s*don.?t\s*know|dont\s*know|do\s*not\s*know|unknown|not\s*sure|no\s*idea|unsure)\b",
    re.IGNORECASE,
)


def extract_answer(completion: str) -> str:
    """Pull the place name after the 'Location:' marker. Returns '' if absent."""
    if not completion:
        return ""
    m = LOCATION_RE.search(completion)
    if not m:
        return ""
    ans = m.group(1).strip()
    # Cut a sentence that follows the name on the same line ("Czech Republic. It is..."),
    # but keep abbreviations like "St. Louis", then drop a leading article.
    ans = re.split(r"(?<=[^\s.]{3})\.(?:\s|$)", ans, maxsplit=1)[0]
    ans = re.sub(r"^(the|a|an)\s+", "", ans, flags=re.IGNORECASE)
    return ans.strip().rstrip(".,").strip()


def is_idk(answer: str) -> bool:
    """True if the answer is an abstention (explicit IDK phrase or empty)."""
    if not answer or not answer.strip():
        return True
    return bool(_IDK_PATTERNS.search(answer))


def finest_element(answer: str) -> str:
    """The most specific part of an answer: "Tokyo, Japan" -> "Tokyo"."""
    if not answer:
        return ""
    return answer.split(",")[0].strip()
