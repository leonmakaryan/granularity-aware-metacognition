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

# The bare question, without instructions or marker.
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


def make_prompt(question: str, template: str = PROMPT_TEMPLATE) -> str:
    return template.format(q=question)



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
