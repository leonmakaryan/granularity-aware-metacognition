"""Prompt and answer extraction for the entity relations (author, creator, owner, ...).

These relations have no fixed hierarchy. Coarser answers are descriptions,
"August Derleth" -> "an American novelist" -> "a novelist".
"""

import re

from .text import find_match_level_v3

# Ordered by how well-defined "coarser" is, best first. P69 (educated at) is last
# because its hierarchies are really places.
ENTITY_RELATIONS = ["P50", "P170", "P264", "P127", "P176", "P112", "P175", "P69"]

ENTITY_PROMPT_TEMPLATE = (
    "Answer the following question. Reply in exactly this format:\n"
    "Answer: <answer>\n\n"
    "Give a single answer after 'Answer:'. If you are unsure of the precise "
    "answer, give a broader description you are confident about instead of "
    "guessing. If you know nothing at all, reply exactly:\n"
    "Answer: I don't know\n\n"
    "Question: {q}"
)

ANSWER_RE = re.compile(r"Answer:\s*([^\n]+)", re.IGNORECASE)


def make_entity_prompt(question: str) -> str:
    return ENTITY_PROMPT_TEMPLATE.format(q=question)


def extract_entity_answer(completion: str) -> str:
    """Text after the 'Answer:' marker, '' if the marker is absent."""
    if not completion:
        return ""
    m = ANSWER_RE.search(completion)
    return m.group(1).strip() if m else ""


def entity_match_level(answer: str, hierarchy: list[str]) -> int | None:
    """Level index the answer earns, using the fixed (v3) matcher."""
    if not answer:
        return None
    return find_match_level_v3(answer, hierarchy)
