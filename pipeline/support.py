"""Support match: does an answer sit at the level the base samples support?

For each evaluation question, the K base samples and the 50%-support rule give an
estimated supported level (an estimate, not observed knowledge). Each answer falls
into exactly one category; a format failure counts as a wrong answer.
"""

CATEGORIES = (
    "exact_support_match",
    "correct_coarser_than_support",
    "correct_finer_than_support",
    "wrong_despite_support",
    "abstained_despite_support",
    "answered_without_support_correct",
    "answered_without_support_wrong",
)


def category(support, g: dict) -> str:
    if support is None:
        if g["kind"] == "abstained":
            return "exact_support_match"
        return ("answered_without_support_correct" if g["correct"]
                else "answered_without_support_wrong")
    if g["kind"] == "abstained":
        return "abstained_despite_support"
    if not g["correct"]:
        return "wrong_despite_support"
    if g["level"] == support:
        return "exact_support_match"
    return ("correct_coarser_than_support" if g["level"] > support
            else "correct_finer_than_support")
