import re


def strip_brackets(text: str) -> str:
    result = []

    paren_depth = 0
    square_depth = 0
    curly_depth = 0

    for char in text:
        if char == "(":
            paren_depth += 1
            continue

        if char == ")":
            paren_depth = max(0, paren_depth - 1)
            continue

        if char == "[":
            square_depth += 1
            continue

        if char == "]":
            square_depth = max(0, square_depth - 1)
            continue

        if char == "{":
            curly_depth += 1
            continue

        if char == "}":
            curly_depth = max(0, curly_depth - 1)
            continue

        if paren_depth == square_depth == curly_depth == 0:
            result.append(char)

    text = "".join(result)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    return text.strip()


def extract_markdown_formatting(
    text: str,
    start: int,
    end: int,
) -> tuple[int, int, str]:
    ALLOWED_PREFIXES = {
        "",
        "in",
        "in the",
        "in the year",
        "on",
        "on the",
        "the",
    }

    marker = ""
    for candidate in ("**", "__", "*", "_"):
        opening = text.rfind(candidate, 0, start)
        closing = text.find(candidate, end)

        if opening == -1 or closing == -1:
            continue

        inside_prefix = text[opening + len(candidate):start].strip().lower()

        if inside_prefix not in ALLOWED_PREFIXES:
            continue

        # only expand if there is no same marker between opening and start
        # and no same marker between end and closing

        if (
            text.find(candidate, opening + len(candidate), start) == -1
            and text.find(candidate, end, closing) == -1
        ):
            return (
                opening,
                closing + len(candidate),
                candidate,
            )

    return start, end, marker