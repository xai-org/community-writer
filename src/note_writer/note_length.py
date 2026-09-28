import re


URL_RE = re.compile(
    r"(?i)(?<![\w@.])(?:"
    r"https?://\S+"
    r"|www\.\S+"
    r"|(?:[a-z0-9-]+\.)+[a-z]{2,}/\S*"
    r"|(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|mil|int|io|co|ai|info|news)\b"
    r")"
)

URL_TRAILING = ".,;:!?)]}>\"'"


NOTE_CHAR_LIMIT = 280


LENGTH_RESTRICTION_PLACEHOLDER = "{length_restriction}"


def url_collapsed_char_length(text: str) -> int:
    stripped = (text or "").strip()
    length = len(stripped)
    for match in URL_RE.finditer(stripped):
        url = match.group(0).rstrip(URL_TRAILING)
        length -= len(url) - 1
    return length


def is_over_length(note_text: str | None) -> bool:
    if not note_text:
        return False
    return url_collapsed_char_length(note_text) > NOTE_CHAR_LIMIT


def supports_length_restriction(writer_prompt: str, length_restriction: str) -> bool:
    return bool(length_restriction) and LENGTH_RESTRICTION_PLACEHOLDER in writer_prompt
