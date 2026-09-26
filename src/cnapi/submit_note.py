import json

from requests_oauthlib import OAuth1Session  # type: ignore


def _extract_api_error_message(response_text: str) -> str:
    try:
        body = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        return response_text

    parts: list[str] = []

    if "errors" in body and isinstance(body["errors"], list):
        for err in body["errors"]:
            if isinstance(err, dict) and "message" in err:
                parts.append(err["message"])

    if "detail" in body:
        parts.append(body["detail"])

    if parts:
        return "; ".join(parts)

    if "title" in body:
        return body["title"]

    return response_text


class NoteSubmissionError(Exception):
    _SHORT_FORMS = [
        (
            "Note should contain at least 1 and at most 280 characters",
            "Note should contain >1 and <280 characters.",
        ),
        ("does not match the regex pattern", "Note doesn't match regex"),
    ]

    def __init__(self, status_code: int, response_text: str):
        self.status_code = status_code
        raw_message = _extract_api_error_message(response_text)
        for pattern, short in self._SHORT_FORMS:
            if pattern in raw_message:
                self.message = short
                super().__init__(f"API: {short}")
                return
        self.message = raw_message
        super().__init__(f"API: {raw_message} (status={status_code})")


def submit_note(
    oauth: OAuth1Session,
    note_text: str,
    post_id: int,
    misleading_tags: list[str],
) -> int:
    payload = {
        "test_mode": False,
        "post_id": str(post_id),
        "info": {
            "text": note_text,
            "classification": "misinformed_or_potentially_misleading",
            "misleading_tags": misleading_tags,
            "trustworthy_sources": True,
        },
    }

    url = "https://api.x.com/2/notes"

    response = oauth.post(url, json=payload)
    try:
        response.raise_for_status()
    except Exception as e:
        raise NoteSubmissionError(response.status_code, response.text) from e

    data = response.json()
    return int(data["data"]["id"])
