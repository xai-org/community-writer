from requests_oauthlib import OAuth1Session  # type: ignore


class NoteDeletionError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(
            f"Note deletion failed (status={status_code}, message={message})"
        )


def delete_note(
    oauth: OAuth1Session,
    note_id: int,
) -> bool:
    url = f"https://api.x.com/2/notes/{note_id}"
    response = oauth.delete(url)
    try:
        response.raise_for_status()
    except Exception as e:
        raise NoteDeletionError(response.status_code, response.text) from e
    data = response.json()
    return data.get("data", {}).get("deleted", False)
