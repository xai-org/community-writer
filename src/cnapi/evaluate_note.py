import asyncio

from requests_oauthlib import OAuth1Session  # type: ignore


class NoteEvaluationError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Note evaluation failed (status {status_code}): {message}")


async def evaluate_note(
    oauth: OAuth1Session,
    note_text: str,
    post_id: int,
    retries: int = 3,
) -> float:
    payload = {
        "note_text": note_text,
        "post_id": str(post_id),
    }
    url = "https://api.x.com/2/evaluate_note"

    if retries < 1:
        raise ValueError(f"retries must be >= 1, got {retries}")

    for attempt in range(1, retries + 1):
        try:
            response = await asyncio.to_thread(oauth.post, url, json=payload)

            if response.ok:
                response_data = response.json()
                claim_opinion_score = response_data.get("data", {}).get(
                    "claim_opinion_score"
                )
                if claim_opinion_score is None:
                    raise ValueError(
                        "Response did not contain claim_opinion_score in expected format"
                    )
                return claim_opinion_score

            raise NoteEvaluationError(response.status_code, response.text)
        except Exception:
            if attempt == retries:
                raise
            await asyncio.sleep(2.0**attempt)

    raise RuntimeError("evaluate_note: retry loop exited without returning or raising")
