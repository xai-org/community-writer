from pydantic import BaseModel


class DeletedNote(BaseModel):
    note_id: int
    submitter: str
    model_score: float | None
    policy_names: list[str]
    total_ratings: int
    nonzero_rating_counts: dict[str, int]
