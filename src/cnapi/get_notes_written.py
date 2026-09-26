from urllib.parse import urlencode

from requests_oauthlib import OAuth1Session  # type: ignore

from data_models.writer_data_models import (
    FactorBucketCounts,
    NoteRatings,
    NoteStatus,
    TagCount,
    TestResult,
)

from utils.snowflake import (
    get_timestamp_from_snowflake as _get_timestamp_from_snowflake,
)

from utils.log_setup import get_logger

logger = get_logger("cnapi")


_EXPECTED_MODEL = "expansion"


def _parse_bucket(raw: dict) -> FactorBucketCounts:
    return FactorBucketCounts(
        helpful_count=raw.get("helpful_count", 0),
        not_helpful_count=raw.get("not_helpful_count", 0),
        somewhat_helpful_count=raw.get("somewhat_helpful_count", 0),
        helpful_tag_counts=[
            TagCount(tag_name=t["tag_name"], tag_count=t["tag_count"])
            for t in raw.get("helpful_tag_counts", [])
        ],
        not_helpful_tag_counts=[
            TagCount(tag_name=t["tag_name"], tag_count=t["tag_count"])
            for t in raw.get("not_helpful_tag_counts", [])
        ],
    )


def _parse_ratings(rating_counts_per_model: list[dict]) -> NoteRatings | None:
    if not rating_counts_per_model:
        return None

    if len(rating_counts_per_model) > 1:
        model_names = [
            entry.get("model_name", "<unknown>") for entry in rating_counts_per_model
        ]
        logger.info(
            f"Warning — expected 1 model in "
            f"rating_counts_per_model but found {len(rating_counts_per_model)}: "
            f"{model_names}"
        )

    expansion_entry = None
    for entry in rating_counts_per_model:
        if entry.get("model_name", "").lower() == _EXPECTED_MODEL:
            expansion_entry = entry
            break

    if expansion_entry is None:
        model_names = [
            entry.get("model_name", "<unknown>") for entry in rating_counts_per_model
        ]
        logger.error(
            f"Error — expected model '{_EXPECTED_MODEL}' in "
            f"rating_counts_per_model but found: {model_names}"
        )
        return None

    value = expansion_entry.get("value", {})
    return NoteRatings(
        negative=_parse_bucket(value.get("negative_factor_bucket_counts", {})),
        neutral=_parse_bucket(value.get("neutral_factor_bucket_counts", {})),
        positive=_parse_bucket(value.get("positive_factor_bucket_counts", {})),
    )


def get_notes_written(
    oauth: OAuth1Session,
    max_results: int | None = None,
    min_created_at_ms: int | None = None,
) -> list[NoteStatus]:
    all_notes: list[NoteStatus] = []
    pagination_token: str | None = None

    while True:
        base_url = "https://api.x.com/2/notes/search/notes_written"
        params = {
            "test_mode": "false",
            "max_results": 100,
        }
        if pagination_token:
            params["pagination_token"] = pagination_token
        url = f"{base_url}?{urlencode(params)}"

        response = oauth.get(url)
        response.raise_for_status()
        data = response.json()

        notes_data = data.get("data", [])
        for note_item in notes_data:
            test_result_list: list[TestResult] | None = None
            test_result_raw = note_item.get("test_result")
            if test_result_raw and isinstance(test_result_raw, dict):
                evaluation_outcomes = test_result_raw.get("evaluation_outcome", [])
                if evaluation_outcomes:
                    test_result_list = [
                        TestResult(
                            evaluator_score_bucket=outcome["evaluator_score_bucket"],
                            evaluator_type=outcome["evaluator_type"],
                        )
                        for outcome in evaluation_outcomes
                    ]

            ratings: NoteRatings | None = None
            scoring_status = note_item.get("scoring_status")
            if scoring_status and scoring_status.get("has_access"):
                ratings = _parse_ratings(
                    scoring_status.get("rating_counts_per_model", [])
                )

            note_status = NoteStatus(
                note_id=int(note_item["id"]),
                post_id=int(note_item["info"]["post_id"]),
                note_text=note_item["info"]["text"],
                status=note_item["status"],
                test_result=test_result_list,
                ratings=ratings,
            )
            all_notes.append(note_status)

            if max_results is not None and len(all_notes) >= max_results:
                return all_notes[:max_results]

        if min_created_at_ms is not None and notes_data:
            oldest_id = min(int(item["id"]) for item in notes_data)
            if _get_timestamp_from_snowflake(oldest_id) < min_created_at_ms:
                break

        meta = data.get("meta", {})
        pagination_token = meta.get("next_token")

        if not pagination_token:
            break

    return all_notes
