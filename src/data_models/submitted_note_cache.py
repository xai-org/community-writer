import asyncio
import glob
import os
import time

import pandas as pd
from pydantic import BaseModel
from requests_oauthlib import OAuth1Session  # type: ignore

from cnapi.get_notes_written import get_notes_written
from data_models.writer_data_models import NoteRatings, NoteStatus
from deletion_model.predict import get_notes_to_delete
from utils.snowflake import (
    get_timestamp_from_snowflake as _get_timestamp_from_snowflake,
)

from utils.log_setup import get_logger

logger = get_logger("note_cache")


_CRH = "CURRENTLY_RATED_HELPFUL"
_CRNH = "CURRENTLY_RATED_NOT_HELPFUL"
_MAX_ROW_AGE_MS = 30 * 86_400_000
_MIN_RATINGS_FOR_MAX_SCORE = 3


_HELPFUL_TAGS = [
    "important_context",
    "addresses_claim",
    "good_sources",
    "unbiased_language",
    "clear",
    "helpful_other",
]
_NOT_HELPFUL_TAGS = [
    "no_sources",
    "missing_key_points",
    "opinion_speculation",
    "rude",
    "not_helpful_other",
    "incorrect",
    "note_not_needed",
    "unclear",
    "twitter_violation_any",
    "irrelevant_sources",
]
_BUCKETS = ["negative", "neutral", "positive"]


_HELPFUL_TAG_NAME_MAP = {
    "ImportantContext": "important_context",
    "AddressesClaim": "addresses_claim",
    "GoodSources": "good_sources",
    "UnbiasedLanguage": "unbiased_language",
    "Clear": "clear",
    "Other": "helpful_other",
}
_NOT_HELPFUL_TAG_NAME_MAP = {
    "NoSources": "no_sources",
    "MissingKeyPoints": "missing_key_points",
    "OpinionSpeculation": "opinion_speculation",
    "Rude": "rude",
    "Other": "not_helpful_other",
    "Incorrect": "incorrect",
    "NoteNotNeeded": "note_not_needed",
    "Unclear": "unclear",
    "TwitterViolationAny": "twitter_violation_any",
    "IrrelevantSources": "irrelevant_sources",
}


_RATINGS_COLUMNS: list[str] = []
for _bucket in _BUCKETS:
    _RATINGS_COLUMNS.append(f"{_bucket}_helpful")
    _RATINGS_COLUMNS.append(f"{_bucket}_not_helpful")
    _RATINGS_COLUMNS.append(f"{_bucket}_somewhat_helpful")
    for _tag in _HELPFUL_TAGS:
        _RATINGS_COLUMNS.append(f"{_bucket}_{_tag}")
    for _tag in _NOT_HELPFUL_TAGS:
        _RATINGS_COLUMNS.append(f"{_bucket}_{_tag}")
_RATINGS_COLUMNS.append("ratings_updated_at_millis")


_COLUMNS = [
    "note_id",
    "post_id",
    "note_text",
    "post_text",
    "author_id",
    "username",
    "lang",
    "created_at_millis",
    "enqueued_at",
    "api_feed",
    "timed_feed",
    "notable_post_prediction",
    "writer_name",
    "submitter",
    "retweet_count",
    "reply_count",
    "like_count",
    "quote_count",
    "bookmark_count",
    "impression_count",
    "num_unique_sources",
    "total_source_suggestions",
    "num_note_request_suggestions",
    "has_photo",
    "has_video",
    "author_followers_count",
    "author_following_count",
    "author_tweet_count",
    "author_listed_count",
    "author_like_count",
    "author_media_count",
    "author_verified_type",
    "author_parody",
    "hist_note_count",
    "hist_crh_count",
    "hist_crnh_count",
    "hist_total_ratings",
    "current_status",
    "first_status",
    "deleted_at_millis",
    "deletion_policy_names",
    "deletion_model_score",
    "max_deletion_model_score",
    *_RATINGS_COLUMNS,
]

_DTYPES: dict[str, object] = {
    "note_id": pd.Int64Dtype(),
    "post_id": pd.Int64Dtype(),
    "writer_name": pd.StringDtype(),
    "submitter": pd.StringDtype(),
    "created_at_millis": pd.Int64Dtype(),
    "note_text": pd.StringDtype(),
    "post_text": pd.StringDtype(),
    "author_id": pd.Int64Dtype(),
    "username": pd.StringDtype(),
    "enqueued_at": pd.Int64Dtype(),
    "lang": pd.StringDtype(),
    "retweet_count": pd.Int64Dtype(),
    "reply_count": pd.Int64Dtype(),
    "like_count": pd.Int64Dtype(),
    "quote_count": pd.Int64Dtype(),
    "bookmark_count": pd.Int64Dtype(),
    "impression_count": pd.Int64Dtype(),
    "author_followers_count": pd.Int64Dtype(),
    "author_following_count": pd.Int64Dtype(),
    "author_tweet_count": pd.Int64Dtype(),
    "author_listed_count": pd.Int64Dtype(),
    "author_like_count": pd.Int64Dtype(),
    "author_media_count": pd.Int64Dtype(),
    "author_verified_type": pd.StringDtype(),
    "author_parody": pd.BooleanDtype(),
    "current_status": pd.StringDtype(),
    "first_status": pd.StringDtype(),
    "deleted_at_millis": pd.Int64Dtype(),
    "deletion_policy_names": pd.StringDtype(),
    "deletion_model_score": pd.Float64Dtype(),
    "max_deletion_model_score": pd.Float64Dtype(),
    "hist_note_count": pd.Int64Dtype(),
    "hist_crh_count": pd.Int64Dtype(),
    "hist_crnh_count": pd.Int64Dtype(),
    "hist_total_ratings": pd.Int64Dtype(),
    "notable_post_prediction": pd.Float64Dtype(),
    "api_feed": pd.StringDtype(),
    "timed_feed": pd.StringDtype(),
    "num_unique_sources": pd.Int64Dtype(),
    "total_source_suggestions": pd.Int64Dtype(),
    "num_note_request_suggestions": pd.Int64Dtype(),
    "has_photo": pd.BooleanDtype(),
    "has_video": pd.BooleanDtype(),
}

for _col in _RATINGS_COLUMNS:
    _DTYPES[_col] = pd.Int64Dtype()


_ZERO_RATINGS: dict[str, int] = {}
for _bucket in _BUCKETS:
    _ZERO_RATINGS[f"{_bucket}_helpful"] = 0
    _ZERO_RATINGS[f"{_bucket}_not_helpful"] = 0
    _ZERO_RATINGS[f"{_bucket}_somewhat_helpful"] = 0
    for _tag in _HELPFUL_TAGS:
        _ZERO_RATINGS[f"{_bucket}_{_tag}"] = 0
    for _tag in _NOT_HELPFUL_TAGS:
        _ZERO_RATINGS[f"{_bucket}_{_tag}"] = 0


def _flatten_ratings(ratings: NoteRatings | None) -> dict:
    if ratings is None:
        return dict(_ZERO_RATINGS)
    result: dict = {}
    for bucket_name, bucket_data in [
        ("negative", ratings.negative),
        ("neutral", ratings.neutral),
        ("positive", ratings.positive),
    ]:
        result[f"{bucket_name}_helpful"] = bucket_data.helpful_count
        result[f"{bucket_name}_not_helpful"] = bucket_data.not_helpful_count
        result[f"{bucket_name}_somewhat_helpful"] = bucket_data.somewhat_helpful_count

        htag_lookup = {
            _HELPFUL_TAG_NAME_MAP.get(tc.tag_name, tc.tag_name): tc.tag_count
            for tc in bucket_data.helpful_tag_counts
        }
        for tag in _HELPFUL_TAGS:
            result[f"{bucket_name}_{tag}"] = htag_lookup.get(tag, 0)

        nhtag_lookup = {
            _NOT_HELPFUL_TAG_NAME_MAP.get(tc.tag_name, tc.tag_name): tc.tag_count
            for tc in bucket_data.not_helpful_tag_counts
        }
        for tag in _NOT_HELPFUL_TAGS:
            result[f"{bucket_name}_{tag}"] = nhtag_lookup.get(tag, 0)
    return result


class PriorNote(BaseModel):
    note_id: int
    post_id: int
    note_text: str
    post_text: str | None = None
    status: str


class PriorExamples(BaseModel):
    crh_notes: list[PriorNote]
    crnh_notes: list[PriorNote]
    n_deleted_as_crnh: int = 0


class PriorPostNote(BaseModel):
    note_id: int
    writer_name: str | None = None
    note_text: str
    is_deleted: bool
    current_status: str | None = None


def _find_latest_submission_history(note_submission_dir: str | None) -> str | None:
    if not note_submission_dir or not os.path.isdir(note_submission_dir):
        return None
    files = glob.glob(
        os.path.join(note_submission_dir, "note_submission_history_*.parquet")
    )
    if not files:
        return None
    return max(
        files,
        key=lambda f: int(
            os.path.basename(f)
            .replace("note_submission_history_", "")
            .replace(".parquet", "")
        ),
    )


def _load_saved_submission_history(
    note_submission_dir: str | None,
) -> pd.DataFrame | None:
    filepath = _find_latest_submission_history(note_submission_dir)
    if filepath is None:
        return None
    try:
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        logger.exception(f"Could not load submission history from {filepath}: {e}")
        return None


def fetch_api_statuses(
    oauth_sessions: dict[str, OAuth1Session],
    min_created_at_ms: int | None = None,
) -> dict[int, NoteStatus]:
    statuses: dict[int, NoteStatus] = {}
    for account_name, oauth in oauth_sessions.items():
        try:
            notes = get_notes_written(
                oauth=oauth,
                min_created_at_ms=min_created_at_ms,
            )
            for note in notes:
                note.status = note.status.upper()
                note.submitter = account_name
                statuses[note.note_id] = note
        except Exception as e:
            logger.exception(
                f"Error fetching notes for {account_name}: {type(e).__name__}: {e}"
            )
    return statuses


class SubmittedNoteCache:
    def __init__(self) -> None:
        self._df = pd.DataFrame(
            {col: pd.array([], dtype=_DTYPES[col]) for col in _COLUMNS}
        )
        self._lock = asyncio.Lock()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def initialize(self, note_submission_dir: str | None) -> None:
        saved_df = _load_saved_submission_history(note_submission_dir)
        if saved_df is None:
            logger.info("No saved submission history found, starting with empty cache")
            return

        df = saved_df.copy()

        if (
            "engagement_prediction" in df.columns
            and "notable_post_prediction" not in df.columns
        ):
            df = df.rename(columns={"engagement_prediction": "notable_post_prediction"})

        legacy_cols = [col for col in df.columns if col not in _COLUMNS]
        if legacy_cols:
            logger.info(
                f"Dropping {len(legacy_cols)} column(s) not in the current schema: {legacy_cols}"
            )
            df = df.drop(columns=legacy_cols)

        missing_cols = [col for col in _COLUMNS if col not in df.columns]
        if missing_cols:
            logger.info(
                f"Warning — filling {len(missing_cols)} missing column(s) with NA: {missing_cols}"
            )
            for col in missing_cols:
                df[col] = pd.NA

        bad_ts = df["created_at_millis"].isna() & df["note_id"].notna()
        assert not bad_ts.any(), (
            f"{bad_ts.sum()} note(s) have missing created_at_millis"
        )

        dupes = df["note_id"].dropna()
        dupes = dupes[dupes.duplicated(keep=False)]
        assert dupes.empty, (
            f"{len(dupes)} rows have duplicate note_id values: "
            f"{sorted(dupes.unique().tolist()[:10])}"
        )

        df = df[_COLUMNS].astype(_DTYPES)
        df = df.sort_values("note_id").reset_index(drop=True)
        self._df = df

    def add_submitted_note(
        self,
        note_id: int,
        post_id: int,
        writer_name: str,
        submitter: str,
        note_text: str,
        post_text: str | None,
        author_id: int | None = None,
        username: str | None = None,
        enqueued_at: int | None = None,
        lang: str | None = None,
        retweet_count: int | None = None,
        reply_count: int | None = None,
        like_count: int | None = None,
        quote_count: int | None = None,
        bookmark_count: int | None = None,
        impression_count: int | None = None,
        author_followers_count: int | None = None,
        author_following_count: int | None = None,
        author_tweet_count: int | None = None,
        author_listed_count: int | None = None,
        author_like_count: int | None = None,
        author_media_count: int | None = None,
        author_verified_type: str | None = None,
        author_parody: bool | None = None,
        hist_note_count: int | None = None,
        hist_crh_count: int | None = None,
        hist_crnh_count: int | None = None,
        hist_total_ratings: int | None = None,
        notable_post_prediction: float | None = None,
        api_feed: str | None = None,
        timed_feed: str | None = None,
        num_unique_sources: int | None = None,
        total_source_suggestions: int | None = None,
        num_note_request_suggestions: int | None = None,
        has_photo: bool | None = None,
        has_video: bool | None = None,
    ) -> None:
        if (self._df["note_id"] == note_id).any():
            return
        row_data: dict = {col: pd.NA for col in _COLUMNS}
        row_data.update(
            {
                "note_id": note_id,
                "post_id": post_id,
                "writer_name": writer_name,
                "submitter": submitter,
                "created_at_millis": _get_timestamp_from_snowflake(note_id)
                if note_id > 0
                else pd.NA,
                "current_status": "NEEDS_MORE_RATINGS",
                "note_text": note_text,
                "post_text": post_text,
                "author_id": author_id,
                "username": username,
                "enqueued_at": enqueued_at,
                "lang": lang,
                "retweet_count": retweet_count,
                "reply_count": reply_count,
                "like_count": like_count,
                "quote_count": quote_count,
                "bookmark_count": bookmark_count,
                "impression_count": impression_count,
                "author_followers_count": author_followers_count,
                "author_following_count": author_following_count,
                "author_tweet_count": author_tweet_count,
                "author_listed_count": author_listed_count,
                "author_like_count": author_like_count,
                "author_media_count": author_media_count,
                "author_verified_type": author_verified_type,
                "author_parody": author_parody,
                "hist_note_count": hist_note_count,
                "hist_crh_count": hist_crh_count,
                "hist_crnh_count": hist_crnh_count,
                "hist_total_ratings": hist_total_ratings,
                "notable_post_prediction": notable_post_prediction,
                "api_feed": api_feed,
                "timed_feed": timed_feed,
                "num_unique_sources": num_unique_sources,
                "total_source_suggestions": total_source_suggestions,
                "num_note_request_suggestions": num_note_request_suggestions,
                "has_photo": has_photo,
                "has_video": has_video,
            }
        )
        new_row = pd.DataFrame([row_data]).astype(_DTYPES)
        self._df = pd.concat([self._df, new_row], ignore_index=True)

    async def get_prior_notes_for_post(self, post_id: int) -> list[PriorPostNote]:
        async with self._lock:
            post_notes = self._df[
                (self._df["post_id"] == post_id) & self._df["note_text"].notna()
            ]
            return [
                PriorPostNote(
                    note_id=int(row["note_id"]),
                    writer_name=str(row["writer_name"])
                    if pd.notna(row["writer_name"])
                    else None,
                    note_text=str(row["note_text"]),
                    is_deleted=pd.notna(row["deleted_at_millis"]),
                    current_status=str(row["current_status"])
                    if pd.notna(row["current_status"])
                    else None,
                )
                for _, row in post_notes.iterrows()
            ]

    async def update_statuses(self, api_statuses: dict[int, NoteStatus]) -> None:
        if not api_statuses:
            return
        async with self._lock:
            t0 = time.time()
            now_ms = int(t0 * 1000)

            ratings_count_cols = [
                c for c in _RATINGS_COLUMNS if c != "ratings_updated_at_millis"
            ]
            update_records = []
            for note_id, ns in api_statuses.items():
                row: dict = {"note_id": note_id, "current_status": ns.status}
                row.update(_flatten_ratings(ns.ratings))
                row["ratings_updated_at_millis"] = now_ms
                update_records.append(row)
            updates_df = pd.DataFrame(update_records)

            api_ids = set(api_statuses.keys())
            in_cache_mask = self._df["note_id"].isin(api_ids)
            rows_to_update = self._df.loc[in_cache_mask].copy()
            rows_unchanged = self._df.loc[~in_cache_mask]

            matched_ids = set(rows_to_update["note_id"].dropna().astype(int))
            missing_ids = api_ids - matched_ids
            if missing_ids:
                backfill_rows = []
                for nid in sorted(missing_ids):
                    ns = api_statuses[nid]
                    logger.info(
                        f"Backfilling missing note note_id={nid} post_id={ns.post_id}"
                    )
                    row_data: dict = {col: pd.NA for col in _COLUMNS}
                    row_data.update(
                        {
                            "note_id": nid,
                            "post_id": ns.post_id,
                            "submitter": ns.submitter,
                            "created_at_millis": _get_timestamp_from_snowflake(nid),
                            "note_text": ns.note_text,
                        }
                    )
                    backfill_rows.append(row_data)
                backfill_df = pd.DataFrame(backfill_rows).astype(_DTYPES)
                rows_to_update = pd.concat(
                    [rows_to_update, backfill_df], ignore_index=True
                )

            api_cols = (
                ["current_status"] + ratings_count_cols + ["ratings_updated_at_millis"]
            )

            keep_cols = [
                c for c in _COLUMNS if c not in api_cols and c != "first_status"
            ]

            merged = rows_to_update[keep_cols].merge(
                updates_df[["note_id"] + api_cols],
                on="note_id",
                how="left",
            )

            old_first = rows_to_update.set_index("note_id")["first_status"]
            new_status = updates_df.set_index("note_id")["current_status"]

            first_status = old_first.copy()
            needs_first = first_status.isna()
            qualifies = new_status.reindex(first_status.index).isin([_CRH, _CRNH])
            first_status.loc[needs_first & qualifies] = new_status.reindex(
                first_status.index
            ).loc[needs_first & qualifies]
            merged["first_status"] = first_status.reindex(merged["note_id"]).values

            merged = merged[_COLUMNS]
            self._df = pd.concat([rows_unchanged, merged], ignore_index=True)
            self._df = self._df.sort_values("note_id").reset_index(drop=True)

            elapsed_ms = (time.time() - t0) * 1000
            if elapsed_ms > 500:
                logger.info(f"update_statuses took {elapsed_ms:.0f}ms")

    async def get_prior_examples(self, n_crh: int, n_crnh: int) -> PriorExamples:
        async with self._lock:
            crh_crnh_or_deleted = (
                self._df["first_status"].isin([_CRH, _CRNH])
                | self._df["deleted_at_millis"].notna()
            )
            unexpected_na = crh_crnh_or_deleted & (
                self._df["note_id"].isna()
                | self._df["post_id"].isna()
                | self._df["note_text"].isna()
            )
            if unexpected_na.sum() > 0:
                logger.info(
                    f"Warning — {unexpected_na.sum()} CRH/CRNH/deleted note(s) "
                    f"excluded from prior examples due to missing note_id, post_id, or note_text"
                )
            has_required = (
                self._df["note_id"].notna()
                & self._df["post_id"].notna()
                & self._df["note_text"].notna()
                & self._df["post_text"].notna()
            )
            eligible = self._df[has_required]

            crh_df = eligible[
                (eligible["first_status"] == _CRH)
                & eligible["deleted_at_millis"].isna()
            ].tail(n_crh)

            crnh_df = eligible[
                (eligible["first_status"] == _CRNH)
                | eligible["deleted_at_millis"].notna()
            ].tail(n_crnh)

            n_deleted_as_crnh = int(
                (
                    crnh_df["deleted_at_millis"].notna()
                    & (crnh_df["first_status"] != _CRNH)
                ).sum()
            )

            def _to_prior_notes(
                sub_df: pd.DataFrame, override_status: str | None = None
            ) -> list[PriorNote]:
                return [
                    PriorNote(
                        note_id=int(row["note_id"]),
                        post_id=int(row["post_id"]),
                        note_text=str(row["note_text"]),
                        post_text=str(row["post_text"]),
                        status=override_status or str(row["first_status"]),
                    )
                    for _, row in sub_df.iterrows()
                ]

            crh_notes = _to_prior_notes(crh_df)
            crnh_notes = _to_prior_notes(crnh_df, override_status=_CRNH)

            return PriorExamples(
                crh_notes=crh_notes,
                crnh_notes=crnh_notes,
                n_deleted_as_crnh=n_deleted_as_crnh,
            )

    def drop_old_rows(self, max_age_ms: int = _MAX_ROW_AGE_MS) -> int:
        cutoff_ms = int(time.time() * 1000) - max_age_ms
        old_mask = self._df["created_at_millis"].notna() & (
            self._df["created_at_millis"] < cutoff_ms
        )
        n_dropped = int(old_mask.sum())
        if n_dropped > 0:
            self._df = self._df[~old_mask].reset_index(drop=True)
            logger.info(
                f"Dropped {n_dropped} row(s) older than {max_age_ms // 86_400_000}d"
            )
        return n_dropped

    def get_save_data(self) -> pd.DataFrame:
        return self._df[_COLUMNS].copy()

    async def evaluate_deletion_policies(
        self,
        model_policies: list,
        model=None,
    ) -> list:
        async with self._lock:
            results, model_scores = get_notes_to_delete(
                self._df,
                model_policies,
                model,
            )
            self._update_max_deletion_scores(model_scores)
            return results

    def _update_max_deletion_scores(self, scores: dict[int, float]) -> None:
        if not scores:
            return
        score_ids = set(scores.keys())
        mask = self._df["note_id"].isin(score_ids)
        scored_indices = self._df.index[mask]
        if scored_indices.empty:
            return

        scored = self._df.loc[scored_indices]
        total = pd.Series(0, index=scored_indices)
        for bucket in _BUCKETS:
            for level in ("helpful", "not_helpful", "somewhat_helpful"):
                total = total + scored[f"{bucket}_{level}"].fillna(0)
        eligible_indices = scored_indices[total >= _MIN_RATINGS_FOR_MAX_SCORE]

        for idx in eligible_indices:
            nid = int(self._df.at[idx, "note_id"])
            new_score = scores[nid]
            existing = self._df.at[idx, "max_deletion_model_score"]
            if pd.isna(existing) or new_score > existing:
                self._df.at[idx, "max_deletion_model_score"] = new_score

    async def mark_deleted(
        self,
        note_id: int,
        deleted_at_millis: int,
        policy_names: list[str] | None = None,
        model_score: float | None = None,
    ) -> None:
        async with self._lock:
            mask = self._df["note_id"] == note_id
            if not mask.any():
                logger.error(
                    f"Error — mark_deleted called for note_id={note_id} which is not in the cache"
                )
                return
            idx = self._df.index[mask][0]
            self._df.at[idx, "deleted_at_millis"] = deleted_at_millis
            if policy_names is not None:
                self._df.at[idx, "deletion_policy_names"] = ",".join(policy_names)
            if model_score is not None:
                self._df.at[idx, "deletion_model_score"] = model_score

    async def size(self) -> tuple[int, int]:
        async with self._lock:
            return (
                int((self._df["first_status"] == _CRH).sum()),
                int((self._df["first_status"] == _CRNH).sum()),
            )
