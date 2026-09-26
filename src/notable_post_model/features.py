from __future__ import annotations

import numpy as np
import pandas as pd

from utils.snowflake import get_timestamp_from_snowflake


ENGAGEMENT_COLUMNS = [
    "retweet_count",
    "reply_count",
    "like_count",
    "quote_count",
    "bookmark_count",
    "impression_count",
]


TOP_LANGUAGES = ("en", "es", "ja", "pt", "tr")


HISTORY_COLUMNS = [
    "hist_note_count",
    "hist_crh_count",
    "hist_crnh_count",
    "hist_total_ratings",
]


NUMERIC_FEATURES = [
    "log_post_age_hours",
    "log_retweet_count",
    "log_reply_count",
    "log_like_count",
    "log_quote_count",
    "log_bookmark_count",
    "log_impression_count",
    "log_retweet_count_per_hour",
    "log_reply_count_per_hour",
    "log_like_count_per_hour",
    "log_quote_count_per_hour",
    "log_bookmark_count_per_hour",
    "log_impression_count_per_hour",
    "retweet_count_per_impression",
    "reply_count_per_impression",
    "like_count_per_impression",
    "quote_count_per_impression",
    "bookmark_count_per_impression",
    "log_author_followers",
    "log_author_following",
    "log_author_follow_ratio",
    "log_author_tweet_count",
    "log_author_listed_count",
    "log_author_like_count",
    "log_author_media_count",
    "log_num_unique_sources",
    "log_total_source_suggestions",
    "log_hist_note_count",
    "log_hist_total_ratings",
    "log_hist_mean_ratings",
    "hist_crh_rate",
    "hist_crnh_rate",
    "hist_has_history",
]

CATEGORICAL_FEATURES = [
    "author_verified_type",
    "lang_top",
    "post_tod_bucket",
    "enqueue_tod_bucket",
    "has_photo",
    "has_video",
    "author_parody",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


_RATING_GROUP_PREFIXES = ("negative_", "neutral_", "positive_")
_RATING_KIND_SUFFIXES = ("helpful", "not_helpful", "somewhat_helpful")


def _rating_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if not c.startswith(_RATING_GROUP_PREFIXES):
            continue
        suffix = c.split("_", 1)[1]
        if suffix in _RATING_KIND_SUFFIXES:
            out.append(c)
    return out


def _safe_log10(x: pd.Series, offset: float = 1.0) -> pd.Series:
    arr = x.astype("float64").fillna(0).clip(lower=0)
    return np.log10(arr + offset)


def compute_author_history_from_cache(
    cache_df: pd.DataFrame,
    author_ids: pd.Series | None = None,
) -> pd.DataFrame:
    df = cache_df[cache_df["note_id"].notna() & cache_df["author_id"].notna()].copy()

    _CRH = "CURRENTLY_RATED_HELPFUL"
    _CRNH = "CURRENTLY_RATED_NOT_HELPFUL"
    is_cur_crh = (df["current_status"] == _CRH).fillna(False)
    is_cur_crnh = (df["current_status"] == _CRNH).fillna(False)
    was_crh = (df["first_status"] == _CRH).fillna(False)
    was_crnh = (df["first_status"] == _CRNH).fillna(False)
    was_deleted = df["deleted_at_millis"].notna()

    df["_is_crh"] = (is_cur_crh | was_crh).astype("int64")
    df["_is_crnh"] = (is_cur_crnh | was_crnh | was_deleted).astype("int64")

    rating_cols = _rating_columns(df)
    if rating_cols:
        df["_total_ratings"] = df[rating_cols].fillna(0).astype("int64").sum(axis=1)
    else:
        df["_total_ratings"] = 0

    g = df.groupby("author_id", sort=False)
    result = pd.DataFrame(
        {
            "hist_note_count": g.size(),
            "hist_crh_count": g["_is_crh"].sum(),
            "hist_crnh_count": g["_is_crnh"].sum(),
            "hist_total_ratings": g["_total_ratings"].sum(),
        }
    ).astype("int64")

    if author_ids is not None:
        unique_ids = author_ids.drop_duplicates()
        result = result.reindex(unique_ids, fill_value=0)

    result = result.reset_index().rename(columns={"index": "author_id"})
    return result


def compute_model_features(
    df: pd.DataFrame,
    top_languages: tuple[str, ...] = TOP_LANGUAGES,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    post_created_at_ms = (
        df["post_id"].astype("int64").apply(get_timestamp_from_snowflake)
    )
    enqueued_ms = df["enqueued_at"].astype("int64")
    post_age_hours = (enqueued_ms - post_created_at_ms) / (1000 * 3600)
    post_age_hours_clipped = post_age_hours.clip(lower=0.01)
    out["log_post_age_hours"] = np.log10(post_age_hours_clipped)

    post_created_at = pd.to_datetime(post_created_at_ms, unit="ms")
    out["post_tod_bucket"] = (post_created_at.dt.hour // 3).astype("int8")
    enqueued_dt = pd.to_datetime(enqueued_ms, unit="ms")
    out["enqueue_tod_bucket"] = (enqueued_dt.dt.hour // 3).astype("int8")

    for col in ENGAGEMENT_COLUMNS:
        out[f"log_{col}"] = _safe_log10(df[col])

    for col in ENGAGEMENT_COLUMNS:
        rate = (
            df[col].astype("float64").fillna(0).clip(lower=0) / post_age_hours_clipped
        )
        out[f"log_{col}_per_hour"] = np.log10(rate + 1.0)

    imp = df["impression_count"].astype("float64").fillna(0)
    for col in [
        "retweet_count",
        "reply_count",
        "like_count",
        "quote_count",
        "bookmark_count",
    ]:
        num = df[col].astype("float64").fillna(0)
        ratio = num / (imp + 1.0)
        out[f"{col}_per_impression"] = ratio.clip(lower=0.0, upper=1.0)

    out["log_author_followers"] = _safe_log10(df["author_followers_count"])
    out["log_author_following"] = _safe_log10(df["author_following_count"])
    follow_ratio = df["author_followers_count"].astype("float64") / (
        df["author_following_count"].astype("float64") + 1.0
    )
    out["log_author_follow_ratio"] = np.log10(
        follow_ratio.fillna(0).clip(lower=0) + 0.01
    )
    out["log_author_tweet_count"] = _safe_log10(df["author_tweet_count"])
    out["log_author_listed_count"] = _safe_log10(df["author_listed_count"])
    out["log_author_like_count"] = _safe_log10(df["author_like_count"])
    out["log_author_media_count"] = _safe_log10(df["author_media_count"])
    out["author_parody"] = df["author_parody"].astype("int8")

    out["log_num_unique_sources"] = _safe_log10(df["num_unique_sources"])
    out["log_total_source_suggestions"] = _safe_log10(df["total_source_suggestions"])

    out["has_photo"] = df["has_photo"].astype("int8")
    out["has_video"] = df["has_video"].astype("int8")

    hist_note_count = df["hist_note_count"].astype("float64").fillna(0)
    hist_crh_count = df["hist_crh_count"].astype("float64").fillna(0)
    hist_crnh_count = df["hist_crnh_count"].astype("float64").fillna(0)
    hist_total_ratings = df["hist_total_ratings"].astype("float64").fillna(0)
    hist_mean_ratings = hist_total_ratings / hist_note_count.clip(lower=1)

    out["hist_has_history"] = (hist_note_count > 0).astype("int8")
    out["hist_crh_rate"] = hist_crh_count / hist_note_count.clip(lower=1)
    out["hist_crnh_rate"] = hist_crnh_count / hist_note_count.clip(lower=1)
    out["log_hist_note_count"] = np.log10(hist_note_count + 1)
    out["log_hist_total_ratings"] = np.log10(hist_total_ratings + 1)
    out["log_hist_mean_ratings"] = np.log10(hist_mean_ratings + 1)

    top = set(top_languages)
    out["lang_top"] = df["lang"].where(df["lang"].isin(top), other="other")

    out["author_verified_type"] = df["author_verified_type"]

    return out[ALL_FEATURES]
