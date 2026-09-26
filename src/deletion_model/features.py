from __future__ import annotations

import numpy as np
import pandas as pd


NEGATIVE_THRESHOLD = -0.15
POSITIVE_THRESHOLD = 0.15

BUCKETS = ["negative", "neutral", "positive"]

HELPFULNESS_LEVELS = ["helpful", "not_helpful", "somewhat_helpful"]


TAG_COLUMN_MAP = {
    "helpfulImportantContext": "important_context",
    "helpfulAddressesClaim": "addresses_claim",
    "helpfulGoodSources": "good_sources",
    "helpfulUnbiasedLanguage": "unbiased_language",
    "helpfulClear": "clear",
    "helpfulOther": "helpful_other",
    "notHelpfulSourcesMissingOrUnreliable": "no_sources",
    "notHelpfulMissingKeyPoints": "missing_key_points",
    "notHelpfulOpinionSpeculation": "opinion_speculation",
    "notHelpfulArgumentativeOrBiased": "rude",
    "notHelpfulOther": "not_helpful_other",
    "notHelpfulIncorrect": "incorrect",
    "notHelpfulNoteNotNeeded": "note_not_needed",
    "notHelpfulHardToUnderstand": "unclear",
    "notHelpfulSpamHarassmentOrAbuse": "twitter_violation_any",
    "notHelpfulIrrelevantSources": "irrelevant_sources",
}

ALL_TAGS = [
    "important_context",
    "addresses_claim",
    "good_sources",
    "unbiased_language",
    "clear",
    "helpful_other",
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

HELPFULNESS_MAP = {
    "HELPFUL": "helpful",
    "NOT_HELPFUL": "not_helpful",
    "SOMEWHAT_HELPFUL": "somewhat_helpful",
}

SNAPSHOT_THRESHOLDS = [3, 4, 5, 6, 8, 10, 12, 15, 20, 30]


API_WRITER_ENROLLMENT_STATE = "apiEarnedIn"


PANELS: list[tuple[str, str, list[int]]] = [
    ("3to4", "ratings 3-4", [3, 4]),
    ("5to10", "ratings 5-10", [5, 6, 8, 10]),
    ("11to30", "ratings 11-30", [12, 15, 20, 30]),
]


RATING_COLUMNS = [
    "negative_helpful",
    "negative_not_helpful",
    "negative_somewhat_helpful",
    "negative_important_context",
    "negative_addresses_claim",
    "negative_good_sources",
    "negative_unbiased_language",
    "negative_clear",
    "negative_helpful_other",
    "negative_no_sources",
    "negative_missing_key_points",
    "negative_opinion_speculation",
    "negative_rude",
    "negative_not_helpful_other",
    "negative_incorrect",
    "negative_note_not_needed",
    "negative_unclear",
    "negative_twitter_violation_any",
    "negative_irrelevant_sources",
    "neutral_helpful",
    "neutral_not_helpful",
    "neutral_somewhat_helpful",
    "neutral_important_context",
    "neutral_addresses_claim",
    "neutral_good_sources",
    "neutral_unbiased_language",
    "neutral_clear",
    "neutral_helpful_other",
    "neutral_no_sources",
    "neutral_missing_key_points",
    "neutral_opinion_speculation",
    "neutral_rude",
    "neutral_not_helpful_other",
    "neutral_incorrect",
    "neutral_note_not_needed",
    "neutral_unclear",
    "neutral_twitter_violation_any",
    "neutral_irrelevant_sources",
    "positive_helpful",
    "positive_not_helpful",
    "positive_somewhat_helpful",
    "positive_important_context",
    "positive_addresses_claim",
    "positive_good_sources",
    "positive_unbiased_language",
    "positive_clear",
    "positive_helpful_other",
    "positive_no_sources",
    "positive_missing_key_points",
    "positive_opinion_speculation",
    "positive_rude",
    "positive_not_helpful_other",
    "positive_incorrect",
    "positive_note_not_needed",
    "positive_unclear",
    "positive_twitter_violation_any",
    "positive_irrelevant_sources",
]

DERIVED_FEATURES = [
    "negative_total",
    "neutral_total",
    "positive_total",
    "total_ratings",
    "negative_fraction",
    "neutral_fraction",
    "positive_fraction",
    "negative_helpful_rate",
    "neutral_helpful_rate",
    "positive_helpful_rate",
    "negative_not_helpful_rate",
    "neutral_not_helpful_rate",
    "positive_not_helpful_rate",
    "overall_helpful_rate",
    "overall_not_helpful_rate",
]

NUMERIC_FEATURES = RATING_COLUMNS + DERIVED_FEATURES


INTERACTION_FEATURES = [
    "negative_total",
    "neutral_total",
    "positive_total",
    "total_ratings",
    "negative_fraction",
    "neutral_fraction",
    "positive_fraction",
    "negative_helpful_rate",
    "neutral_helpful_rate",
    "positive_helpful_rate",
    "negative_not_helpful_rate",
    "neutral_not_helpful_rate",
    "positive_not_helpful_rate",
    "overall_helpful_rate",
    "overall_not_helpful_rate",
    "negative_helpful",
    "negative_not_helpful",
    "negative_somewhat_helpful",
    "neutral_helpful",
    "neutral_not_helpful",
    "neutral_somewhat_helpful",
    "positive_helpful",
    "positive_not_helpful",
    "positive_somewhat_helpful",
]


CROSS_TAG_COLUMNS = [f"{bucket}_{tag}" for bucket in BUCKETS for tag in ALL_TAGS]

CROSS_INPUT_COLUMNS = CROSS_TAG_COLUMNS + [
    "negative_total",
    "neutral_total",
    "positive_total",
]


N_TAGS = len(ALL_TAGS)


TAG_BUCKET_PAIRS = []
for i in range(N_TAGS):
    TAG_BUCKET_PAIRS.append((i, 48))
    TAG_BUCKET_PAIRS.append((N_TAGS + i, 49))
    TAG_BUCKET_PAIRS.append((2 * N_TAGS + i, 50))


CROSS_BUCKET_PAIRS = []
for i in range(N_TAGS):
    CROSS_BUCKET_PAIRS.append((i, N_TAGS + i))
    CROSS_BUCKET_PAIRS.append((i, 2 * N_TAGS + i))
    CROSS_BUCKET_PAIRS.append((N_TAGS + i, 2 * N_TAGS + i))

ALL_CROSS_PAIRS = TAG_BUCKET_PAIRS + CROSS_BUCKET_PAIRS


def load_api_writer_ids(enrollment_path: str) -> set[str]:
    enrollment = pd.read_csv(
        enrollment_path,
        sep="\t",
        header=None,
        usecols=[0, 1],
        names=["participant_id", "enrollment_state"],
        dtype=str,
    )
    mask = enrollment["enrollment_state"] == API_WRITER_ENROLLMENT_STATE
    return set(enrollment.loc[mask, "participant_id"])


def assign_bucket(factor: float) -> str:
    if factor < NEGATIVE_THRESHOLD:
        return "negative"
    elif factor > POSITIVE_THRESHOLD:
        return "positive"
    else:
        return "neutral"


def compute_model_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df[RATING_COLUMNS].copy()

    for bucket in BUCKETS:
        bucket_cols = [f"{bucket}_{level}" for level in HELPFULNESS_LEVELS]
        features[f"{bucket}_total"] = features[bucket_cols].sum(axis=1)

    total_cols = [f"{bucket}_total" for bucket in BUCKETS]
    features["total_ratings"] = features[total_cols].sum(axis=1)

    for bucket in BUCKETS:
        total_denom = features["total_ratings"].clip(lower=1)
        features[f"{bucket}_fraction"] = features[f"{bucket}_total"] / total_denom

        bucket_denom = features[f"{bucket}_total"].clip(lower=1)
        features[f"{bucket}_helpful_rate"] = (
            features[f"{bucket}_helpful"] / bucket_denom
        )
        features[f"{bucket}_not_helpful_rate"] = (
            features[f"{bucket}_not_helpful"] / bucket_denom
        )

    overall_helpful = sum(features[f"{b}_helpful"] for b in BUCKETS)
    overall_not_helpful = sum(features[f"{b}_not_helpful"] for b in BUCKETS)
    denom = features["total_ratings"].clip(lower=1)
    features["overall_helpful_rate"] = overall_helpful / denom
    features["overall_not_helpful_rate"] = overall_not_helpful / denom

    return features[NUMERIC_FEATURES]
