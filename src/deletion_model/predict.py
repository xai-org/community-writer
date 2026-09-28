from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from data_models.arena_config import ModelDeletionPolicy
from data_models.deleted_note import DeletedNote
from deletion_model.features import RATING_COLUMNS, compute_model_features

from utils.log_setup import get_logger

logger = get_logger("deletion")

_BUCKETS = ["negative", "neutral", "positive"]
_HELPFULNESS_LEVELS = ["helpful", "not_helpful", "somewhat_helpful"]


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)


def _eligible_mask(df: pd.DataFrame, max_age_days: int) -> pd.Series:
    cutoff_ms = int(time.time() * 1000) - max_age_days * 86_400_000
    return (
        df["created_at_millis"].notna()
        & (df["created_at_millis"] >= cutoff_ms)
        & df["deleted_at_millis"].isna()
        & df["ratings_updated_at_millis"].notna()
    )


def _compute_total_ratings(eligible: pd.DataFrame) -> pd.Series:
    total = pd.Series(0, index=eligible.index)
    for bucket in _BUCKETS:
        for level in _HELPFULNESS_LEVELS:
            total = total + eligible[f"{bucket}_{level}"].fillna(0)
    return total


def _build_nonzero_rating_counts(row: pd.Series) -> dict[str, int]:
    result = {}
    for col in RATING_COLUMNS:
        val = row.get(col, 0)
        if pd.notna(val) and int(val) > 0:
            result[col] = int(val)
    return result


def _evaluate_model_policies(
    df: pd.DataFrame,
    policies: list[ModelDeletionPolicy],
    model: Pipeline,
) -> tuple[dict[int, list[str]], dict[int, float]]:
    matched: dict[int, list[str]] = {}
    scores: dict[int, float] = {}

    all_eligible_mask = pd.Series(False, index=df.index)
    for policy in policies:
        all_eligible_mask = all_eligible_mask | _eligible_mask(df, policy.max_age_days)
    all_eligible = df[all_eligible_mask]
    if all_eligible.empty:
        return matched, scores

    model_input = all_eligible[RATING_COLUMNS].fillna(0)
    features = compute_model_features(model_input)
    raw_scores = model.predict_proba(features)[:, 1]

    score_series = pd.Series(raw_scores, index=all_eligible.index)
    for idx, score in score_series.items():
        nid = int(all_eligible.at[idx, "note_id"])
        scores[nid] = float(score)

    total_ratings = _compute_total_ratings(all_eligible)

    for policy in policies:
        eligible_mask = _eligible_mask(df, policy.max_age_days)
        eligible = df[eligible_mask]
        n_eligible = len(eligible)
        if n_eligible == 0:
            continue

        policy_eligible_idx = eligible.index
        policy_scores = score_series.reindex(policy_eligible_idx)
        policy_totals = total_ratings.reindex(policy_eligible_idx)

        candidates_mask = (policy_totals >= policy.min_total_ratings) & (
            policy_scores >= policy.score_threshold
        )

        candidates_mask = candidates_mask.fillna(False)
        candidates = eligible[candidates_mask]

        n_candidates = len(candidates)
        if n_candidates == 0:
            continue
        fraction = n_candidates / n_eligible
        if fraction > policy.max_deletion_fraction:
            logger.error(
                f"Safeguard tripped for model policy '{policy.policy_name}' — "
                f"{n_candidates}/{n_eligible} ({fraction:.1%}) exceeds "
                f"max_deletion_fraction={policy.max_deletion_fraction:.1%}. Skipping."
            )
            continue

        for idx in candidates.index:
            nid = int(candidates.at[idx, "note_id"])
            matched.setdefault(nid, []).append(policy.policy_name)

    return matched, scores


def get_notes_to_delete(
    cache_df: pd.DataFrame,
    model_policies: list[ModelDeletionPolicy],
    model: Pipeline | None = None,
) -> tuple[list[DeletedNote], dict[int, float]]:
    model_matched: dict[int, list[str]] = {}
    model_scores: dict[int, float] = {}
    if model_policies and model is not None:
        model_matched, model_scores = _evaluate_model_policies(
            cache_df,
            model_policies,
            model,
        )

    all_note_ids = set(model_matched.keys())
    if not all_note_ids:
        return [], model_scores

    eligible_df = cache_df[cache_df["note_id"].isin(all_note_ids)]
    total_ratings_map: dict[int, int] = {}
    totals = _compute_total_ratings(eligible_df)
    for idx in eligible_df.index:
        nid = int(eligible_df.at[idx, "note_id"])
        total_ratings_map[nid] = int(totals.at[idx])

    results = []
    for idx in eligible_df.index:
        nid = int(eligible_df.at[idx, "note_id"])
        if nid not in all_note_ids:
            continue
        policy_names = model_matched.get(nid, [])
        submitter = (
            str(eligible_df.at[idx, "submitter"])
            if pd.notna(eligible_df.at[idx, "submitter"])
            else ""
        )
        results.append(
            DeletedNote(
                note_id=nid,
                submitter=submitter,
                model_score=model_scores.get(nid),
                policy_names=policy_names,
                total_ratings=total_ratings_map[nid],
                nonzero_rating_counts=_build_nonzero_rating_counts(
                    eligible_df.loc[idx]
                ),
            )
        )

    return results, model_scores
