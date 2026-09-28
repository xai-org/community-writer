import pandas as pd

from data_models.arena_config import ArenaConfig, FeedDefinition, GrokWriter
from note_writer.note_length import is_over_length


def find_best_draft(
    writer: GrokWriter,
    writing_results_df: pd.DataFrame,
) -> tuple[pd.Series, int] | None:
    writer_drafts = writing_results_df[
        (writing_results_df["writer_name"] == writer.writer_name)
        & writing_results_df["grok_note"].notna()
    ]
    if writer_drafts.empty:
        return None
    within_limit = ~writer_drafts["grok_note"].map(is_over_length).astype(bool)
    writer_drafts = writer_drafts.assign(_within_limit=within_limit).sort_values(
        by=["_within_limit", "co_score"], ascending=False
    )
    return writer_drafts.iloc[0].drop("_within_limit"), writer_drafts.index[0]


def is_draft_on_track_for_submission(
    best_writing_result: pd.Series,
    feed_def: FeedDefinition,
    post_id: int,
    arena_config: ArenaConfig,
) -> bool:
    return (
        not pd.isna(best_writing_result.grok_note)
        and not pd.isna(best_writing_result.co_score)
        and best_writing_result.co_score >= feed_def.co_threshold
        and (
            not feed_def.rl_rejector
            or (
                not pd.isna(best_writing_result.get("rejection_status"))
                and best_writing_result.rejection_status == "PASS"
            )
        )
        and (
            not feed_def.recent_context_rejector
            or (
                not pd.isna(best_writing_result.get("recent_context_rejection_status"))
                and best_writing_result.recent_context_rejection_status == "PASS"
            )
        )
        and (
            arena_config.get_multi_note_policy(post_id) == "once_per_writer"
            or pd.isna(best_writing_result.get("revision_rejection_status"))
            or best_writing_result.revision_rejection_status == "PASS"
        )
    )
