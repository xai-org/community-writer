import random
import time
from datetime import datetime

import pandas as pd
from requests_oauthlib import OAuth1Session  # type: ignore

from cnapi.submit_note import submit_note, NoteSubmissionError
from data_models.writer_data_models import PostWithContext
from data_models.submitted_note_cache import SubmittedNoteCache
from note_writer.write_note import _check_for_unsupported_media_in_post_with_context
from rejectors import find_best_draft, is_draft_on_track_for_submission
from data_models.arena_config import (
    ALLOCATION_END,
    SUBMISSION_PURPOSE,
    GrokWriter,
    ArenaConfig,
    SubmissionConfig,
    is_writer_active_for_work_item,
    post_digest,
)

from utils.log_setup import get_logger

logger = get_logger("submit_note")


def _get_eligible_submitters(
    submission_configs: list[SubmissionConfig],
    writer: GrokWriter,
    post_id: int,
    note_records: pd.DataFrame,
) -> list[str]:
    eligible_submitters = []
    for config in submission_configs:
        past_submitters = note_records[note_records["post_id"] == post_id][
            "submitter"
        ].values
        if config.account_name in past_submitters:
            continue

        cutoff = 1000 * (time.time() - (60 * 60 * 24))
        recent_submissions = note_records[
            (note_records["submitter"] == config.account_name)
            & (note_records["created_at_millis"] > cutoff)
        ]
        if len(recent_submissions) >= config.daily_limit:
            continue
        eligible_submitters.append(config.account_name)

    last_submission_for_writer = note_records[
        note_records["writer_name"] == writer.writer_name
    ]["note_id"].max()
    last_submitter = note_records[
        note_records["note_id"] == last_submission_for_writer
    ]["submitter"].values
    assert 0 <= len(last_submitter) <= 1, (
        "last_submitter should be a single value or empty"
    )

    if (
        pd.isna(last_submission_for_writer)
        or last_submitter[0] not in eligible_submitters
    ):
        random.shuffle(eligible_submitters)
    else:
        index = (eligible_submitters.index(last_submitter[0]) + 1) % len(
            eligible_submitters
        )
        return eligible_submitters[index:] + eligible_submitters[:index]
    return eligible_submitters


def submit_note_for_post(
    writer_name: str,
    arena_config: ArenaConfig,
    post_with_context: PostWithContext,
    writing_results_df: pd.DataFrame,
    work_started_at: datetime,
    submitted_note_cache: SubmittedNoteCache,
    oauth_sessions: dict[str, OAuth1Session],
    dry_run: bool,
) -> None:
    writer = next(
        (w for w in arena_config.grok_writers if w.writer_name == writer_name), None
    )
    if writer is None:
        return
    post_policy = arena_config.get_multi_note_policy(post_with_context.post.post_id)

    skip_feed_check = post_policy == "once_per_writer"
    if not is_writer_active_for_work_item(
        writer, post_with_context, skip_feed_check=skip_feed_check
    ):
        return

    effective_dry_run = dry_run or writer.dry_run

    feed_def = arena_config.get_api_feed_def(post_with_context.api_feed)
    digest = post_digest(
        post_with_context.post.post_id, SUBMISSION_PURPOSE, ALLOCATION_END
    )

    writer_drafts = writing_results_df[
        writing_results_df["writer_name"] == writer.writer_name
    ]

    if effective_dry_run:
        for index in writer_drafts.index:
            writing_results_df.at[index, "dry_run"] = "other"
    post_in_enabled_range = False
    for start, end in feed_def.submission_ranges:
        if digest >= start and digest < end:
            post_in_enabled_range = True
            break
    post_media_compatible = (
        len(
            _check_for_unsupported_media_in_post_with_context(
                post_with_context, writer.allowed_media_types
            )
        )
        == 0
    )
    post_age_seconds = (
        work_started_at - post_with_context.post.created_at
    ).total_seconds()
    post_recent_enough = post_age_seconds <= feed_def.max_post_age_seconds
    post_eligible = (
        post_in_enabled_range and post_media_compatible and post_recent_enough
    )
    for index in writer_drafts.index:
        writing_results_df.at[index, "submission_eligible"] = post_eligible

    best_draft = find_best_draft(writer, writing_results_df)
    if best_draft is not None:
        best_writing_result, best_row_idx = best_draft
    else:
        fallback = writer_drafts.sort_values(by="co_score", ascending=False)
        best_writing_result = fallback.iloc[0]
        best_row_idx = fallback.index[0]

    if not post_eligible or not is_draft_on_track_for_submission(
        best_writing_result, feed_def, post_with_context.post.post_id, arena_config
    ):
        if effective_dry_run:
            writing_results_df.at[best_row_idx, "dry_run"] = "reject"
        return

    if feed_def.screenshot_rejector:
        if (
            pd.isna(best_writing_result.screenshot_rejection_status)
            or best_writing_result.screenshot_rejection_status != "PASS"
        ):
            if effective_dry_run:
                writing_results_df.at[best_row_idx, "dry_run"] = "reject"
            return

    if effective_dry_run:
        writing_results_df.at[best_row_idx, "dry_run"] = "submit"
        return

    note_records = submitted_note_cache.df
    post_id = post_with_context.post.post_id
    try:
        note_id = None
        successful_submitter = None

        crh_notes = note_records[
            (note_records["post_id"] == post_id)
            & (note_records["current_status"] == "CURRENTLY_RATED_HELPFUL")
        ]
        if not crh_notes.empty:
            raise ValueError("Post already has a CURRENTLY_RATED_HELPFUL note")

        post_notes = note_records[note_records["post_id"] == post_id]
        if len(post_notes) >= arena_config.max_published_notes_per_post:
            raise ValueError(
                f"Max published notes per post exceeded ({len(post_notes)} >= {arena_config.max_published_notes_per_post})"
            )
        current_notes = post_notes[post_notes["deleted_at_millis"].isna()]
        if len(current_notes) >= arena_config.max_current_notes_per_post:
            raise ValueError(
                f"Max current notes per post exceeded ({len(current_notes)} >= {arena_config.max_current_notes_per_post})"
            )

        if post_policy == "allow_revisions":
            revision_status = best_writing_result.get("revision_rejection_status")
            if pd.isna(revision_status):
                if not post_notes.empty:
                    raise ValueError(
                        "No revision rejector result and post already has a note"
                    )
            elif revision_status == "PASS":
                pass

        elif post_policy == "prioritize":
            if not post_notes.empty:
                raise ValueError("Multiple notes not allowed (prioritize)")
        else:
            if not note_records[
                (note_records["post_id"] == post_id)
                & (note_records["writer_name"] == writer.writer_name)
            ].empty:
                raise ValueError("Writer already submitted a note on this post")

        eligible_submitters = _get_eligible_submitters(
            arena_config.submission_configs,
            writer,
            post_id,
            note_records,
        )
        if len(eligible_submitters) == 0:
            raise ValueError("No submitters eligible.")
        for submitter in eligible_submitters:
            try:
                note_id = submit_note(
                    oauth=oauth_sessions[submitter],
                    note_text=best_writing_result.grok_note,
                    post_id=best_writing_result.post_id,
                    misleading_tags=best_writing_result.misleading_tags,
                )
                successful_submitter = submitter
                break
            except NoteSubmissionError as e:
                logger.exception(
                    f"Note submission error on post {post_with_context.post.post_id} for writer {writer.writer_name} with submitter {submitter}: {e}"
                )
                if e.status_code == 400:
                    raise
        if note_id is None:
            raise ValueError("All submission attempts failed.")
        assert note_id is not None, "note_id must be set"
        assert successful_submitter is not None, "successful_submitter must be set"
        writing_results_df.at[best_row_idx, "note_id"] = note_id
        writing_results_df.at[best_row_idx, "submitter"] = successful_submitter

        post = post_with_context.post
        submitted_note_cache.add_submitted_note(
            note_id=note_id,
            post_id=post.post_id,
            writer_name=writer.writer_name,
            submitter=successful_submitter,
            note_text=best_writing_result.grok_note,
            post_text=best_writing_result.post_text
            if pd.notna(best_writing_result.post_text)
            else None,
            author_id=post.author_id,
            username=post.username,
            enqueued_at=post_with_context.enqueued_at,
            lang=post.lang,
            retweet_count=post.public_metrics.retweet_count
            if post.public_metrics
            else None,
            reply_count=post.public_metrics.reply_count
            if post.public_metrics
            else None,
            like_count=post.public_metrics.like_count if post.public_metrics else None,
            quote_count=post.public_metrics.quote_count
            if post.public_metrics
            else None,
            bookmark_count=post.public_metrics.bookmark_count
            if post.public_metrics
            else None,
            impression_count=post.public_metrics.impression_count
            if post.public_metrics
            else None,
            author_followers_count=post.author_public_metrics.followers_count
            if post.author_public_metrics
            else None,
            author_following_count=post.author_public_metrics.following_count
            if post.author_public_metrics
            else None,
            author_tweet_count=post.author_public_metrics.tweet_count
            if post.author_public_metrics
            else None,
            author_listed_count=post.author_public_metrics.listed_count
            if post.author_public_metrics
            else None,
            author_like_count=post.author_public_metrics.like_count
            if post.author_public_metrics
            else None,
            author_media_count=post.author_public_metrics.media_count
            if post.author_public_metrics
            else None,
            author_verified_type=post.author_verified_type,
            author_parody=post.author_parody,
            hist_note_count=post_with_context.author_history.hist_note_count
            if post_with_context.author_history
            else None,
            hist_crh_count=post_with_context.author_history.hist_crh_count
            if post_with_context.author_history
            else None,
            hist_crnh_count=post_with_context.author_history.hist_crnh_count
            if post_with_context.author_history
            else None,
            hist_total_ratings=post_with_context.author_history.hist_total_ratings
            if post_with_context.author_history
            else None,
            notable_post_prediction=post_with_context.notable_post_prediction,
            api_feed=post_with_context.api_feed,
            timed_feed=post_with_context.timed_feed,
            num_unique_sources=len(post_with_context.suggested_sources),
            total_source_suggestions=sum(
                s.count for s in post_with_context.suggested_sources
            ),
            num_note_request_suggestions=len(
                post_with_context.note_request_suggestions
            ),
            has_photo=any(m.media_type == "photo" for m in post.media),
            has_video=any(m.media_type == "video" for m in post.media),
        )
    except Exception as e:
        writing_results_df.at[best_row_idx, "note_submission_error"] = str(e)
