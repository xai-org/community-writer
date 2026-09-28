import asyncio
import json
import re
import traceback
from collections.abc import Awaitable, Callable

import pandas as pd
from playwright.async_api import async_playwright
from requests_oauthlib import OAuth1Session  # type: ignore

from browser.session import launch_browser

from cnapi.evaluate_note import evaluate_note
from data_models.writer_data_models import (
    AuthorHistory,
    GrokOutput,
    NoteResult,
    Post,
    PostWithContext,
    ProposedMisleadingNote,
)
from note_writer.grok_client import GrokClient
from note_writer.misleading_tags import get_misleading_tags
from note_writer.note_length import is_over_length, supports_length_restriction
from note_writer.suggestion_context import build_writer_suggestion_block
from utils.url_utils import extract_and_validate_urls
from data_models.arena_config import (
    GrokWriter,
    ArenaConfig,
    is_writer_active_for_work_item,
)
from data_models.environment_variables import KeySet
from data_models.submitted_note_cache import PriorExamples
from rejectors import find_best_draft, is_draft_on_track_for_submission
from rejectors.rl_rejector import query_rl_rejector_for_post
from rejectors.recent_context_rejector import query_recent_context_rejector_for_post
from rejectors.revision_rejector import query_revision_rejector_for_post
from rejectors.screenshot_rejector import (
    query_screenshot_rejector_for_post,
    any_note_needs_screenshot_rejection,
)

from utils.log_setup import get_logger

logger = get_logger("write_note")


_RESULT_DTYPES: dict[str, object] = {
    "submission_eligible": pd.BooleanDtype(),
    "author_parody": pd.BooleanDtype(),
    "has_photo": pd.BooleanDtype(),
    "has_video": pd.BooleanDtype(),
    "post_id": pd.Int64Dtype(),
    "author_id": pd.Int64Dtype(),
    "note_id": pd.Int64Dtype(),
    "config_timestamp": pd.Int64Dtype(),
    "enqueued_at": pd.Int64Dtype(),
    "work_started_at": pd.Int64Dtype(),
    "work_finished_at": pd.Int64Dtype(),
    "peak_rss_bytes": pd.Int64Dtype(),
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
    "hist_note_count": pd.Int64Dtype(),
    "hist_crh_count": pd.Int64Dtype(),
    "hist_crnh_count": pd.Int64Dtype(),
    "hist_total_ratings": pd.Int64Dtype(),
    "num_unique_sources": pd.Int64Dtype(),
    "total_source_suggestions": pd.Int64Dtype(),
    "num_note_request_suggestions": pd.Int64Dtype(),
    "attempt_id": pd.Int64Dtype(),
    "lang": pd.StringDtype(),
    "author_verified_type": pd.StringDtype(),
    "api_feed": pd.StringDtype(),
    "timed_feed": pd.StringDtype(),
    "username": pd.StringDtype(),
    "post_text": pd.StringDtype(),
    "suggested_sources": pd.StringDtype(),
    "note_request_suggestions": pd.StringDtype(),
    "quoted_post_text": pd.StringDtype(),
    "reply_to_post_text": pd.StringDtype(),
    "retweeted_post_text": pd.StringDtype(),
    "writer_name": pd.StringDtype(),
    "writing_prompt": pd.StringDtype(),
    "grok_note": pd.StringDtype(),
    "over_length_note": pd.StringDtype(),
    "final_prompt": pd.StringDtype(),
    "final_reasoning_content": pd.StringDtype(),
    "final_response_content": pd.StringDtype(),
    "refusal": pd.StringDtype(),
    "error": pd.StringDtype(),
    "task_timeout": pd.StringDtype(),
    "task_memory_error": pd.StringDtype(),
    "task_exception": pd.StringDtype(),
    "tool_calls": pd.StringDtype(),
    "note_submission_error": pd.StringDtype(),
    "submitter": pd.StringDtype(),
    "parsed_trace": pd.StringDtype(),
    "rejector": pd.StringDtype(),
    "rejection_status": pd.StringDtype(),
    "rejector_content": pd.StringDtype(),
    "screenshot_rejector": pd.StringDtype(),
    "screenshot_rejection_status": pd.StringDtype(),
    "screenshot_rejector_content": pd.StringDtype(),
    "recent_context_rejector": pd.StringDtype(),
    "recent_context_rejection_status": pd.StringDtype(),
    "recent_context_rejector_content": pd.StringDtype(),
    "revision_rejector": pd.StringDtype(),
    "revision_rejection_status": pd.StringDtype(),
    "revision_rejector_content": pd.StringDtype(),
    "endpoint": pd.StringDtype(),
    "model": pd.StringDtype(),
    "response_id": pd.StringDtype(),
    "multi_note_policy": pd.StringDtype(),
    "dry_run": pd.StringDtype(),
    "co_threshold": pd.Float64Dtype(),
    "co_score": pd.Float64Dtype(),
    "rejection_score": pd.Float64Dtype(),
    "screenshot_rejection_score": pd.Float64Dtype(),
    "recent_context_rejection_score": pd.Float64Dtype(),
    "revision_rejection_score": pd.Float64Dtype(),
    "latency": pd.Float64Dtype(),
    "notable_post_prediction": pd.Float64Dtype(),
    "notable_post_threshold": pd.Float64Dtype(),
    "misleading_tags": object,
    "unsupported_media_types": object,
    "citations": object,
}


def _user_context_columns(post_with_context: PostWithContext) -> dict:
    return {
        "num_unique_sources": len(post_with_context.suggested_sources),
        "total_source_suggestions": sum(
            s.count for s in post_with_context.suggested_sources
        ),
        "num_note_request_suggestions": len(post_with_context.note_request_suggestions),
        "suggested_sources": json.dumps(
            [
                {"count": s.count, "source_url": s.link}
                for s in post_with_context.suggested_sources
            ]
        ),
        "note_request_suggestions": json.dumps(
            post_with_context.note_request_suggestions
        ),
    }


_BOLD_PATTERN = re.compile(r"\*\*")


def _strip_markdown(note_or_refusal: str) -> str:
    return _BOLD_PATTERN.sub("", note_or_refusal)


def _check_for_unsupported_media(
    post: Post, allowed_media_types: list[str]
) -> set[str]:
    unsupported_media_types = set()
    for media in post.media:
        if media.media_type not in allowed_media_types:
            unsupported_media_types.add(media.media_type)
    return unsupported_media_types


def _check_for_unsupported_media_in_post_with_context(
    post_with_context: PostWithContext, allowed_media_types: list[str]
) -> set[str]:
    unsupported_media_types = _check_for_unsupported_media(
        post_with_context.post, allowed_media_types
    )
    if post_with_context.quoted_post:
        unsupported_media_types.update(
            _check_for_unsupported_media(
                post_with_context.quoted_post, allowed_media_types
            )
        )
    if post_with_context.in_reply_to_post:
        unsupported_media_types.update(
            _check_for_unsupported_media(
                post_with_context.in_reply_to_post, allowed_media_types
            )
        )
    if post_with_context.retweeted_post:
        unsupported_media_types.update(
            _check_for_unsupported_media(
                post_with_context.retweeted_post, allowed_media_types
            )
        )
    return unsupported_media_types


async def _research_post_and_write_note(
    post_with_context: PostWithContext,
    writer_config: GrokWriter,
    xai_api_key: str,
    note_result: NoteResult,
    allow_over_length: bool,
) -> None:
    llm_client = GrokClient(
        api_key=xai_api_key,
        model=writer_config.model_name,
        model_uri=writer_config.model_uri,
        enable_web_image_understanding="photo" in writer_config.allowed_media_types,
        enable_x_image_understanding="photo" in writer_config.allowed_media_types,
        enable_x_video_understanding="video" in writer_config.allowed_media_types,
        temperature=writer_config.temperature,
        timeout=writer_config.timeout,
    )
    grok_output: GrokOutput = await llm_client.get_grok_response(
        note_result.writing_prompt,
        timeout=writer_config.timeout,
        retries=writer_config.max_retries_llm,
        base_delay=writer_config.retry_base_delay,
        post_id=post_with_context.post.post_id,
    )
    note_result.grok_output = grok_output
    note_or_refusal_str = grok_output.content
    if note_or_refusal_str is None:
        raise ValueError("Grok output content unavailable.")
    note_or_refusal_str = _strip_markdown(note_or_refusal_str)

    if ("NO NOTE NEEDED" in note_or_refusal_str) or (
        "NOT ENOUGH EVIDENCE TO WRITE A GOOD COMMUNITY NOTE" in note_or_refusal_str
    ):
        note_result.refusal = note_or_refusal_str
        return

    if not allow_over_length and is_over_length(note_or_refusal_str):
        note_result.over_length_note = note_or_refusal_str
        return

    misleading_tags = await get_misleading_tags(
        post_with_context,
        note_or_refusal_str,
        llm_client,
        timeout=writer_config.timeout,
        retries=writer_config.max_retries_llm,
        base_delay=writer_config.retry_base_delay,
    )

    citations = note_result.grok_output.citations if note_result.grok_output else None
    failed_urls = await extract_and_validate_urls(note_or_refusal_str, citations or [])
    error_details = []
    if failed_urls:
        for url, status_code in failed_urls:
            if status_code is not None:
                error_details.append(f"  {status_code:<4} {url}")
            else:
                error_details.append(f"  None {url}")

        error_details_str = "\n".join(error_details)
        error_message = (
            f"One or more URLs returned non-2xx/3xx status codes:\n{error_details_str}"
        )
        note_result.error = error_message
        return

    note_result.note = ProposedMisleadingNote(
        post_id=post_with_context.post.post_id,
        note_text=note_or_refusal_str,
        misleading_tags=misleading_tags,
    )
    return


def _should_retry_with_length_restriction(
    writer: GrokWriter,
    results: list[NoteResult],
) -> bool:
    if not supports_length_restriction(writer.writer_prompt, writer.length_restriction):
        return False
    if any(result.note is not None for result in results):
        return False
    return any(result.over_length_note for result in results)


async def _draft_with_length_retry(
    writer: GrokWriter,
    run_attempt: Callable[[GrokWriter, int, bool], Awaitable[NoteResult]],
) -> list[NoteResult]:
    attempts = range(writer.num_drafts)
    results = list(
        await asyncio.gather(*(run_attempt(writer, i, False) for i in attempts))
    )
    if not _should_retry_with_length_restriction(writer, results):
        return results

    retried = list(
        await asyncio.gather(*(run_attempt(writer, i, True) for i in attempts))
    )
    for retry_result, first_result in zip(retried, results):
        retry_result.over_length_note = first_result.over_length_note
    return retried


async def _process_writer_attempt(
    post_with_context: PostWithContext,
    writer_config: GrokWriter,
    xai_api_key: str,
    oauth: OAuth1Session,
    attempt_id: int,
    length_restricted: bool,
    co_threshold: float | None = None,
) -> NoteResult:
    post_link = f"https://x.com/{post_with_context.post.username}/status/{post_with_context.post.post_id}"
    suggested_sources = build_writer_suggestion_block(post_with_context, attempt_id)
    prompt = writer_config.writer_prompt.format(
        post_link=post_link,
        suggested_sources=suggested_sources,
        length_restriction=writer_config.length_restriction
        if length_restricted
        else "",
    )

    note_result = NoteResult(
        post=post_with_context,
        writer_name=writer_config.writer_name,
        writing_prompt=prompt,
        attempt_id=attempt_id,
        co_threshold=co_threshold,
    )

    unsupported_media_types = _check_for_unsupported_media_in_post_with_context(
        post_with_context, writer_config.allowed_media_types
    )
    if unsupported_media_types:
        note_result.unsupported_media_types = unsupported_media_types
        return note_result

    try:
        await _research_post_and_write_note(
            post_with_context,
            writer_config,
            xai_api_key,
            note_result,
            allow_over_length=length_restricted,
        )

        if note_result.note is not None:
            co_score = await evaluate_note(
                oauth, note_result.note.note_text, note_result.note.post_id
            )
            note_result.co_score = co_score
        return note_result
    except Exception as e:
        logger.exception(f"Error in _process_writer_attempt: {type(e).__name__}: {e}")
        note_result.error = f"{type(e).__name__}: {e}"
        return note_result


def _writing_results_to_dataframe(
    writing_results: list[NoteResult],
    config_timestamp: int,
    enqueued_at: int | None = None,
    notable_post_prediction: float | None = None,
    notable_post_threshold: float | None = None,
    author_history: AuthorHistory | None = None,
    multi_note_policy: str | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "post_id": result.post.post.post_id,
                "api_feed": result.post.api_feed,
                "timed_feed": result.post.timed_feed,
                "author_id": result.post.post.author_id,
                "username": result.post.post.username,
                "post_text": result.post.post.text,
                "quoted_post_text": result.post.quoted_post.text
                if result.post.quoted_post
                else pd.NA,
                "reply_to_post_text": result.post.in_reply_to_post.text
                if result.post.in_reply_to_post
                else pd.NA,
                "retweeted_post_text": result.post.retweeted_post.text
                if result.post.retweeted_post
                else pd.NA,
                "lang": result.post.post.lang,
                "retweet_count": result.post.post.public_metrics.retweet_count
                if result.post.post.public_metrics
                else pd.NA,
                "reply_count": result.post.post.public_metrics.reply_count
                if result.post.post.public_metrics
                else pd.NA,
                "like_count": result.post.post.public_metrics.like_count
                if result.post.post.public_metrics
                else pd.NA,
                "quote_count": result.post.post.public_metrics.quote_count
                if result.post.post.public_metrics
                else pd.NA,
                "bookmark_count": result.post.post.public_metrics.bookmark_count
                if result.post.post.public_metrics
                else pd.NA,
                "impression_count": result.post.post.public_metrics.impression_count
                if result.post.post.public_metrics
                else pd.NA,
                "author_verified_type": result.post.post.author_verified_type,
                "author_parody": result.post.post.author_parody,
                "author_followers_count": result.post.post.author_public_metrics.followers_count
                if result.post.post.author_public_metrics
                else pd.NA,
                "author_following_count": result.post.post.author_public_metrics.following_count
                if result.post.post.author_public_metrics
                else pd.NA,
                "author_tweet_count": result.post.post.author_public_metrics.tweet_count
                if result.post.post.author_public_metrics
                else pd.NA,
                "author_listed_count": result.post.post.author_public_metrics.listed_count
                if result.post.post.author_public_metrics
                else pd.NA,
                "author_like_count": result.post.post.author_public_metrics.like_count
                if result.post.post.author_public_metrics
                else pd.NA,
                "author_media_count": result.post.post.author_public_metrics.media_count
                if result.post.post.author_public_metrics
                else pd.NA,
                "notable_post_prediction": notable_post_prediction
                if notable_post_prediction is not None
                else pd.NA,
                "notable_post_threshold": notable_post_threshold
                if notable_post_threshold is not None
                else pd.NA,
                **_user_context_columns(result.post),
                "has_photo": any(
                    m.media_type == "photo" for m in result.post.post.media
                ),
                "has_video": any(
                    m.media_type == "video" for m in result.post.post.media
                ),
                "hist_note_count": author_history.hist_note_count
                if author_history
                else pd.NA,
                "hist_crh_count": author_history.hist_crh_count
                if author_history
                else pd.NA,
                "hist_crnh_count": author_history.hist_crnh_count
                if author_history
                else pd.NA,
                "hist_total_ratings": author_history.hist_total_ratings
                if author_history
                else pd.NA,
                "writer_name": result.writer_name,
                "attempt_id": result.attempt_id,
                "writing_prompt": result.writing_prompt,
                "co_threshold": result.co_threshold
                if result.co_threshold is not None
                else pd.NA,
                "grok_note": result.note.note_text
                if result.note is not None
                else pd.NA,
                "over_length_note": result.over_length_note
                if result.over_length_note is not None
                else pd.NA,
                "misleading_tags": [tag.value for tag in result.note.misleading_tags]
                if result.note is not None
                else pd.NA,
                "unsupported_media_types": result.unsupported_media_types,
                "parsed_trace": result.grok_output.parsed_trace
                if result.grok_output
                else pd.NA,
                "final_prompt": result.grok_output.final_prompt
                if result.grok_output
                else pd.NA,
                "final_reasoning_content": result.grok_output.final_reasoning_content
                if result.grok_output
                else pd.NA,
                "final_response_content": result.grok_output.final_response_content
                if result.grok_output
                else pd.NA,
                "refusal": result.refusal,
                "error": result.error,
                "task_timeout": pd.NA,
                "task_memory_error": pd.NA,
                "task_exception": pd.NA,
                "co_score": result.co_score,
                "citations": (
                    (
                        result.grok_output.citations
                        if result.grok_output.citations is not None
                        else pd.NA
                    )
                    if result.grok_output
                    else pd.NA
                ),
                "tool_calls": (
                    json.dumps(result.grok_output.tool_calls)
                    if result.grok_output and result.grok_output.tool_calls is not None
                    else pd.NA
                ),
                "endpoint": "eapi" if result.grok_output else pd.NA,
                "model": result.grok_output.model if result.grok_output else pd.NA,
                "response_id": result.grok_output.response_id
                if result.grok_output
                else pd.NA,
                "latency": result.grok_output.latency if result.grok_output else pd.NA,
                "rejector": pd.NA,
                "rejection_score": pd.NA,
                "rejection_status": pd.NA,
                "rejector_content": pd.NA,
                "screenshot_rejector": pd.NA,
                "screenshot_rejection_score": pd.NA,
                "screenshot_rejection_status": pd.NA,
                "screenshot_rejector_content": pd.NA,
                "recent_context_rejector": pd.NA,
                "recent_context_rejection_score": pd.NA,
                "recent_context_rejection_status": pd.NA,
                "recent_context_rejector_content": pd.NA,
                "revision_rejector": pd.NA,
                "revision_rejection_score": pd.NA,
                "revision_rejection_status": pd.NA,
                "revision_rejector_content": pd.NA,
                "note_id": pd.NA,
                "submission_eligible": pd.NA,
                "submitter": pd.NA,
                "note_submission_error": pd.NA,
                "config_timestamp": config_timestamp,
                "multi_note_policy": multi_note_policy,
                "dry_run": pd.NA,
                "enqueued_at": enqueued_at,
                "work_started_at": pd.NA,
                "work_finished_at": pd.NA,
                "peak_rss_bytes": pd.NA,
            }
            for result in writing_results
        ]
    )

    df = df.astype(_RESULT_DTYPES)
    return df


def _build_post_metadata_row(
    post_with_context: PostWithContext,
    config_timestamp: int,
    work_started_at: int,
    work_finished_at: int,
    multi_note_policy: str | None = None,
) -> dict:
    post = post_with_context.post
    pm = post.public_metrics
    apm = post.author_public_metrics
    ah = post_with_context.author_history

    row: dict = {col: pd.NA for col in _RESULT_DTYPES}
    row.update(
        {
            "post_id": post.post_id,
            "api_feed": post_with_context.api_feed,
            "timed_feed": post_with_context.timed_feed,
            "author_id": post.author_id,
            "username": post.username,
            "post_text": post.text,
            "quoted_post_text": post_with_context.quoted_post.text
            if post_with_context.quoted_post
            else pd.NA,
            "reply_to_post_text": post_with_context.in_reply_to_post.text
            if post_with_context.in_reply_to_post
            else pd.NA,
            "retweeted_post_text": post_with_context.retweeted_post.text
            if post_with_context.retweeted_post
            else pd.NA,
            "lang": post.lang,
            "retweet_count": pm.retweet_count if pm else pd.NA,
            "reply_count": pm.reply_count if pm else pd.NA,
            "like_count": pm.like_count if pm else pd.NA,
            "quote_count": pm.quote_count if pm else pd.NA,
            "bookmark_count": pm.bookmark_count if pm else pd.NA,
            "impression_count": pm.impression_count if pm else pd.NA,
            "author_verified_type": post.author_verified_type,
            "author_parody": post.author_parody,
            "author_followers_count": apm.followers_count if apm else pd.NA,
            "author_following_count": apm.following_count if apm else pd.NA,
            "author_tweet_count": apm.tweet_count if apm else pd.NA,
            "author_listed_count": apm.listed_count if apm else pd.NA,
            "author_like_count": apm.like_count if apm else pd.NA,
            "author_media_count": apm.media_count if apm else pd.NA,
            "notable_post_prediction": post_with_context.notable_post_prediction
            if post_with_context.notable_post_prediction is not None
            else pd.NA,
            **_user_context_columns(post_with_context),
            "has_photo": any(m.media_type == "photo" for m in post.media),
            "has_video": any(m.media_type == "video" for m in post.media),
            "hist_note_count": ah.hist_note_count if ah else pd.NA,
            "hist_crh_count": ah.hist_crh_count if ah else pd.NA,
            "hist_crnh_count": ah.hist_crnh_count if ah else pd.NA,
            "hist_total_ratings": ah.hist_total_ratings if ah else pd.NA,
            "config_timestamp": config_timestamp,
            "multi_note_policy": multi_note_policy,
            "enqueued_at": post_with_context.enqueued_at,
            "work_started_at": work_started_at,
            "work_finished_at": work_finished_at,
        }
    )
    return row


def build_notable_post_rejected_result(
    post_with_context: PostWithContext,
    config_timestamp: int,
    notable_post_threshold: float,
    work_started_at: int,
    work_finished_at: int,
    multi_note_policy: str | None = None,
) -> pd.DataFrame:
    row = _build_post_metadata_row(
        post_with_context,
        config_timestamp,
        work_started_at,
        work_finished_at,
        multi_note_policy,
    )
    row["notable_post_threshold"] = notable_post_threshold

    df = pd.DataFrame([row])
    df = df.astype(_RESULT_DTYPES)
    return df


FAILED_TASK_COLUMNS = ("task_timeout", "task_memory_error", "task_exception")


def build_failed_task_result(
    post_with_context: PostWithContext,
    config_timestamp: int,
    work_started_at: int,
    work_finished_at: int,
    failure_column: str,
    message: str,
    multi_note_policy: str | None = None,
    peak_rss_bytes: int | None = None,
) -> pd.DataFrame:
    assert failure_column in FAILED_TASK_COLUMNS, (
        f"unknown failure column: {failure_column!r}"
    )
    row = _build_post_metadata_row(
        post_with_context,
        config_timestamp,
        work_started_at,
        work_finished_at,
        multi_note_policy,
    )
    row[failure_column] = message
    if peak_rss_bytes is not None:
        row["peak_rss_bytes"] = peak_rss_bytes

    df = pd.DataFrame([row])
    df = df.astype(_RESULT_DTYPES)
    return df


async def _draft_and_query_rejector(
    post_with_context: PostWithContext,
    arena_config: ArenaConfig,
    xai_api_key: str,
    oauth_keyset: KeySet,
    note_concurrency: int,
    screenshot_dir: str | None = None,
    work_started_at: int | None = None,
    prior_examples: PriorExamples | None = None,
    prior_post_notes: list | None = None,
) -> pd.DataFrame:
    oauth = OAuth1Session(
        oauth_keyset.x_api_key,
        client_secret=oauth_keyset.x_api_secret_key,
        resource_owner_key=oauth_keyset.x_access_token,
        resource_owner_secret=oauth_keyset.x_access_token_secret,
    )

    feed_def = arena_config.get_api_feed_def(post_with_context.api_feed)

    semaphore = asyncio.Semaphore(note_concurrency)
    df_lock = asyncio.Lock()

    async def _process_writer_attempt_with_semaphore(
        writer: GrokWriter,
        attempt_id: int,
        length_restricted: bool,
    ) -> NoteResult:
        async with semaphore:
            return await _process_writer_attempt(
                post_with_context,
                writer,
                xai_api_key,
                oauth,
                attempt_id,
                length_restricted=length_restricted,
                co_threshold=feed_def.co_threshold,
            )

    multi_note_policy = arena_config.get_multi_note_policy(
        post_with_context.post.post_id
    )

    skip_feed_check = multi_note_policy == "once_per_writer"

    writing_tasks = [
        _draft_with_length_retry(writer, _process_writer_attempt_with_semaphore)
        for writer in arena_config.grok_writers
        if is_writer_active_for_work_item(
            writer, post_with_context, skip_feed_check=skip_feed_check
        )
    ]
    writing_results = [
        result
        for writer_results in await asyncio.gather(*writing_tasks)
        for result in writer_results
    ]
    writing_results_df = _writing_results_to_dataframe(
        writing_results,
        arena_config.config_timestamp,
        enqueued_at=post_with_context.enqueued_at,
        notable_post_prediction=post_with_context.notable_post_prediction,
        notable_post_threshold=feed_def.notable_post_threshold
        if hasattr(feed_def, "notable_post_threshold")
        else None,
        author_history=post_with_context.author_history,
        multi_note_policy=multi_note_policy,
    )

    rejector_tasks = []
    for writer in arena_config.grok_writers:
        result = find_best_draft(writer, writing_results_df)
        if result is None:
            continue
        best_writing_result, best_row_idx = result
        common_kwargs = dict(
            arena_config=arena_config,
            post_with_context=post_with_context,
            writing_results_df=writing_results_df,
            feed_def=feed_def,
            best_writing_result=best_writing_result,
            best_row_idx=best_row_idx,
            xai_api_key=xai_api_key,
            semaphore=semaphore,
            df_lock=df_lock,
        )
        if feed_def.rl_rejector:
            rejector_tasks.append(query_rl_rejector_for_post(**common_kwargs))
        if (
            feed_def.recent_context_rejector
            and prior_examples
            and (prior_examples.crh_notes or prior_examples.crnh_notes)
        ):
            rejector_tasks.append(
                query_recent_context_rejector_for_post(
                    **common_kwargs, prior_examples=prior_examples
                )
            )

        if post_with_context.timed_feed and prior_post_notes:
            rev_def = next(
                (
                    rf
                    for rf in arena_config.revision_feeds
                    if rf.name == post_with_context.timed_feed
                ),
                None,
            )
            if rev_def:
                rejector_tasks.append(
                    query_revision_rejector_for_post(
                        arena_config=arena_config,
                        post_with_context=post_with_context,
                        writing_results_df=writing_results_df,
                        best_writing_result=best_writing_result,
                        best_row_idx=best_row_idx,
                        prior_post_notes=prior_post_notes,
                        xai_api_key=xai_api_key,
                        semaphore=semaphore,
                        df_lock=df_lock,
                        revision_rejector_name=rev_def.revision_rejector,
                        revision_rejector_pass_threshold=rev_def.revision_rejector_pass_threshold,
                    )
                )
    await asyncio.gather(*rejector_tasks)

    if (
        screenshot_dir
        and work_started_at
        and any_note_needs_screenshot_rejection(
            arena_config, post_with_context, writing_results_df
        )
    ):
        async with async_playwright() as p:
            browser = await launch_browser(p)
            try:
                screenshot_tasks = []
                for writer in arena_config.grok_writers:
                    result = find_best_draft(writer, writing_results_df)
                    if result is None:
                        continue
                    best_writing_result, best_row_idx = result
                    if not feed_def.screenshot_rejector:
                        continue
                    if not is_draft_on_track_for_submission(
                        best_writing_result,
                        feed_def,
                        post_with_context.post.post_id,
                        arena_config,
                    ):
                        continue
                    screenshot_tasks.append(
                        query_screenshot_rejector_for_post(
                            arena_config=arena_config,
                            post_with_context=post_with_context,
                            writing_results_df=writing_results_df,
                            feed_def=feed_def,
                            best_writing_result=best_writing_result,
                            best_row_idx=best_row_idx,
                            xai_api_key=xai_api_key,
                            semaphore=semaphore,
                            df_lock=df_lock,
                            screenshot_dir=screenshot_dir,
                            work_started_at=work_started_at,
                            browser=browser,
                        )
                    )
                await asyncio.gather(*screenshot_tasks)
            finally:
                await browser.close()

    return writing_results_df
