import asyncio
import json
import traceback

import pandas as pd

from data_models.arena_config import RecentContextRejector, FeedDefinition, ArenaConfig
from data_models.submitted_note_cache import PriorExamples, PriorNote
from data_models.writer_data_models import (
    PostWithContext,
    RejectorResult,
    RejectorSampleResult,
)
from note_writer.grok_client import GrokClient

from utils.log_setup import get_logger

logger = get_logger("recent_ctx")


def build_prior_examples_text(prior_examples: PriorExamples) -> tuple[str, int]:
    all_notes: list[PriorNote] = sorted(
        prior_examples.crh_notes + prior_examples.crnh_notes,
        key=lambda n: n.note_id,
    )
    lines: list[str] = []
    for i, note in enumerate(all_notes):
        assert note.note_text, f"Prior example {note.note_id} has empty note_text"
        assert note.post_text, f"Prior example {note.note_id} has empty post_text"
        entry = f"--- Example {i + 1} ---\n"
        entry += f"noteId: {note.note_id}\ntweetId: {note.post_id}\n"
        entry += f"Note text: {note.note_text[:500]}\n"
        entry += f"Post text: {note.post_text[:500]}\n"
        entry += f"Status: {note.status}\n"
        lines.append(entry)
    return "\n".join(lines), len(all_notes)


def build_candidate_text(
    post_link: str,
    note_text: str,
    post_text: str,
) -> str:
    assert note_text, "Candidate note_text must be non-empty"
    assert post_text, "Candidate post_text must be non-empty"
    text = f"Post: {post_link}\n"
    text += f"Note text: {note_text[:500]}\n"
    text += f"Post text: {post_text[:500]}\n"
    return text


def build_user_message(
    prior_text: str,
    n_used: int,
    candidate_text: str,
) -> str:
    return (
        f"Here are {n_used} prior Community Notes with their ratings:\n\n"
        f"{prior_text}\n\n"
        f"Now classify this candidate note:\n\n"
        f"{candidate_text}\n\n"
        f"Predict whether this note will be CURRENTLY_RATED_HELPFUL or CURRENTLY_RATED_NOT_HELPFUL."
    )


def parse_prediction_to_score(prediction: str) -> float:
    if prediction == "LIKELY_HELPFUL":
        return 1.0
    return 0.0


async def query_recent_context_rejector(
    rejector_config: RecentContextRejector,
    post_with_context: PostWithContext,
    note_text: str,
    post_text: str | None,
    prior_examples: PriorExamples,
    xai_api_key: str,
    semaphore: asyncio.Semaphore,
    pass_threshold: float,
) -> RejectorResult:
    post_link = f"https://x.com/{post_with_context.post.username}/status/{post_with_context.post.post_id}"
    prior_text, n_used = build_prior_examples_text(prior_examples)
    candidate_text = build_candidate_text(post_link, note_text, post_text)
    user_message = build_user_message(prior_text, n_used, candidate_text)

    async def _single_query() -> RejectorSampleResult:
        async with semaphore:
            sample_result = RejectorSampleResult(
                model=rejector_config.model_name,
            )
            try:
                llm_client = GrokClient(
                    api_key=xai_api_key,
                    model=rejector_config.model_name,
                    model_uri=rejector_config.model_uri,
                    enable_web_image_understanding=False,
                    enable_x_image_understanding=False,
                    enable_x_video_understanding=False,
                    temperature=rejector_config.temperature,
                    timeout=rejector_config.timeout,
                )
                grok_output = await llm_client.get_grok_response(
                    prompt=user_message,
                    timeout=rejector_config.timeout,
                    system_prompt=rejector_config.rejector_prompt,
                    expect_json=True,
                    retries=rejector_config.max_retries_llm,
                    base_delay=rejector_config.retry_base_delay,
                    post_id=post_with_context.post.post_id,
                )
                sample_result.response_id = grok_output.response_id
                sample_result.latency = grok_output.latency
                if grok_output.content is None:
                    raise ValueError("Grok output content unavailable.")
                response = json.loads(grok_output.content)
                prediction = response.get("prediction", "")
                sample_result.score = parse_prediction_to_score(prediction)
                sample_result.reasoning = response.get("reasoning", "")
                return sample_result
            except Exception as e:
                logger.exception(
                    f"Error in recent_context_rejector _single_query: {type(e).__name__}: {e}"
                )
                sample_result.error = f"{type(e).__name__}: {e}"
                return sample_result

    sample_tasks = [_single_query() for _ in range(rejector_config.num_samples)]
    sample_results = await asyncio.gather(*sample_tasks)

    all_completed = all(r.error is None for r in sample_results)
    scores = [r.score for r in sample_results if r.score is not None]
    mean_score = (
        sum(scores) / len(scores)
        if ((len(scores) == rejector_config.num_samples) and all_completed)
        else None
    )

    if (not all_completed) or (mean_score is None):
        status = "ERROR"
    elif mean_score >= pass_threshold:
        status = "PASS"
    else:
        status = "REJECT"

    crh_ids = [n.note_id for n in prior_examples.crh_notes]
    crnh_ids = [n.note_id for n in prior_examples.crnh_notes]

    return RejectorResult(
        status=status,
        mean_score=mean_score,
        sample_results=list(sample_results),
        n_crh_examples=len(crh_ids),
        n_crnh_examples=len(crnh_ids),
        n_deleted_as_crnh=prior_examples.n_deleted_as_crnh,
        crh_note_id_range=(min(crh_ids), max(crh_ids)) if crh_ids else None,
        crnh_note_id_range=(min(crnh_ids), max(crnh_ids)) if crnh_ids else None,
    )


async def query_recent_context_rejector_for_post(
    arena_config: ArenaConfig,
    post_with_context: PostWithContext,
    writing_results_df: pd.DataFrame,
    feed_def: FeedDefinition,
    best_writing_result: pd.Series,
    best_row_idx: int,
    prior_examples: PriorExamples,
    xai_api_key: str,
    semaphore: asyncio.Semaphore,
    df_lock: asyncio.Lock,
) -> None:
    rejector_config = next(
        (
            r
            for r in arena_config.recent_context_rejectors
            if r.rejector_name == feed_def.recent_context_rejector
        ),
        None,
    )
    assert rejector_config is not None, (
        f"RecentContextRejector {feed_def.recent_context_rejector} not found in "
        f"arena_config.recent_context_rejectors"
    )

    rejector_result = await query_recent_context_rejector(
        rejector_config=rejector_config,
        post_with_context=post_with_context,
        note_text=best_writing_result.grok_note,
        post_text=str(best_writing_result.post_text)
        if pd.notna(best_writing_result.get("post_text"))
        else None,
        prior_examples=prior_examples,
        xai_api_key=xai_api_key,
        semaphore=semaphore,
        pass_threshold=feed_def.recent_context_rejector_pass_threshold,
    )

    async with df_lock:
        writing_results_df.at[best_row_idx, "recent_context_rejector"] = (
            rejector_config.rejector_name
        )
        writing_results_df.at[best_row_idx, "recent_context_rejection_status"] = (
            rejector_result.status
        )
        writing_results_df.at[best_row_idx, "recent_context_rejection_score"] = (
            rejector_result.mean_score
            if rejector_result.mean_score is not None
            else pd.NA
        )
        writing_results_df.at[best_row_idx, "recent_context_rejector_content"] = (
            json.dumps(
                [r.model_dump() for r in rejector_result.sample_results],
                indent=4,
            )
        )
