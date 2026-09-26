import asyncio
import json
import traceback
from difflib import SequenceMatcher

import pandas as pd

from data_models.arena_config import RevisionRejector, ArenaConfig
from data_models.submitted_note_cache import PriorPostNote
from data_models.writer_data_models import (
    PostWithContext,
    RejectorResult,
    RejectorSampleResult,
)
from note_writer.grok_client import GrokClient
from utils.url_utils import remove_urls

from utils.log_setup import get_logger

logger = get_logger("revision")


def compute_similarity(a: str, b: str) -> float:
    return round(SequenceMatcher(None, a, b).ratio(), 2)


async def query_revision_rejector(
    rejector_config: RevisionRejector,
    post_with_context: PostWithContext,
    candidate_note: str,
    baseline_note: str,
    xai_api_key: str,
    semaphore: asyncio.Semaphore,
    pass_threshold: float,
) -> RejectorResult:
    post_link = f"https://x.com/{post_with_context.post.username}/status/{post_with_context.post.post_id}"
    sim_no_urls = compute_similarity(
        remove_urls(baseline_note), remove_urls(candidate_note)
    )
    prompt = rejector_config.rejector_prompt.format(
        post_link=post_link,
        baseline_note=baseline_note,
        candidate_note=candidate_note,
    )

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
                    enable_web_image_understanding=True,
                    enable_x_image_understanding=True,
                    enable_x_video_understanding=True,
                    temperature=rejector_config.temperature,
                    timeout=rejector_config.timeout,
                )
                grok_output = await llm_client.get_grok_response(
                    prompt,
                    timeout=rejector_config.timeout,
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
                raw_score = response["score"]
                sample_result.score = raw_score * (1 - sim_no_urls)
                sample_result.reasoning = response.get("reasoning", "")
                return sample_result
            except Exception as e:
                logger.exception(
                    f"Error in revision_rejector _single_query: {type(e).__name__}: {e}"
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

    return RejectorResult(
        status=status,
        mean_score=mean_score,
        sample_results=list(sample_results),
    )


async def query_revision_rejector_for_post(
    arena_config: ArenaConfig,
    post_with_context: PostWithContext,
    writing_results_df: pd.DataFrame,
    best_writing_result: pd.Series,
    best_row_idx: int,
    prior_post_notes: list[PriorPostNote],
    xai_api_key: str,
    semaphore: asyncio.Semaphore,
    df_lock: asyncio.Lock,
    revision_rejector_name: str,
    revision_rejector_pass_threshold: float,
) -> None:
    rejector_config = next(
        (
            r
            for r in arena_config.revision_rejectors
            if r.rejector_name == revision_rejector_name
        ),
        None,
    )
    assert rejector_config is not None, (
        f"RevisionRejector {revision_rejector_name} not found in "
        f"arena_config.revision_rejectors"
    )

    comparison_results = []
    for prior_note in prior_post_notes:
        rejector_result = await query_revision_rejector(
            rejector_config=rejector_config,
            post_with_context=post_with_context,
            candidate_note=best_writing_result.grok_note,
            baseline_note=prior_note.note_text,
            xai_api_key=xai_api_key,
            semaphore=semaphore,
            pass_threshold=revision_rejector_pass_threshold,
        )
        sim_no_urls = compute_similarity(
            remove_urls(prior_note.note_text),
            remove_urls(best_writing_result.grok_note),
        )
        comparison_results.append(
            {
                "baseline_note_id": prior_note.note_id,
                "baseline_writer_name": prior_note.writer_name,
                "baseline_is_deleted": prior_note.is_deleted,
                "baseline_current_status": prior_note.current_status,
                "sim_no_urls": sim_no_urls,
                "status": rejector_result.status,
                "scaled_score": rejector_result.mean_score,
                "sample_results": [
                    r.model_dump() for r in rejector_result.sample_results
                ],
            }
        )

    statuses = [c["status"] for c in comparison_results]
    if any(s == "ERROR" for s in statuses):
        overall_status = "ERROR"
    elif all(s == "PASS" for s in statuses):
        overall_status = "PASS"
    else:
        overall_status = "REJECT"

    scores = [
        c["scaled_score"] for c in comparison_results if c["scaled_score"] is not None
    ]
    overall_score = min(scores) if scores and overall_status != "ERROR" else None

    async with df_lock:
        writing_results_df.at[best_row_idx, "revision_rejector"] = (
            rejector_config.rejector_name
        )
        writing_results_df.at[best_row_idx, "revision_rejection_status"] = (
            overall_status
        )
        writing_results_df.at[best_row_idx, "revision_rejection_score"] = (
            overall_score if overall_score is not None else pd.NA
        )
        writing_results_df.at[best_row_idx, "revision_rejector_content"] = json.dumps(
            comparison_results,
            indent=4,
        )
