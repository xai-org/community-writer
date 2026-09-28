import asyncio
import json
import traceback

import pandas as pd

from data_models.arena_config import RLRejector, FeedDefinition, ArenaConfig
from data_models.writer_data_models import (
    PostWithContext,
    RejectorResult,
    RejectorSampleResult,
)
from note_writer.grok_client import GrokClient
from note_writer.suggestion_context import build_rejector_suggestion_block

from utils.log_setup import get_logger

logger = get_logger("rl_rejector")


async def query_rl_rejector(
    rejector_config: RLRejector,
    post_with_context: PostWithContext,
    note_text: str,
    xai_api_key: str,
    semaphore: asyncio.Semaphore,
    pass_threshold: float,
    attempt_id: int,
) -> RejectorResult:
    post_link = f"https://x.com/{post_with_context.post.username}/status/{post_with_context.post.post_id}"
    prompt = rejector_config.rejector_prompt.format(
        post_link=post_link,
        note_text=note_text,
        suggested_sources=build_rejector_suggestion_block(
            post_with_context, attempt_id
        ),
    )

    async def _single_rejector_query() -> RejectorSampleResult:
        async with semaphore:
            rejector_result = RejectorSampleResult(
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
                rejector_result.response_id = grok_output.response_id
                rejector_result.latency = grok_output.latency
                if grok_output.content is None:
                    raise ValueError("Grok output content unavailable.")
                response = json.loads(grok_output.content)
                rejector_result.score = response["score"]
                rejector_result.reasoning = response["reasoning"]
                return rejector_result
            except Exception as e:
                logger.exception(
                    f"Error in _single_rejector_query: {type(e).__name__}: {e}"
                )
                rejector_result.error = f"{type(e).__name__}: {e}"
                return rejector_result

    sample_tasks = [
        _single_rejector_query() for _ in range(rejector_config.num_samples)
    ]
    sample_results = await asyncio.gather(*sample_tasks)

    all_completed = all(result.error is None for result in sample_results)
    scores = [result.score for result in sample_results if result.score is not None]
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


async def query_rl_rejector_for_post(
    arena_config: ArenaConfig,
    post_with_context: PostWithContext,
    writing_results_df: pd.DataFrame,
    feed_def: FeedDefinition,
    best_writing_result: pd.Series,
    best_row_idx: int,
    xai_api_key: str,
    semaphore: asyncio.Semaphore,
    df_lock: asyncio.Lock,
) -> None:
    rejector_config = next(
        (
            r
            for r in arena_config.rl_rejectors
            if r.rejector_name == feed_def.rl_rejector
        ),
        None,
    )
    assert rejector_config is not None, (
        f"Rejector {feed_def.rl_rejector} not found in arena_config.rl_rejectors"
    )

    rejector_result = await query_rl_rejector(
        rejector_config=rejector_config,
        post_with_context=post_with_context,
        note_text=best_writing_result.grok_note,
        xai_api_key=xai_api_key,
        semaphore=semaphore,
        pass_threshold=feed_def.rejector_pass_threshold,
        attempt_id=int(best_writing_result.attempt_id),
    )

    async with df_lock:
        writing_results_df.at[best_row_idx, "rejector"] = rejector_config.rejector_name
        writing_results_df.at[best_row_idx, "rejection_status"] = rejector_result.status
        writing_results_df.at[best_row_idx, "rejection_score"] = (
            rejector_result.mean_score
            if rejector_result.mean_score is not None
            else pd.NA
        )
        writing_results_df.at[best_row_idx, "rejector_content"] = json.dumps(
            [r.model_dump() for r in rejector_result.sample_results], indent=4
        )
