import asyncio
import base64
import json
import mimetypes
import os
import traceback
from pathlib import Path

import pandas as pd

from data_models.arena_config import FeedDefinition, ScreenshotRejector, ArenaConfig
from data_models.writer_data_models import (
    PostWithContext,
    RejectorSampleResult,
    ScreenshotRejectorResult,
)
from note_writer.grok_client import GrokClient
from xai_sdk.chat import image
from fetcher.fetcher import (
    capture_screenshots_for_url,
    compute_url_hash,
    extract_base_domain,
)
from rejectors import find_best_draft, is_draft_on_track_for_submission
from utils.url_utils import URL_PATTERN

from utils.log_setup import get_logger

logger = get_logger("screenshot")


_MAX_SAMPLE_SCORE = 1.0


def _extract_urls(text: str) -> list[str]:
    return URL_PATTERN.findall(text)


def _encode_image_to_base64_data_uri(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        ext = path.suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        base64_str = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_str}"


def _select_best_screenshot(
    screenshot_dir: str, file_prefix: str, url_hash: str
) -> str | None:
    for suffix in ["delay", "idle", "load"]:
        path = os.path.join(screenshot_dir, f"{file_prefix}_{url_hash}_{suffix}.png")
        if os.path.exists(path):
            return path
    return None


async def _capture_note_screenshots(
    note_text: str,
    file_prefix: str,
    screenshot_dir: str,
    screenshot_config: ScreenshotRejector,
    browser,
) -> list[tuple[str, str | None]]:
    urls = _extract_urls(note_text)
    results: list[tuple[str, str | None]] = []
    for url in urls:
        domain = extract_base_domain(url)
        url_file_prefix = f"{file_prefix}_{domain}"
        await capture_screenshots_for_url(
            file_prefix=url_file_prefix,
            url=url,
            output_dir=Path(screenshot_dir),
            browser=browser,
            page_timeout=screenshot_config.page_timeout,
            post_load_delay=screenshot_config.post_load_delay,
            max_retries=screenshot_config.max_retries,
            max_screenshot_height=screenshot_config.max_screenshot_height,
        )
        url_hash = compute_url_hash(url)
        best = _select_best_screenshot(screenshot_dir, url_file_prefix, url_hash)
        results.append((url, best))
    return results


def _prepare_screenshot_rejector_prompt(
    rejector_config: ScreenshotRejector,
    post_link: str,
    note_text: str,
    screenshots: list[tuple[str, str]],
) -> list:
    prompt: list = [
        rejector_config.rejector_prompt.format(post_link=post_link, note_text=note_text)
    ]
    for url, screenshot_path in screenshots:
        blob = _encode_image_to_base64_data_uri(screenshot_path)
        prompt.append(f"URL: {url}")
        prompt.append(image(blob, detail="high"))
    return prompt


async def _query_screenshot_rejector(
    rejector_config: ScreenshotRejector,
    post_with_context: PostWithContext,
    note_text: str,
    screenshots: list[tuple[str, str]],
    xai_api_key: str,
    semaphore: asyncio.Semaphore,
    pass_threshold: float,
) -> ScreenshotRejectorResult:
    post_link = f"https://x.com/{post_with_context.post.username}/status/{post_with_context.post.post_id}"
    prompt = _prepare_screenshot_rejector_prompt(
        rejector_config, post_link, note_text, screenshots
    )

    async def _single_screenshot_rejector_query() -> RejectorSampleResult:
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
                sample_result.score = response["score"]
                sample_result.reasoning = response["reasoning"]
                return sample_result
            except Exception as e:
                logger.exception(
                    f"Error in _single_screenshot_rejector_query: {type(e).__name__}: {e}"
                )
                sample_result.error = f"{type(e).__name__}: {e}"
                return sample_result

    round1_results = await asyncio.gather(
        *[
            _single_screenshot_rejector_query()
            for _ in range(rejector_config.min_samples)
        ]
    )
    sample_results: list[RejectorSampleResult] = list(round1_results)

    round1_all_completed = all(
        r.error is None and r.score is not None for r in round1_results
    )
    round1_has_fail = any(
        r.score is not None and r.score < pass_threshold for r in round1_results
    )
    additional_samples = rejector_config.max_samples - rejector_config.min_samples
    best_case_mean = (
        sum(r.score for r in round1_results if r.score is not None)
        + additional_samples * _MAX_SAMPLE_SCORE
    ) / rejector_config.max_samples
    if (
        round1_all_completed
        and round1_has_fail
        and additional_samples > 0
        and best_case_mean >= pass_threshold
    ):
        round2_results = await asyncio.gather(
            *[_single_screenshot_rejector_query() for _ in range(additional_samples)]
        )
        sample_results.extend(round2_results)

    all_completed = all(r.error is None for r in sample_results)
    scores = [r.score for r in sample_results if r.score is not None]
    mean_score = (
        sum(scores) / len(scores)
        if ((len(scores) == len(sample_results)) and all_completed)
        else None
    )

    if (not all_completed) or (mean_score is None):
        status = "ERROR"
    elif mean_score >= pass_threshold:
        status = "PASS"
    else:
        status = "REJECT"

    screenshot_files = [os.path.basename(path) for _, path in screenshots]

    return ScreenshotRejectorResult(
        status=status,
        mean_score=mean_score,
        sample_results=list(sample_results),
        screenshot_files=screenshot_files,
    )


async def query_screenshot_rejector_for_post(
    arena_config: ArenaConfig,
    post_with_context: PostWithContext,
    writing_results_df: pd.DataFrame,
    feed_def: FeedDefinition,
    best_writing_result: pd.Series,
    best_row_idx: int,
    xai_api_key: str,
    semaphore: asyncio.Semaphore,
    df_lock: asyncio.Lock,
    screenshot_dir: str,
    work_started_at: int,
    browser,
) -> None:
    rejector_config = next(
        (
            r
            for r in arena_config.screenshot_rejectors
            if r.rejector_name == feed_def.screenshot_rejector
        ),
        None,
    )
    assert rejector_config is not None, (
        f"ScreenshotRejector {feed_def.screenshot_rejector} not found in arena_config.screenshot_rejectors"
    )

    try:
        file_prefix = f"{work_started_at}_{post_with_context.post.post_id}"
        all_screenshots = await _capture_note_screenshots(
            note_text=best_writing_result.grok_note,
            file_prefix=file_prefix,
            screenshot_dir=screenshot_dir,
            screenshot_config=rejector_config,
            browser=browser,
        )

        successful_screenshots = [
            (url, path) for url, path in all_screenshots if path is not None
        ]

        rejector_result = await _query_screenshot_rejector(
            rejector_config=rejector_config,
            post_with_context=post_with_context,
            note_text=best_writing_result.grok_note,
            screenshots=successful_screenshots,
            xai_api_key=xai_api_key,
            semaphore=semaphore,
            pass_threshold=feed_def.screenshot_rejector_pass_threshold,
        )

        async with df_lock:
            writing_results_df.at[best_row_idx, "screenshot_rejector"] = (
                rejector_config.rejector_name
            )
            writing_results_df.at[best_row_idx, "screenshot_rejection_status"] = (
                rejector_result.status
            )
            writing_results_df.at[best_row_idx, "screenshot_rejection_score"] = (
                rejector_result.mean_score
                if rejector_result.mean_score is not None
                else pd.NA
            )
            writing_results_df.at[best_row_idx, "screenshot_rejector_content"] = (
                json.dumps(
                    {
                        "sample_results": [
                            r.model_dump() for r in rejector_result.sample_results
                        ],
                        "screenshot_files": rejector_result.screenshot_files,
                    },
                    indent=4,
                )
            )

    except Exception as e:
        logger.exception(
            f"Error in query_screenshot_rejector_for_post: {type(e).__name__}: {e}"
        )
        async with df_lock:
            writing_results_df.at[best_row_idx, "screenshot_rejector"] = (
                feed_def.screenshot_rejector
            )
            writing_results_df.at[best_row_idx, "screenshot_rejection_status"] = "ERROR"
            writing_results_df.at[best_row_idx, "screenshot_rejector_content"] = (
                f"{type(e).__name__}: {e}"
            )


def any_note_needs_screenshot_rejection(
    arena_config: ArenaConfig,
    post_with_context: PostWithContext,
    writing_results_df: pd.DataFrame,
) -> bool:
    feed_def = arena_config.get_api_feed_def(post_with_context.api_feed)
    if not feed_def.screenshot_rejector:
        return False
    for writer in arena_config.grok_writers:
        result = find_best_draft(writer, writing_results_df)
        if result is None:
            continue
        best_draft, _ = result
        if is_draft_on_track_for_submission(
            best_draft, feed_def, post_with_context.post.post_id, arena_config
        ):
            return True
    return False
