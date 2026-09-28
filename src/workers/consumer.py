import asyncio
import os
import pickle
import random
import signal
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from typing import List

import pandas as pd
import psutil
from requests_oauthlib import OAuth1Session  # type: ignore

from data_models.arena_config import ArenaConfig
from data_models.environment_variables import KeySet
from data_models.feed import Feed, RevisionFeed, TimedFeed
from data_models.submitted_note_cache import SubmittedNoteCache, PriorExamples
from data_models.updating_config import UpdatingConfig
from data_models.writer_data_models import PostWithContext
from note_writer.table_printer import print_writing_results
from note_writer.submit_note_for_post import submit_note_for_post
from note_writer.write_note import (
    build_notable_post_rejected_result,
    build_failed_task_result,
)

from utils.log_setup import get_logger

logger = get_logger("consumer")


_TASK_RUNNER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "task_runner.py"
)


def _poll_feeds(feeds: list[Feed]) -> PostWithContext | None:
    for feed in feeds:
        try:
            return feed.queue.get_nowait()
        except asyncio.QueueEmpty:
            continue
    return None


def _effective_concurrency(
    post_concurrency: int,
    ramp_initial_fraction: float,
    ramp_duration_seconds: int,
    elapsed_seconds: float,
) -> int:
    if ramp_duration_seconds <= 0 or ramp_initial_fraction >= 1.0:
        return post_concurrency

    initial = post_concurrency * ramp_initial_fraction
    if elapsed_seconds >= ramp_duration_seconds:
        return post_concurrency

    progress = elapsed_seconds / ramp_duration_seconds
    effective = initial + (post_concurrency - initial) * progress
    return max(1, int(effective))


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, OSError) as e:
        logger.exception(f"Failed to kill process group for pid {pid}: {e}")


async def _run_writing_task(
    task_kwargs: dict,
    max_task_age_seconds: float,
    max_task_rss_bytes: int | None = None,
) -> tuple[pd.DataFrame, int]:
    fd_in, input_path = tempfile.mkstemp(suffix=".pkl")
    fd_out, output_path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd_out)

    try:
        with os.fdopen(fd_in, "wb") as f:
            pickle.dump(task_kwargs, f)

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            _TASK_RUNNER,
            "--input",
            input_path,
            "--output",
            output_path,
            start_new_session=True,
        )

        peak_rss_bytes = 0
        killed_for_memory = False

        async def _monitor_rss() -> None:
            nonlocal peak_rss_bytes, killed_for_memory
            while proc.returncode is None:
                try:
                    ps = psutil.Process(proc.pid)
                    rss = ps.memory_info().rss
                    for child in ps.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    peak_rss_bytes = max(peak_rss_bytes, rss)
                    if max_task_rss_bytes is not None and rss > max_task_rss_bytes:
                        _kill_process_group(proc.pid)
                        killed_for_memory = True
                        return
                except psutil.NoSuchProcess:
                    return
                await asyncio.sleep(5)

        monitor = asyncio.create_task(_monitor_rss())

        try:
            await asyncio.wait_for(proc.wait(), timeout=max_task_age_seconds)
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            _kill_process_group(proc.pid)
            await proc.wait()

            e.peak_rss_bytes = peak_rss_bytes
            raise
        finally:
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass

        if killed_for_memory:
            err = MemoryError(
                f"exceeded memory limit "
                f"(peak RSS {peak_rss_bytes / (1024**3):.1f} GB, "
                f"limit {max_task_rss_bytes / (1024**3):.1f} GB)"
            )
            err.peak_rss_bytes = peak_rss_bytes
            raise err

        if proc.returncode != 0:
            err = RuntimeError(f"task process exited with code {proc.returncode}")

            err.peak_rss_bytes = peak_rss_bytes
            raise err

        with open(output_path, "rb") as f:
            result_df = pickle.load(f)

        return result_df, peak_rss_bytes

    finally:
        for path in (input_path, output_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


async def consumer(
    api_feeds: list[Feed],
    oauth_sessions: dict[str, OAuth1Session],
    xai_api_key: str,
    config_state: UpdatingConfig,
    note_concurrency: int,
    dry_run: bool,
    shutdown_event: asyncio.Event,
    max_results_before_flush: int,
    output_dir: str | None,
    screenshot_dir: str | None,
    x_api_keys: dict[str, KeySet],
    post_concurrency: int,
    min_available_ram_gb: float | None = None,
    min_seconds_between_posts: float = 3.0,
    ramp_initial_fraction: float = 0.25,
    ramp_duration_seconds: int = 3600,
    submitted_note_cache: SubmittedNoteCache | None = None,
    drain_timeout_seconds: float = 300.0,
    max_task_age_seconds: float = 1200.0,
    max_task_rss_bytes: int | None = None,
    timed_feeds: list[TimedFeed] | None = None,
) -> None:
    if timed_feeds is None:
        timed_feeds = []
    revision_feed_names = {
        tf.name for tf in timed_feeds if isinstance(tf, RevisionFeed)
    }

    co_eval_accounts = list(x_api_keys.keys())
    co_eval_counter = 0

    if screenshot_dir:
        os.makedirs(screenshot_dir, exist_ok=True)

    results_list: List[pd.DataFrame] = []
    file_start_time = int(time.time())

    def _flush_results() -> None:
        nonlocal file_start_time
        if output_dir and results_list:
            end_time = int(time.time())
            combined_df = pd.concat(results_list, ignore_index=True)
            output_file = os.path.join(
                output_dir, f"{file_start_time}_to_{end_time}.parquet"
            )
            combined_df.to_parquet(output_file)
            logger.info(f"Saved {len(combined_df)} rows to {output_file}")
            file_start_time = end_time
        results_list.clear()

    def _record_failed_task(
        post: PostWithContext,
        arena_config: ArenaConfig,
        work_started_at: int,
        work_finished_at: int,
        failure_column: str,
        message: str,
        peak_rss_bytes: int | None = None,
    ) -> None:
        if output_dir is None:
            return
        result_df = build_failed_task_result(
            post_with_context=post,
            config_timestamp=arena_config.config_timestamp,
            work_started_at=work_started_at,
            work_finished_at=work_finished_at,
            failure_column=failure_column,
            message=message,
            multi_note_policy=arena_config.get_multi_note_policy(post.post.post_id),
            peak_rss_bytes=peak_rss_bytes,
        )
        results_list.append(result_df)
        if len(results_list) >= max_results_before_flush:
            _flush_results()

    column_specs = [
        ("post_id", 23),
        ("notable_post_prediction", 12, "np_pred"),
        ("api_feed", 18),
        ("timed_feed", 12),
        ("writer_name", 15),
        ("username", 18),
        ("post_text", 50),
        ("grok_note", 50),
        ("refusal", 20),
        ("error", 20),
        ("co_score", 12),
        ("rejection_status", 20, "rl_rejector"),
        ("recent_context_rejection_status", 20, "rc_rejector"),
        ("revision_rejection_status", 20, "rv_rejector"),
        ("screenshot_rejection_status", 20, "ss_rejector"),
        ("submission_eligible", 14, "post_eligible"),
        ("note_id", 23),
        ("note_submission_error", 30),
        ("submitter", 30),
    ]

    async def _process_post(post: PostWithContext, arena_config: ArenaConfig) -> None:
        nonlocal co_eval_counter
        work_started_at = int(time.time() * 1000)

        oauth_keyset = x_api_keys[
            co_eval_accounts[co_eval_counter % len(co_eval_accounts)]
        ]
        co_eval_counter += 1

        prior_examples: PriorExamples | None = None
        prior_post_notes: list = []
        if submitted_note_cache is not None:
            prior_examples = await submitted_note_cache.get_prior_examples(100, 100)
            prior_post_notes = await submitted_note_cache.get_prior_notes_for_post(
                post.post.post_id
            )

        task_kwargs = dict(
            post_with_context=post,
            arena_config=arena_config,
            xai_api_key=xai_api_key,
            oauth_keyset=oauth_keyset,
            note_concurrency=note_concurrency,
            screenshot_dir=screenshot_dir,
            work_started_at=work_started_at,
            prior_examples=prior_examples,
            prior_post_notes=prior_post_notes,
        )

        result_df = None
        try:
            result_df, peak_rss = await _run_writing_task(
                task_kwargs,
                max_task_age_seconds,
                max_task_rss_bytes,
            )

            work_finished_at = int(time.time() * 1000)
            elapsed_s = (work_finished_at - work_started_at) / 1000
            peak_rss_gb = peak_rss / (1024**3)

            result_df["work_started_at"] = work_started_at
            result_df["work_finished_at"] = work_finished_at
            result_df["peak_rss_bytes"] = peak_rss

            post_policy = arena_config.get_multi_note_policy(post.post.post_id)
            submission_order = list(arena_config.writer_priority)
            if post_policy == "once_per_writer":
                random.shuffle(submission_order)
            for writer_name in submission_order:
                submit_note_for_post(
                    writer_name=writer_name,
                    arena_config=arena_config,
                    post_with_context=post,
                    writing_results_df=result_df,
                    work_started_at=datetime.fromtimestamp(
                        work_started_at / 1000, timezone.utc
                    ),
                    submitted_note_cache=submitted_note_cache,
                    oauth_sessions=oauth_sessions,
                    dry_run=dry_run,
                )

            print_writing_results(result_df, column_specs)
            logger.info(
                f"Finished post {post.post.post_id} "
                f"({elapsed_s:.0f}s, peak RSS {peak_rss_gb:.1f} GB)"
            )

            from_revision_feed = (
                post.timed_feed is not None and post.timed_feed in revision_feed_names
            )
            if (
                timed_feeds
                and not from_revision_feed
                and result_df["note_id"].notna().any()
            ):
                now = time.time()
                for tf in timed_feeds:
                    if isinstance(tf, RevisionFeed):
                        tf.pending.schedule(
                            post.post.post_id, now + tf.latency_seconds, post
                        )

            results_list.append(result_df)
            result_df = None
            if len(results_list) >= max_results_before_flush:
                _flush_results()

        except asyncio.TimeoutError as e:
            work_finished_at = int(time.time() * 1000)
            elapsed_s = (work_finished_at - work_started_at) / 1000
            logger.info(
                f"Task timed out for post {post.post.post_id} "
                f"({elapsed_s:.0f}s, limit {max_task_age_seconds / 60:.0f} min)"
            )
            _record_failed_task(
                post,
                arena_config,
                work_started_at,
                work_finished_at,
                "task_timeout",
                f"timed out after {elapsed_s:.0f}s (limit {max_task_age_seconds:.0f}s)",
                peak_rss_bytes=getattr(e, "peak_rss_bytes", None),
            )
        except MemoryError as e:
            work_finished_at = int(time.time() * 1000)
            elapsed_s = (work_finished_at - work_started_at) / 1000
            logger.info(
                f"Task exceeded memory limit for post {post.post.post_id} "
                f"({elapsed_s:.0f}s): {e}"
            )
            _record_failed_task(
                post,
                arena_config,
                work_started_at,
                work_finished_at,
                "task_memory_error",
                str(e),
                peak_rss_bytes=getattr(e, "peak_rss_bytes", None),
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            work_finished_at = int(time.time() * 1000)
            elapsed_s = (work_finished_at - work_started_at) / 1000
            logger.exception(
                f"Error processing post {post.post.post_id} ({elapsed_s:.0f}s): {e}"
            )
            message = f"{type(e).__name__}: {e}"
            if result_df is None:
                _record_failed_task(
                    post,
                    arena_config,
                    work_started_at,
                    work_finished_at,
                    "task_exception",
                    message,
                    peak_rss_bytes=getattr(e, "peak_rss_bytes", None),
                )
            else:
                result_df["task_exception"] = message
                results_list.append(result_df)
                if len(results_list) >= max_results_before_flush:
                    _flush_results()

    in_flight: set[asyncio.Task] = set()
    last_dispatch_time: float = 0.0

    consumer_start_time = time.monotonic()
    prev_effective_concurrency = 0

    if ramp_initial_fraction < 1.0 and ramp_duration_seconds > 0:
        initial_concurrency = max(1, int(post_concurrency * ramp_initial_fraction))
        logger.info(
            f"Ramp-up enabled — starting at {initial_concurrency} "
            f"({ramp_initial_fraction:.0%} of {post_concurrency}), "
            f"ramping to {post_concurrency} over {ramp_duration_seconds}s"
        )

    while not shutdown_event.is_set():
        elapsed = time.monotonic() - consumer_start_time
        current_concurrency = _effective_concurrency(
            post_concurrency,
            ramp_initial_fraction,
            ramp_duration_seconds,
            elapsed,
        )
        if current_concurrency != prev_effective_concurrency:
            logger.info(
                f"Ramp-up — effective concurrency now {current_concurrency}/{post_concurrency}"
            )
            prev_effective_concurrency = current_concurrency

        if len(in_flight) >= current_concurrency:
            if in_flight:
                await asyncio.wait(
                    set(in_flight), timeout=1.0, return_when=asyncio.FIRST_COMPLETED
                )
            else:
                await asyncio.sleep(1.0)
            continue

        post = _poll_feeds(list(api_feeds) + list(timed_feeds))
        if post is None:
            if in_flight:
                await asyncio.wait(
                    set(in_flight), timeout=1.0, return_when=asyncio.FIRST_COMPLETED
                )
            else:
                await asyncio.sleep(1.0)
            continue

        arena_config = await config_state.get()

        feed_def = arena_config.get_api_feed_def(post.api_feed)
        if (
            post.notable_post_prediction is not None
            and feed_def.notable_post_threshold is not None
            and post.notable_post_prediction < feed_def.notable_post_threshold
        ):
            work_ts = int(time.time() * 1000)
            result_df = build_notable_post_rejected_result(
                post_with_context=post,
                config_timestamp=arena_config.config_timestamp,
                notable_post_threshold=feed_def.notable_post_threshold,
                work_started_at=work_ts,
                work_finished_at=work_ts,
                multi_note_policy=arena_config.get_multi_note_policy(post.post.post_id),
            )
            print_writing_results(result_df, column_specs)
            results_list.append(result_df)
            if len(results_list) >= max_results_before_flush:
                _flush_results()
            continue

        elapsed_since_last = time.monotonic() - last_dispatch_time
        if elapsed_since_last < min_seconds_between_posts:
            await asyncio.sleep(min_seconds_between_posts - elapsed_since_last)

        if min_available_ram_gb is not None:
            available_gb = psutil.virtual_memory().available / (1024**3)
            if available_gb < min_available_ram_gb:
                logger.info(
                    f"Waiting for RAM — "
                    f"{available_gb:.1f} GB available, need {min_available_ram_gb:.1f} GB "
                    f"({len(in_flight)} tasks in flight)"
                )
                await asyncio.sleep(5.0)
                continue

        last_dispatch_time = time.monotonic()
        feed_label = post.timed_feed if post.timed_feed else post.api_feed
        logger.info(
            f"Processing post {post.post.post_id} from {feed_label} queue ({len(in_flight) + 1}/{current_concurrency} active tasks)"
        )
        task = asyncio.create_task(_process_post(post, arena_config))
        in_flight.add(task)
        task.add_done_callback(in_flight.discard)

    if in_flight:
        logger.info(f"Draining {len(in_flight)} in-flight task(s)...")
        drain_deadline = time.monotonic() + drain_timeout_seconds
        while in_flight and time.monotonic() < drain_deadline:
            remaining = drain_deadline - time.monotonic()
            done, _ = await asyncio.wait(
                set(in_flight),
                timeout=min(remaining, 30.0),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done and in_flight:
                logger.info(
                    f"Drain — {len(in_flight)} task(s) still in flight, "
                    f"{drain_deadline - time.monotonic():.0f}s remaining"
                )
        if in_flight:
            logger.info(
                f"Drain timeout ({drain_timeout_seconds}s) reached — "
                f"cancelling {len(in_flight)} task(s)"
            )
            tasks_to_cancel = set(in_flight)
            for t in tasks_to_cancel:
                t.cancel()
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

    _flush_results()
    logger.info("Shutting down")
