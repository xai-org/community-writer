from __future__ import annotations

import asyncio
import time
import traceback
from collections import defaultdict
from typing import TYPE_CHECKING

from requests_oauthlib import OAuth1Session  # type: ignore

from cnapi.get_api_eligible_posts import FeedFetchResult, fetch_feed
from data_models.arena_config import (
    ALLOCATION_END,
    FEED_PURPOSE,
    FeedDefinition,
    post_digest,
)
from data_models.environment_variables import KeySet
from data_models.feed import Feed, RetryFeed, TimedFeed
from data_models.updating_config import UpdatingConfig
from data_models.writer_data_models import PostWithContext
from notable_post_model.predict import compute_notable_post_predictions

from utils.log_setup import get_logger

logger = get_logger("producer")

if TYPE_CHECKING:
    from data_models.submitted_note_cache import SubmittedNoteCache


def _filter_by_language(
    posts: list[PostWithContext],
    feed_def: FeedDefinition,
) -> list[PostWithContext]:
    if feed_def.included_languages:
        included = set(feed_def.included_languages)
        return [p for p in posts if p.post.lang in included]
    if feed_def.excluded_languages:
        excluded = set(feed_def.excluded_languages)
        return [p for p in posts if p.post.lang not in excluded]
    return posts


def _filter_by_notable_post_range(
    posts: list[PostWithContext],
    feed_def: FeedDefinition,
) -> list[PostWithContext]:
    if feed_def.notable_post_range is None:
        return posts
    floor, ceiling = feed_def.notable_post_range
    return [
        p
        for p in posts
        if p.notable_post_prediction is not None
        and floor <= p.notable_post_prediction < ceiling
    ]


def _find_api_feed_for_post(
    post: PostWithContext,
    feed_size_priority: list[str],
    seen_in_sizes: dict[str, set[int]],
    feed_defs: dict[str, FeedDefinition],
    feeds_by_size: dict[str, list[Feed]],
) -> str | None:
    post_id = post.post.post_id
    for feed_size in feed_size_priority:
        if post_id not in seen_in_sizes.get(feed_size, set()):
            continue
        for feed in feeds_by_size.get(feed_size, []):
            fd = feed_defs[feed.name]
            if _filter_by_language([post], fd) and _filter_by_notable_post_range(
                [post], fd
            ):
                return feed.name
    return None


async def producer(
    api_feeds: list[Feed],
    x_api_keys: dict[str, KeySet],
    config_state: UpdatingConfig,
    producer_interval: int,
    shutdown_event: asyncio.Event,
    timed_feeds: list[TimedFeed] | None = None,
    submitted_note_cache: SubmittedNoteCache | None = None,
    notable_post_model=None,
    seen_in_sizes: dict[str, set[int]] | None = None,
) -> None:
    if timed_feeds is None:
        timed_feeds = []
    if seen_in_sizes is None:
        seen_in_sizes = {}
    all_accounts = list(x_api_keys.keys())
    cycle_counter = 0

    initial_cutoffs: dict[str, int] = {}

    if timed_feeds:
        max_timed_feed_buffer_ms = (
            max(tf.latency_seconds for tf in timed_feeds) + 3600
        ) * 1000
    else:
        max_timed_feed_buffer_ms = 0

    while not shutdown_event.is_set():
        cycle_start = time.monotonic()
        try:
            current_account = all_accounts[cycle_counter % len(all_accounts)]
            keyset = x_api_keys[current_account]

            config = config_state.config
            feed_defs: dict[str, FeedDefinition] = {f.name: f for f in config.api_feeds}

            feeds_by_size: dict[str, list[Feed]] = defaultdict(list)
            for feed in api_feeds:
                feeds_by_size[feed_defs[feed.name].feed_size].append(feed)

            for feed_size in feeds_by_size:
                if feed_size not in seen_in_sizes:
                    seen_in_sizes[feed_size] = set()

            now_millis_for_cutoff = int(time.time() * 1000)
            group_cutoffs: dict[str, int | None] = {}
            for feed_size in feeds_by_size:
                if feed_size not in initial_cutoffs:
                    group_cutoffs[feed_size] = None
                else:
                    group_cutoffs[feed_size] = max(
                        initial_cutoffs[feed_size],
                        now_millis_for_cutoff - max_timed_feed_buffer_ms,
                    )

            feed_sizes = list(feeds_by_size.keys())

            async def _fetch_group(feed_size: str) -> FeedFetchResult | Exception:
                oauth = OAuth1Session(
                    keyset.x_api_key,
                    client_secret=keyset.x_api_secret_key,
                    resource_owner_key=keyset.x_access_token,
                    resource_owner_secret=keyset.x_access_token_secret,
                )
                try:
                    return await asyncio.to_thread(
                        fetch_feed,
                        oauth=oauth,
                        feed_name=feed_size,
                        post_selection=feed_size,
                        cutoff=group_cutoffs[feed_size],
                    )
                except Exception as e:
                    return e

            group_results: list[FeedFetchResult | Exception] = await asyncio.gather(
                *[_fetch_group(fs) for fs in feed_sizes]
            )

            now = time.time()
            now_millis = int(now * 1000)

            logger.info(
                f"Fetch cycle {cycle_counter} "
                f"(account: {current_account}, timestamp: {now_millis})"
            )

            fetched_posts_by_id: dict[int, PostWithContext] = {}
            retry_feeds = [tf for tf in timed_feeds if isinstance(tf, RetryFeed)]

            for feed_size, result in zip(feed_sizes, group_results):
                group_feeds = feeds_by_size[feed_size]

                if isinstance(result, Exception):
                    feed_names = ", ".join(f.name for f in group_feeds)
                    logger.error(
                        f"  {feed_size} ({feed_names}): ERROR - {result}\n"
                        f"    {traceback.format_exception_only(type(result), result)[-1].strip()}"
                    )
                    continue

                if feed_size not in initial_cutoffs:
                    initial_cutoffs[feed_size] = result.oldest_timestamp_ms

                group_parts = [
                    f"  {feed_size}: {len(result.posts)} posts",
                    f"{result.pages_fetched} pages",
                    f"{result.elapsed_seconds:.1f}s",
                    f"cutoff={group_cutoffs[feed_size]}",
                ]
                if result.error_count > 0:
                    group_parts.append(f"errors={result.error_count}")
                logger.info(", ".join(group_parts))

                for p in result.posts:
                    p.enqueued_at = now_millis

                if notable_post_model is not None and submitted_note_cache is not None:
                    try:
                        compute_notable_post_predictions(
                            notable_post_model,
                            result.posts,
                            submitted_note_cache.df,
                        )
                    except Exception as e:
                        logger.exception(f"  Notable-post prediction error: {e}")

                for p in result.posts:
                    fetched_posts_by_id[p.post.post_id] = p

                for feed in group_feeds:
                    feed_def = feed_defs[feed.name]

                    lang_posts = _filter_by_language(result.posts, feed_def)
                    range_posts = _filter_by_notable_post_range(lang_posts, feed_def)
                    feed_posts = [
                        p.model_copy(update={"api_feed": feed.name})
                        for p in range_posts
                    ]

                    enqueue_posts = feed_posts
                    ranges_filtered = 0
                    if feed_def.enabled_ranges:
                        pre_filter_count = len(enqueue_posts)
                        enqueue_posts = [
                            p
                            for p in enqueue_posts
                            if any(
                                start
                                <= post_digest(
                                    p.post.post_id, FEED_PURPOSE, ALLOCATION_END
                                )
                                < end
                                for start, end in feed_def.enabled_ranges
                            )
                        ]
                        ranges_filtered = pre_filter_count - len(enqueue_posts)

                    added = 0
                    for post in enqueue_posts:
                        if post.post.post_id not in seen_in_sizes[feed_size]:
                            await feed.queue.put(post)
                            seen_in_sizes[feed_size].add(post.post.post_id)
                            added += 1

                            post_created_at = post.post.created_at.timestamp()
                            for rf in retry_feeds:
                                post_age_s = now - post_created_at
                                if post_age_s <= rf.max_post_age_seconds:
                                    rf.pending.schedule(
                                        post.post.post_id,
                                        now + rf.latency_seconds,
                                        post,
                                    )
                                else:
                                    rf.pending.remove(post.post.post_id)

                    posts_part = f"    {feed.name}: {len(feed_posts)} posts"
                    if feed_def.notable_post_threshold is not None:
                        low_score = sum(
                            1
                            for p in feed_posts
                            if p.notable_post_prediction is not None
                            and p.notable_post_prediction
                            < feed_def.notable_post_threshold
                        )
                        posts_part += f" ({low_score} low score)"
                    parts = [posts_part, f"enqueued={added}"]
                    if ranges_filtered > 0:
                        parts.append(f"ranges_filtered={ranges_filtered}")
                    logger.info(", ".join(parts))

            for timed_feed in timed_feeds:
                due_items = timed_feed.pending.drain_due(now)
                drained = 0
                skipped_note_exists = 0
                skipped_not_in_feed = 0
                for post_id, original_post in due_items:
                    if isinstance(timed_feed, RetryFeed):
                        if submitted_note_cache is not None:
                            cache_df = submitted_note_cache.df
                            has_note = (
                                (cache_df["post_id"] == post_id)
                                & cache_df["note_id"].notna()
                            ).any()
                            if has_note:
                                skipped_note_exists += 1
                                continue

                    fresh_post = fetched_posts_by_id.get(post_id)
                    if fresh_post is None:
                        skipped_not_in_feed += 1
                        continue

                    if (
                        notable_post_model is not None
                        and submitted_note_cache is not None
                        and fresh_post.notable_post_prediction is None
                    ):
                        try:
                            compute_notable_post_predictions(
                                notable_post_model,
                                [fresh_post],
                                submitted_note_cache.df,
                            )
                        except Exception as e:
                            logger.exception(
                                f"  Notable-post prediction error for timed feed post {post_id}: {e}"
                            )

                    api_feed_name = _find_api_feed_for_post(
                        fresh_post,
                        config.feed_size_priority,
                        seen_in_sizes,
                        feed_defs,
                        feeds_by_size,
                    )
                    if api_feed_name is None:
                        logger.error(
                            f"Warning - post {post_id} not routable "
                            f"for {timed_feed.name}, skipping"
                        )
                        continue
                    work_item = fresh_post.model_copy(
                        update={
                            "api_feed": api_feed_name,
                            "timed_feed": timed_feed.name,
                        }
                    )
                    work_item.enqueued_at = now_millis
                    await timed_feed.queue.put(work_item)
                    drained += 1
                if (
                    drained > 0
                    or timed_feed.pending
                    or skipped_note_exists
                    or skipped_not_in_feed
                ):
                    parts = [
                        f"{timed_feed.name}: drained={drained}, pending={len(timed_feed.pending)}"
                    ]
                    if skipped_note_exists:
                        parts.append(f"skipped_note_exists={skipped_note_exists}")
                    if skipped_not_in_feed:
                        parts.append(f"skipped_not_in_feed={skipped_not_in_feed}")
                    logger.info(", ".join(parts))

            all_feeds = list(api_feeds) + list(timed_feeds)
            queue_sizes = ", ".join(
                f"{feed.name}: {feed.queue.qsize()}" for feed in all_feeds
            )
            logger.info(f"Queue sizes: {{ {queue_sizes} }}")
            cycle_counter += 1

        except Exception as e:
            logger.exception(f"Error in fetch cycle: {e}")

        elapsed = time.monotonic() - cycle_start
        sleep_time = max(0, producer_interval - elapsed)
        logger.info(f"Cycle took {elapsed:.1f}s, sleeping for {sleep_time:.1f}s")
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=sleep_time)

            break
        except asyncio.TimeoutError:
            pass

    logger.info("Shutting down")
