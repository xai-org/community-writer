import argparse
import asyncio
import os
import signal

import dotenv
import pydantic
from requests_oauthlib import OAuth1Session  # type: ignore


from deletion_model.predict import load_model as load_deletion_model
from notable_post_model.predict import load_notable_post_model
from data_models.arena_config import ArenaConfig
from data_models.environment_variables import KeySet
from data_models.feed import Feed, RevisionFeed, RetryFeed, TimedFeed
from data_models.updating_config import UpdatingConfig
from workers.producer import producer
from workers.consumer import consumer
from workers.config_watcher import (
    config_watcher,
    find_latest_config_file,
    load_config_from_file,
    update_active_symlink,
)
from workers.note_status_watcher import (
    note_status_watcher,
    save_note_submission_history,
)
from data_models.submitted_note_cache import SubmittedNoteCache
import browser.session as browser_session
from utils.init_helpers import (
    load_environment_variables,
    create_oauth_sessions,
)
from utils.load_processed_post_ids import load_processed_post_ids

from utils.log_setup import configure_logging, get_logger

logger = get_logger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run note-writing bot as a continuous workflow."
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Forcibly block all note submissions and deletions regardless of config; drafts are "
        "still written, scored and logged, with the outcome recorded in the dry_run column",
    )
    _ = parser.add_argument(
        "--post-concurrency",
        type=int,
        default=4,
        help="Maximum number of posts to process concurrently via the process pool",
    )
    _ = parser.add_argument(
        "--note-concurrency",
        type=int,
        default=2,
        help="Number of concurrent note writing tasks per post",
    )
    _ = parser.add_argument(
        "--producer-interval",
        type=int,
        default=60,
        help="Seconds between producer runs to fetch new eligible posts (default: 60)",
    )
    _ = parser.add_argument(
        "--config-check-interval",
        type=int,
        default=60,
        help="Seconds between checks for new writer config files (default: 60)",
    )
    _ = parser.add_argument(
        "--max-results-before-flush",
        type=int,
        default=10,
        help="Maximum number of results to accumulate before flushing to disk (default: 10)",
    )
    _ = parser.add_argument(
        "--min-available-ram-gb",
        type=float,
        default=40.0,
        help="Minimum available RAM in GB before starting work on a new post. "
        "The consumer will wait until this much RAM is available before dispatching a new post. "
        "(default: 40.0)",
    )
    _ = parser.add_argument(
        "--min-seconds-between-posts",
        type=float,
        default=3.0,
        help="Minimum number of seconds to wait between dispatching new posts to the process pool (default: 3.0)",
    )
    _ = parser.add_argument(
        "--ramp-initial-fraction",
        type=float,
        default=0.25,
        help="Fraction of --post-concurrency to start with during ramp-up (0.0 to 1.0, default: 0.25)",
    )
    _ = parser.add_argument(
        "--ramp-duration-seconds",
        type=int,
        default=3600,
        help="Seconds over which to linearly ramp from initial to full post_concurrency (default: 3600)",
    )
    _ = parser.add_argument(
        "--max-task-age-seconds",
        type=float,
        default=1200.0,
        help="Maximum seconds a single task may run before its process group is killed via SIGKILL. "
        "Prevents stuck or runaway tasks from holding resources indefinitely (default: 1200 = 20 minutes).",
    )
    _ = parser.add_argument(
        "--max-task-rss-gb",
        type=float,
        default=None,
        help="Maximum RSS in GB that a single task's process tree may use before being killed. "
        "Monitors the full process tree (including Chromium sub-processes). "
        "None (default) disables the per-task memory limit.",
    )
    _ = parser.add_argument(
        "--drain-timeout-seconds",
        type=float,
        default=300.0,
        help="Maximum seconds to wait for in-flight tasks to complete during graceful shutdown (default: 300).",
    )
    _ = parser.add_argument(
        "--status-refresh-interval",
        type=int,
        default=60,
        help="Seconds between note status refresh cycles (default: 60).",
    )
    _ = parser.add_argument(
        "--status-lookback-days",
        type=int,
        default=3,
        help="Only fetch notes from the last N days during status refreshes (default: 3).",
    )
    _ = parser.add_argument(
        "--parquet-lookback-hours",
        type=float,
        default=72.0,
        help="Skip output parquet files whose filename timestamp is older than this many hours (default: 72 = 3 days).",
    )
    _ = parser.add_argument(
        "--notable-post-model",
        type=str,
        default=None,
        help="Path to a trained notable post model directory (containing "
        "config.json, model.pt, preprocessing.joblib). If set, the "
        "producer computes notable-post predictions for each post and "
        "the consumer applies notable-post thresholds before writing.",
    )
    _ = parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Override the sentence-transformers model used for text "
        "embeddings (default: use the model specified in the "
        "notable post model's config.json).",
    )
    _ = parser.add_argument(
        "--deletion-model",
        type=str,
        default=None,
        help="Path to a trained deletion model (joblib file). Required if the config "
        "contains model_deletion_policies.",
    )
    args = parser.parse_args()

    if not (0.0 < args.ramp_initial_fraction <= 1.0):
        parser.error("--ramp-initial-fraction must be in the range (0.0, 1.0]")
    if args.ramp_duration_seconds < 0:
        parser.error("--ramp-duration-seconds must be >= 0")
    if args.ramp_initial_fraction < 1.0 and args.ramp_duration_seconds == 0:
        parser.error(
            "--ramp-initial-fraction < 1.0 requires --ramp-duration-seconds > 0"
        )

    dotenv.load_dotenv(override=True)

    path_vars = {
        "CONFIG_DIR": "config_dir",
        "FEED_LOGS_DIR": "output_dir",
        "SUBMISSION_LOGS_DIR": "note_submission_dir",
        "SCREENSHOT_DIR": "screenshot_dir",
    }
    missing = [var for var in path_vars if not os.environ.get(var)]
    if missing:
        parser.error(f"missing required environment variables: {', '.join(missing)}")
    for var, attr in path_vars.items():
        setattr(args, attr, os.environ[var])
    logger.info("Loaded paths from the environment:")
    logger.info(f"  CONFIG_DIR          = {args.config_dir}")
    logger.info(f"  FEED_LOGS_DIR       = {args.output_dir}")
    logger.info(f"  SUBMISSION_LOGS_DIR = {args.note_submission_dir}")
    logger.info(f"  SCREENSHOT_DIR      = {args.screenshot_dir}")

    latest_config = find_latest_config_file(args.config_dir)
    if latest_config is None:
        raise ValueError(f"No valid config files found in {args.config_dir}")

    config_filepath, config_timestamp = latest_config
    arena_config = load_config_from_file(config_filepath)
    update_active_symlink(args.config_dir, os.path.basename(config_filepath))
    logger.info(
        f"Loaded initial config from {config_filepath} (timestamp: {config_timestamp})"
    )

    args.initial_arena_config = arena_config

    return args


async def main(
    oauth_sessions: dict[str, OAuth1Session],
    xai_api_key: str,
    x_api_keys: dict[str, KeySet],
    initial_arena_config: ArenaConfig,
    config_dir: str,
    config_check_interval: int,
    post_concurrency: int,
    note_concurrency: int,
    producer_interval: int,
    output_dir: str | None,
    note_submission_dir: str | None = None,
    screenshot_dir: str | None = None,
    dry_run: bool = False,
    max_results_before_flush: int = 10,
    min_available_ram_gb: float | None = None,
    min_seconds_between_posts: float = 3.0,
    ramp_initial_fraction: float = 0.25,
    ramp_duration_seconds: int = 3600,
    drain_timeout_seconds: float = 300.0,
    max_task_age_seconds: float = 1200.0,
    max_task_rss_bytes: int | None = None,
    status_refresh_interval: int = 60,
    status_lookback_days: int = 3,
    parquet_lookback_hours: float | None = 72.0,
    notable_post_model=None,
    deletion_model=None,
):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    submitted_note_cache = SubmittedNoteCache()
    submitted_note_cache.initialize(note_submission_dir=note_submission_dir)
    cache_df = submitted_note_cache.df
    n_crh = int((cache_df["first_status"] == "CURRENTLY_RATED_HELPFUL").sum())
    n_crnh = int((cache_df["first_status"] == "CURRENTLY_RATED_NOT_HELPFUL").sum())
    n_deleted = int(cache_df["deleted_at_millis"].notna().sum())
    logger.info(
        f"Note status cache initialized: {len(cache_df)} notes, {n_crh} CRH, {n_crnh} CRNH (by first_status), {n_deleted} deleted"
    )

    feed_name_to_size = {f.name: f.feed_size for f in initial_arena_config.api_feeds}
    if output_dir:
        seen_in_sizes = load_processed_post_ids(
            output_dir, feed_name_to_size, max_age_hours=parquet_lookback_hours
        )
    else:
        seen_in_sizes = {size: set() for size in set(feed_name_to_size.values())}

    api_feeds: list[Feed] = [
        Feed(name=feed_def.name) for feed_def in initial_arena_config.api_feeds
    ]

    timed_feeds: list[TimedFeed] = [
        RevisionFeed(name=rf.name, latency_seconds=rf.latency_seconds)
        for rf in initial_arena_config.revision_feeds
    ] + [
        RetryFeed(
            name=rf.name,
            latency_seconds=rf.latency_seconds,
            max_post_age_seconds=rf.max_post_age_seconds,
        )
        for rf in initial_arena_config.retry_feeds
    ]

    shutdown_event = asyncio.Event()
    config_state = UpdatingConfig(config=initial_arena_config, config_dir=config_dir)

    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("\nReceived shutdown signal, finishing current work...")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    producer_task = asyncio.create_task(
        producer(
            api_feeds=api_feeds,
            x_api_keys=x_api_keys,
            config_state=config_state,
            producer_interval=producer_interval,
            shutdown_event=shutdown_event,
            timed_feeds=timed_feeds,
            submitted_note_cache=submitted_note_cache,
            notable_post_model=notable_post_model,
            seen_in_sizes=seen_in_sizes,
        )
    )

    config_watcher_task = asyncio.create_task(
        config_watcher(
            config_dir=config_dir,
            config_state=config_state,
            config_check_interval=config_check_interval,
            shutdown_event=shutdown_event,
        )
    )

    note_status_watcher_task = asyncio.create_task(
        note_status_watcher(
            submitted_note_cache=submitted_note_cache,
            oauth_sessions=oauth_sessions,
            shutdown_event=shutdown_event,
            config_state=config_state,
            note_submission_dir=note_submission_dir,
            refresh_interval=status_refresh_interval,
            lookback_days=status_lookback_days,
            deletion_model=deletion_model,
            dry_run=dry_run,
        )
    )

    consumer_task = asyncio.create_task(
        consumer(
            api_feeds=api_feeds,
            oauth_sessions=oauth_sessions,
            xai_api_key=xai_api_key,
            config_state=config_state,
            note_concurrency=note_concurrency,
            dry_run=dry_run,
            shutdown_event=shutdown_event,
            max_results_before_flush=max_results_before_flush,
            output_dir=output_dir,
            screenshot_dir=screenshot_dir,
            x_api_keys=x_api_keys,
            post_concurrency=post_concurrency,
            min_available_ram_gb=min_available_ram_gb,
            min_seconds_between_posts=min_seconds_between_posts,
            ramp_initial_fraction=ramp_initial_fraction,
            ramp_duration_seconds=ramp_duration_seconds,
            submitted_note_cache=submitted_note_cache,
            drain_timeout_seconds=drain_timeout_seconds,
            max_task_age_seconds=max_task_age_seconds,
            max_task_rss_bytes=max_task_rss_bytes,
            timed_feeds=timed_feeds,
        )
    )

    feed_order = " > ".join(feed.name for feed in api_feeds)
    logger.info(
        f"Started 1 producer, 1 config watcher, 1 note status watcher, and 1 consumer (post_concurrency={post_concurrency})"
    )
    logger.info(f"Producer will fetch new posts every {producer_interval} seconds")
    logger.info(
        f"Config watcher will check for new configs every {config_check_interval} seconds"
    )
    logger.info(
        f"Note status watcher will refresh every {status_refresh_interval} seconds (lookback: {status_lookback_days} days)"
    )
    logger.info(
        f"Using LIFO queue architecture with API feeds (priority order): {feed_order}"
    )
    revision_feeds = [tf for tf in timed_feeds if isinstance(tf, RevisionFeed)]
    retry_feeds = [tf for tf in timed_feeds if isinstance(tf, RetryFeed)]
    if revision_feeds:
        revision_info = ", ".join(
            f"{tf.name} ({tf.latency_seconds}s)" for tf in revision_feeds
        )
        logger.info(f"Revision feeds: {revision_info}")
    if retry_feeds:
        retry_info = ", ".join(
            f"{tf.name} ({tf.latency_seconds}s, max_age={tf.max_post_age_seconds}s)"
            for tf in retry_feeds
        )
        logger.info(f"Retry feeds: {retry_info}")
    logger.info("Press Ctrl+C to gracefully shutdown\n")

    try:
        await asyncio.gather(
            producer_task, config_watcher_task, note_status_watcher_task, consumer_task
        )
    except asyncio.CancelledError:
        pass

    save_note_submission_history(submitted_note_cache, note_submission_dir)

    logger.info("Done.")


if __name__ == "__main__":
    configure_logging()

    dotenv.load_dotenv(override=True)

    args = parse_args()

    browser_session.preflight()

    logger.info("Arguments:")
    for key, value in args.__dict__.items():
        if isinstance(value, pydantic.BaseModel):
            logger.info(f"  {key:<20}\n{value.model_dump_json(indent=4)}")
        else:
            logger.info(f"  {key:<20} {value}")
    env_vars = load_environment_variables(
        [config.account_name for config in args.initial_arena_config.submission_configs]
    )

    notable_post_model = None
    if args.notable_post_model:
        embedding_override = getattr(args, "embedding_model", None)
        notable_post_model = load_notable_post_model(
            args.notable_post_model,
            embedding_model=embedding_override,
        )

        timed_latencies = [
            rf.latency_seconds for rf in args.initial_arena_config.revision_feeds
        ] + [rf.latency_seconds for rf in args.initial_arena_config.retry_feeds]
        if timed_latencies:
            cache_ttl = max(timed_latencies) + 3 * 3600
        else:
            cache_ttl = 5 * 3600
        notable_post_model.embedding_cache.ttl_seconds = cache_ttl
        logger.info(
            f"\nLoaded notable post model from {args.notable_post_model} "
            f"(embedding cache TTL: {cache_ttl / 3600:.1f}h)"
        )

    deletion_model = None
    if args.deletion_model:
        deletion_model = load_deletion_model(args.deletion_model)
        logger.info(f"\nLoaded deletion model from {args.deletion_model}")
    elif args.initial_arena_config.model_deletion_policies:
        raise ValueError(
            "Config contains model_deletion_policies but --deletion-model was not provided. "
            "Either supply --deletion-model or remove model_deletion_policies from the config."
        )

    oauth_sessions = create_oauth_sessions(env_vars.x_api_keys)

    asyncio.run(
        main(
            oauth_sessions=oauth_sessions,
            xai_api_key=env_vars.xai_api_key,
            x_api_keys=env_vars.x_api_keys,
            initial_arena_config=args.initial_arena_config,
            config_dir=args.config_dir,
            config_check_interval=args.config_check_interval,
            post_concurrency=args.post_concurrency,
            note_concurrency=args.note_concurrency,
            producer_interval=args.producer_interval,
            output_dir=args.output_dir,
            note_submission_dir=args.note_submission_dir,
            screenshot_dir=args.screenshot_dir,
            dry_run=args.dry_run,
            max_results_before_flush=args.max_results_before_flush,
            min_available_ram_gb=args.min_available_ram_gb,
            min_seconds_between_posts=args.min_seconds_between_posts,
            ramp_initial_fraction=args.ramp_initial_fraction,
            ramp_duration_seconds=args.ramp_duration_seconds,
            drain_timeout_seconds=args.drain_timeout_seconds,
            max_task_age_seconds=args.max_task_age_seconds,
            max_task_rss_bytes=int(args.max_task_rss_gb * (1024**3))
            if args.max_task_rss_gb
            else None,
            status_refresh_interval=args.status_refresh_interval,
            status_lookback_days=args.status_lookback_days,
            parquet_lookback_hours=args.parquet_lookback_hours,
            notable_post_model=notable_post_model,
            deletion_model=deletion_model,
        )
    )
