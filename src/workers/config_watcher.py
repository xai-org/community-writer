import asyncio
import os
import tempfile
import traceback

from data_models.arena_config import ArenaConfig, load_and_validate_config
from data_models.updating_config import UpdatingConfig

from utils.log_setup import get_logger

logger = get_logger("cfg_watcher")


def find_latest_config_file(config_dir: str) -> tuple[str, int] | None:
    if not os.path.isdir(config_dir):
        return None

    latest_timestamp = -1
    latest_filepath = None

    for filename in os.listdir(config_dir):
        if not filename.endswith(".toml"):
            continue

        try:
            timestamp = int(filename[:-5])
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_filepath = os.path.join(config_dir, filename)
        except ValueError:
            continue

    if latest_filepath is None:
        return None
    return (latest_filepath, latest_timestamp)


def load_config_from_file(filepath: str) -> ArenaConfig:
    filename = os.path.basename(filepath)
    if not filename.endswith(".toml"):
        raise ValueError(f"Config file must have .toml extension: {filepath}")
    try:
        timestamp = int(filename[:-5])
    except ValueError:
        raise ValueError(
            f"Config filename must be a valid timestamp (e.g., 12345.toml): {filepath}"
        )

    return load_and_validate_config(filepath, config_timestamp=timestamp)


def update_active_symlink(config_dir: str, config_filename: str) -> None:
    symlink_path = os.path.join(config_dir, "active.toml")
    fd, tmp_path = tempfile.mkstemp(dir=config_dir, prefix=".active_", suffix=".tmp")
    os.close(fd)
    os.unlink(tmp_path)
    try:
        os.symlink(config_filename, tmp_path)
        os.replace(tmp_path, symlink_path)
        logger.info(f"Updated active.toml -> {config_filename}")
    except OSError as e:
        logger.exception(f"Failed to update active.toml symlink: {e}")
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def config_watcher(
    config_dir: str,
    config_state: UpdatingConfig,
    config_check_interval: int,
    shutdown_event: asyncio.Event,
) -> None:
    while not shutdown_event.is_set():
        try:
            current_config = await config_state.get()
            current_timestamp = current_config.config_timestamp

            latest_config = find_latest_config_file(config_dir)
            if latest_config is not None:
                config_filepath, new_timestamp = latest_config
                if new_timestamp > current_timestamp:
                    logger.info(f"New config detected (timestamp: {new_timestamp})")
                    logger.info(f"Loading config from {config_filepath}")

                    try:
                        new_config = load_config_from_file(config_filepath)

                        current_feed_names = {f.name for f in current_config.api_feeds}
                        new_feed_names = {f.name for f in new_config.api_feeds}
                        if current_feed_names != new_feed_names:
                            added = new_feed_names - current_feed_names
                            removed = current_feed_names - new_feed_names
                            logger.error(
                                f"ERROR - Rejecting config {new_timestamp}: "
                                f"feed names changed (added={sorted(added)}, removed={sorted(removed)}). "
                                f"Restart the service to change the set of feeds."
                            )
                        elif {
                            (rf.name, rf.latency_seconds)
                            for rf in current_config.revision_feeds
                        } != {
                            (rf.name, rf.latency_seconds)
                            for rf in new_config.revision_feeds
                        }:
                            logger.error(
                                f"ERROR - Rejecting config {new_timestamp}: "
                                f"revision_feeds changed. Restart the service to change revision feeds."
                            )
                        else:
                            await config_state.update(new_config)
                            update_active_symlink(
                                config_dir, os.path.basename(config_filepath)
                            )
                            logger.info(
                                f"Config updated successfully to timestamp {new_config.config_timestamp}"
                            )
                    except Exception as e:
                        logger.exception(f"Error loading new config: {e}")

        except Exception as e:
            logger.exception(f"Error checking for new config: {e}")

        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=config_check_interval)

            break
        except asyncio.TimeoutError:
            pass

    logger.info("Shutting down")
