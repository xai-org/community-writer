from __future__ import annotations

import asyncio
import os
import time
import traceback

from requests_oauthlib import OAuth1Session  # type: ignore
from sklearn.pipeline import Pipeline

from cnapi.delete_note import delete_note, NoteDeletionError
from data_models.submitted_note_cache import SubmittedNoteCache, fetch_api_statuses
from data_models.updating_config import UpdatingConfig

from utils.log_setup import get_logger

logger = get_logger("note_status")


def save_note_submission_history(
    submitted_note_cache: SubmittedNoteCache, note_submission_dir: str | None
) -> None:
    if not note_submission_dir:
        return
    submitted_note_cache.drop_old_rows()
    save_df = submitted_note_cache.get_save_data()
    if save_df.empty:
        return
    os.makedirs(note_submission_dir, exist_ok=True)
    ts = int(time.time())
    path = os.path.join(note_submission_dir, f"note_submission_history_{ts}.parquet")
    save_df.to_parquet(path, index=False)
    logger.info(f"Saved {len(save_df)} rows to {path}")


async def note_status_watcher(
    submitted_note_cache: SubmittedNoteCache,
    oauth_sessions: dict[str, OAuth1Session],
    shutdown_event: asyncio.Event,
    config_state: UpdatingConfig,
    note_submission_dir: str | None = None,
    refresh_interval: int = 60,
    lookback_days: int = 3,
    deletion_model: Pipeline | None = None,
    dry_run: bool = False,
) -> None:
    lookback_ms = lookback_days * 24 * 60 * 60 * 1000
    last_flush_time = time.time()
    flush_interval = 600

    while not shutdown_event.is_set():
        try:
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=refresh_interval,
            )

            break
        except asyncio.TimeoutError:
            pass

        try:
            t_start = time.time()
            min_created_at_ms = int(t_start * 1000) - lookback_ms
            api_statuses = await asyncio.to_thread(
                fetch_api_statuses,
                oauth_sessions,
                min_created_at_ms,
            )
            t_fetch = time.time()
            await submitted_note_cache.update_statuses(api_statuses)
            t_update = time.time()
            n_crh, n_crnh = await submitted_note_cache.size()
            logger.info(
                f"Refresh complete — "
                f"{len(api_statuses)} notes fetched in {(t_fetch - t_start) * 1000:.0f}ms, "
                f"cache updated in {(t_update - t_fetch) * 1000:.0f}ms, "
                f"total {(t_update - t_start) * 1000:.0f}ms, "
                f"cache has {n_crh} CRH, {n_crnh} CRNH (by first_status)"
            )

            arena_config = await config_state.get()
            if arena_config.model_deletion_policies:
                notes_to_delete = await submitted_note_cache.evaluate_deletion_policies(
                    arena_config.model_deletion_policies,
                    deletion_model,
                )
                deleted_any = False
                for dn in notes_to_delete:
                    logger.info(
                        f"{'DRY RUN, would delete' if dry_run else 'Deleting'} "
                        f"note_id={dn.note_id} submitter={dn.submitter} "
                        f"policies={','.join(dn.policy_names)} "
                        f"model_score={dn.model_score} "
                        f"total_ratings={dn.total_ratings} "
                        f"ratings={dn.nonzero_rating_counts}"
                    )
                    if dry_run:
                        continue
                    if dn.submitter not in oauth_sessions:
                        logger.error(
                            f"Delete failed for note {dn.note_id} — no OAuth session for '{dn.submitter}'"
                        )
                        continue
                    try:
                        success = await asyncio.to_thread(
                            delete_note, oauth_sessions[dn.submitter], dn.note_id
                        )
                        if success:
                            await submitted_note_cache.mark_deleted(
                                dn.note_id,
                                int(time.time() * 1000),
                                policy_names=dn.policy_names,
                                model_score=dn.model_score,
                            )
                            deleted_any = True
                    except NoteDeletionError as e:
                        if "is not found" in e.message:
                            logger.info(
                                f"Note {dn.note_id} already gone — marking as deleted"
                            )
                            await submitted_note_cache.mark_deleted(
                                dn.note_id,
                                int(time.time() * 1000),
                                policy_names=dn.policy_names,
                                model_score=dn.model_score,
                            )
                            deleted_any = True
                        else:
                            logger.exception(
                                f"Delete failed for note {dn.note_id}: {e}"
                            )
                if deleted_any:
                    save_note_submission_history(
                        submitted_note_cache, note_submission_dir
                    )
                    last_flush_time = time.time()

        except Exception as e:
            logger.exception(f"Error during refresh: {type(e).__name__}: {e}")

        if time.time() - last_flush_time >= flush_interval:
            save_note_submission_history(submitted_note_cache, note_submission_dir)
            last_flush_time = time.time()

    logger.info("Shutting down")
