import glob
import os
import re
import time

import pandas as pd
import pyarrow.parquet as pq

from utils.log_setup import get_logger

logger = get_logger("post_ids")

_TIMESTAMP_RE = re.compile(r"(\d+)_to_(\d+)\.parquet$")


def _filter_parquet_files_by_age(
    parquet_files: list[str], max_age_hours: float | None
) -> list[str]:
    if max_age_hours is None:
        logger.info(
            f"Parquet age filter disabled: keeping all {len(parquet_files)} file(s)"
        )
        return parquet_files

    cutoff = time.time() - max_age_hours * 3600
    kept: list[str] = []
    skipped = 0
    for fp in parquet_files:
        m = _TIMESTAMP_RE.search(os.path.basename(fp))
        if m is None:
            kept.append(fp)
            continue
        end_ts = int(m.group(2))
        if end_ts >= cutoff:
            kept.append(fp)
        else:
            skipped += 1

    logger.info(
        f"Parquet age filter ({max_age_hours}h): "
        f"kept {len(kept)}, skipped {skipped} (of {len(parquet_files)} total)"
    )
    return kept


def load_processed_post_ids(
    output_dir: str,
    feed_name_to_size: dict[str, str],
    max_age_hours: float | None = 72.0,
) -> dict[str, set[int]]:
    feed_sizes = set(feed_name_to_size.values())
    result: dict[str, set[int]] = {size: set() for size in feed_sizes}
    parquet_files = glob.glob(os.path.join(output_dir, "*.parquet"))
    parquet_files = _filter_parquet_files_by_age(parquet_files, max_age_hours)
    total_files = len(parquet_files)
    loaded = 0
    total_rows = 0
    log_interval = max(1, min(50, total_files // 10)) if total_files else 1

    logger.info(f"Scanning {total_files} parquet file(s) for processed post IDs...")

    for i, filepath in enumerate(parquet_files, 1):
        try:
            available = pq.read_schema(filepath).names
            feed_col: str | None = "api_feed" if "api_feed" in available else None
            cols_to_read = ["post_id"] + ([feed_col] if feed_col else [])
            if "post_id" not in available:
                continue
            df = pd.read_parquet(filepath, columns=cols_to_read)

            valid = df[df["post_id"].notna()]
            if "api_feed" in valid.columns:
                for _, row in valid.iterrows():
                    feed_name = row["api_feed"]
                    feed_size = feed_name_to_size.get(
                        feed_name if pd.notna(feed_name) else ""
                    )
                    if feed_size is not None:
                        result[feed_size].add(int(row["post_id"]))
            else:
                all_pids = {int(pid) for pid in valid["post_id"]}
                for size in feed_sizes:
                    result[size].update(all_pids)

            total_rows += len(valid)
            loaded += 1
        except Exception as e:
            logger.warning(f"Warning: Could not load columns from {filepath}: {e}")

        if i % log_interval == 0 or i == total_files:
            pct = i * 100 // total_files
            logger.info(
                f"  Progress: {i}/{total_files} files scanned ({pct}%), "
                f"{loaded} loaded, {total_rows} rows"
            )

    totals = ", ".join(f"{len(result[size])} {size}" for size in sorted(feed_sizes))
    logger.info(f"Processed post IDs by size: {totals}")
    return result
