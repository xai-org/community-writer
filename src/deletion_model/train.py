from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline

from deletion_model.features import (
    BUCKETS,
    HELPFULNESS_LEVELS,
    HELPFULNESS_MAP,
    PANELS,
    RATING_COLUMNS,
    SNAPSHOT_THRESHOLDS,
    TAG_COLUMN_MAP,
    assign_bucket,
    compute_model_features,
    load_api_writer_ids,
)
from deletion_model.model import build_model


_AGE_WINDOW_MIN_DAYS = 3
_AGE_WINDOW_MAX_DAYS = 180

_NOTE_LOAD_COLUMNS = [
    "noteId",
    "noteAuthorParticipantId",
    "createdAtMillis",
    "currentStatus",
]

_RATER_LOAD_COLUMNS = ["raterParticipantId", "expansionRaterFactor1"]

_RATING_LOAD_COLUMNS = [
    "noteId",
    "raterParticipantId",
    "createdAtMillis",
    "helpfulnessLevel",
] + list(TAG_COLUMN_MAP.keys())


def load_and_prepare_data(
    status_path: str,
    rater_path: str,
    ratings_path: str,
    enrollment_path: str,
    writer_account_ids: set[str],
    excluded_account_ids: set[str],
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    now_ms = int(time.time() * 1000)
    day_ms = 86_400_000
    cutoff_recent_ms = now_ms - _AGE_WINDOW_MIN_DAYS * day_ms
    cutoff_oldest_ms = now_ms - _AGE_WINDOW_MAX_DAYS * day_ms

    print(
        f"  Loading note status history "
        f"(window {_AGE_WINDOW_MIN_DAYS}..{_AGE_WINDOW_MAX_DAYS}d, "
        f"{len(_NOTE_LOAD_COLUMNS)} cols, predicate pushdown)...",
        flush=True,
    )

    notes = pd.read_parquet(
        status_path,
        columns=_NOTE_LOAD_COLUMNS,
        filters=[
            ("createdAtMillis", ">=", cutoff_oldest_ms),
            ("createdAtMillis", "<=", cutoff_recent_ms),
        ],
    ).reset_index(drop=True)
    print(f"    {len(notes):,} notes in window", flush=True)

    print(
        f"  Loading rater model output ({len(_RATER_LOAD_COLUMNS)} cols)...", flush=True
    )
    raters = pd.read_parquet(rater_path, columns=_RATER_LOAD_COLUMNS)
    print(f"    {len(raters):,} rater rows loaded", flush=True)

    print("  Loading user enrollment states...", flush=True)
    other_bot_ids = (
        load_api_writer_ids(enrollment_path) - writer_account_ids - excluded_account_ids
    )
    print(f"    {len(other_bot_ids):,} other-bot accounts derived", flush=True)

    n_before = len(notes)
    notes = notes[~notes["noteAuthorParticipantId"].isin(excluded_account_ids)].copy()
    print(
        f"  Notes after excluding blocked authors: {len(notes):,} "
        f"({n_before - len(notes):,} dropped)",
        flush=True,
    )

    notes = _sample_mixed_authors(notes, writer_account_ids, other_bot_ids, seed=seed)

    our_note_ids = set(notes["noteId"].values)
    print(
        f"  Loading ratings for {len(our_note_ids):,} notes "
        f"({len(_RATING_LOAD_COLUMNS)} cols, predicate pushdown)...",
        flush=True,
    )
    ratings = pd.read_parquet(
        ratings_path,
        columns=_RATING_LOAD_COLUMNS,
        filters=[("noteId", "in", list(our_note_ids))],
    )
    print(f"    {len(ratings):,} rating rows loaded", flush=True)

    print("  Merging rater factors...", flush=True)
    ratings = ratings.merge(raters, on="raterParticipantId", how="left")
    n_before = len(ratings)
    ratings = ratings.dropna(subset=["expansionRaterFactor1"])
    print(
        f"    {len(ratings):,} ratings with defined rater factor "
        f"({n_before - len(ratings):,} dropped)",
        flush=True,
    )

    print("  Bucketing ratings by rater factor...", flush=True)
    ratings["bucket"] = ratings["expansionRaterFactor1"].apply(assign_bucket)

    return notes, ratings


def _sample_mixed_authors(
    notes: pd.DataFrame,
    writer_account_ids: set[str],
    other_bot_ids: set[str],
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    print("  Partitioning notes by author class...", flush=True)
    our_mask = notes["noteAuthorParticipantId"].isin(writer_account_ids)
    other_mask = notes["noteAuthorParticipantId"].isin(other_bot_ids)
    human_mask = ~(our_mask | other_mask)

    our_notes = notes[our_mask]
    other_notes = notes[other_mask]
    human_notes = notes[human_mask]

    print(
        f"    available -- our: {len(our_notes):,}  other: {len(other_notes):,}  "
        f"human: {len(human_notes):,}",
        flush=True,
    )

    n = len(our_notes)
    if n == 0:
        print(
            "  WARNING: no notes from our writer accounts in the age window; "
            "returning empty training set.",
            flush=True,
        )
        return our_notes.copy()

    n_other_avail = len(other_notes)
    n_other_keep = min(n, n_other_avail)
    if n_other_keep < n_other_avail:
        pos = rng.choice(n_other_avail, size=n_other_keep, replace=False)
        other_kept = other_notes.iloc[pos]
    else:
        other_kept = other_notes

    n_human_target = max(0, 2 * n - n_other_keep)
    n_human_avail = len(human_notes)
    n_human_keep = min(n_human_target, n_human_avail)
    if n_human_keep < n_human_avail:
        pos = rng.choice(n_human_avail, size=n_human_keep, replace=False)
        human_kept = human_notes.iloc[pos]
    else:
        human_kept = human_notes

    print(f"  Our bots:    N={n:,}", flush=True)
    print(
        f"  Other bots:  {n_other_keep:,} sampled from {n_other_avail:,} available "
        f"(cap = N = {n:,})",
        flush=True,
    )
    print(
        f"  Humans:      {n_human_keep:,} sampled from {n_human_avail:,} available "
        f"(target = 2*N - other = {n_human_target:,})",
        flush=True,
    )
    total = n + n_other_keep + n_human_keep
    print(f"  Total notes: {total:,}", flush=True)

    result = pd.concat(
        [our_notes, other_kept, human_kept],
        ignore_index=True,
    )
    if len(result) != total:
        raise RuntimeError(
            f"_sample_mixed_authors size mismatch: result has {len(result):,} rows "
            f"but expected {total:,} (our={len(our_notes)}, "
            f"other={len(other_kept)}, human={len(human_kept)})"
        )
    return result


def _snapshot_ranges() -> list[tuple[int, int, int]]:
    ranges = []
    for i, t in enumerate(SNAPSHOT_THRESHOLDS):
        lo = (SNAPSHOT_THRESHOLDS[i - 1] + 1) if i > 0 else t
        ranges.append((lo, t, t))
    return ranges


def generate_training_instances(
    notes: pd.DataFrame,
    ratings: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    tag_parquet_cols = list(TAG_COLUMN_MAP.keys())
    missing_tags = [c for c in tag_parquet_cols if c not in ratings.columns]
    assert not missing_tags, (
        f"Expected tag columns missing from ratings: {missing_tags}"
    )
    keep_cols = [
        "noteId",
        "createdAtMillis",
        "helpfulnessLevel",
        "bucket",
    ] + tag_parquet_cols
    ratings_slim = ratings[keep_cols].copy()
    ratings_slim = ratings_slim.sort_values(["noteId", "createdAtMillis"]).reset_index(
        drop=True
    )

    max_per_note = max(SNAPSHOT_THRESHOLDS)
    n_before_prune = len(ratings_slim)
    ratings_slim = (
        ratings_slim.groupby("noteId", sort=False)
        .head(max_per_note)
        .reset_index(drop=True)
    )
    print(
        f"  Pruned ratings to first {max_per_note} per note: "
        f"{n_before_prune:,} -> {len(ratings_slim):,} rows",
        flush=True,
    )

    note_meta = notes.set_index("noteId")[["currentStatus"]]

    bucket_vals = ratings_slim["bucket"].values
    helpfulness_vals = ratings_slim["helpfulnessLevel"].values
    for bucket in BUCKETS:
        is_bucket = bucket_vals == bucket
        for raw_level, prod_level in HELPFULNESS_MAP.items():
            ratings_slim[f"{bucket}_{prod_level}"] = (
                is_bucket & (helpfulness_vals == raw_level)
            ).astype(np.int8)
        for parquet_col, prod_name in TAG_COLUMN_MAP.items():
            ratings_slim[f"{bucket}_{prod_name}"] = np.where(
                is_bucket,
                ratings_slim[parquet_col].fillna(0).values,
                0,
            )

    missing_cols = [col for col in RATING_COLUMNS if col not in ratings_slim.columns]
    assert not missing_cols, f"Rating columns missing after encoding: {missing_cols}"

    helpfulness_cols = [f"{b}_{l}" for b in BUCKETS for l in HELPFULNESS_LEVELS]
    row_sums = ratings_slim[helpfulness_cols].sum(axis=1)
    assert (row_sums == 1).all(), (
        f"Expected exactly 1 helpfulness indicator per row, "
        f"found {row_sums.nunique()} distinct sums: {sorted(row_sums.unique())}"
    )

    for bucket in BUCKETS:
        bucket_mask = bucket_vals == bucket
        tag_cols = [f"{bucket}_{TAG_COLUMN_MAP[c]}" for c in tag_parquet_cols]
        other_bucket_tags = ratings_slim.loc[~bucket_mask, tag_cols]
        assert (other_bucket_tags == 0).all().all(), (
            f"Non-zero tag values found outside the '{bucket}' bucket"
        )

    ratings_slim["_pos"] = ratings_slim.groupby("noteId").cumcount()
    ratings_slim[RATING_COLUMNS] = ratings_slim.groupby("noteId")[
        RATING_COLUMNS
    ].cumsum()

    group_sizes = ratings_slim.groupby("noteId").size()
    valid_note_ids = group_sizes.index.intersection(note_meta.index)
    valid_sizes = group_sizes.loc[valid_note_ids]
    snap_ranges = _snapshot_ranges()

    all_note_ids: list[np.ndarray] = []
    all_sampled_n: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for lo, hi, label in snap_ranges:
        eligible_mask = valid_sizes.values >= hi
        n_eligible = eligible_mask.sum()
        if n_eligible == 0:
            continue
        all_note_ids.append(valid_sizes.index[eligible_mask].values)

        all_sampled_n.append(rng.randint(lo, hi + 1, size=n_eligible))

        all_labels.append(np.full(n_eligible, label))

    if not all_note_ids:
        return pd.DataFrame()

    note_ids = np.concatenate(all_note_ids)
    sampled_ns = np.concatenate(all_sampled_n)
    labels = np.concatenate(all_labels)

    cumsum_indexed = ratings_slim.set_index(["noteId", "_pos"])[RATING_COLUMNS]
    lookup_idx = pd.MultiIndex.from_arrays([note_ids, sampled_ns - 1])
    instances = cumsum_indexed.loc[lookup_idx].reset_index(drop=True)

    instances["noteId"] = note_ids
    instances["num_ratings"] = labels
    instances["total_ratings"] = sampled_ns
    meta = note_meta.loc[note_ids].reset_index(drop=True)
    instances["currentStatus"] = meta["currentStatus"].values

    return instances


def prepare_features_and_labels(inst: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = compute_model_features(inst)
    labels = (inst["currentStatus"] != "CURRENTLY_RATED_HELPFUL").astype(int)
    return features, labels


def split_data(
    instances: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    note_ids = instances["noteId"].unique()
    train_notes, eval_notes = train_test_split(
        note_ids, test_size=test_size, random_state=seed
    )
    return (
        instances[instances["noteId"].isin(set(train_notes))].copy(),
        instances[instances["noteId"].isin(set(eval_notes))].copy(),
    )


SWEEP_TARGET_FPR = 0.01


def run_sweep(
    grid: list[dict],
    fixed_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: np.ndarray,
    cv: bool = True,
    n_splits: int = 5,
    seed: int = 42,
) -> list[dict]:
    if cv:
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X_train, y_train, groups=groups))
        eval_label = f"{n_splits}-fold GroupKFold CV"
    else:
        unique_groups = np.unique(groups)
        rng = np.random.RandomState(seed)
        rng.shuffle(unique_groups)
        split_point = len(unique_groups) * (n_splits - 1) // n_splits
        train_groups = set(unique_groups[:split_point])
        train_mask = np.isin(groups, list(train_groups))
        splits = [
            (
                np.where(train_mask)[0],
                np.where(~train_mask)[0],
            )
        ]
        eval_label = f"single grouped 1/{n_splits} holdout"

    print()
    print("=" * 72)
    print(f"Sweep: {len(grid)} configs x {eval_label}")
    print(f"Target: recall at FPR <= {SWEEP_TARGET_FPR}")
    print(f"Fixed params: {fixed_params}")
    print("=" * 72)

    results = []
    for cfg_idx, cfg in enumerate(grid):
        full = {**fixed_params, **cfg}
        fold_recalls = []
        for train_idx, val_idx in splits:
            pipe = build_model(**full)
            pipe.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            proba = pipe.predict_proba(X_train.iloc[val_idx])[:, 1]
            y_val = y_train.iloc[val_idx]
            if len(y_val.unique()) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_val, proba)
            idx = max(np.searchsorted(fpr, SWEEP_TARGET_FPR, side="right") - 1, 0)
            fold_recalls.append(float(tpr[idx]))
        if not fold_recalls:
            continue
        mean_r = float(np.mean(fold_recalls))
        std_r = float(np.std(fold_recalls))
        results.append(
            {
                **full,
                "mean_recall_at_1pct_fpr": round(mean_r, 5),
                "std_recall_at_1pct_fpr": round(std_r, 5),
            }
        )
        print(
            f"  [{cfg_idx + 1:2d}/{len(grid)}] {cfg} -> "
            f"recall@1%FPR={mean_r:.4f} +/- {std_r:.4f}"
        )
    results.sort(key=lambda r: r["mean_recall_at_1pct_fpr"], reverse=True)

    print()
    print("Top 10 configs:")
    for r in results[: min(10, len(results))]:
        print(
            f"  {r['mean_recall_at_1pct_fpr']:.4f} +/- "
            f"{r['std_recall_at_1pct_fpr']:.4f}  "
            f"n_bins_fine={r['n_bins_fine']} C={r['C']} select_k={r['select_k']}"
        )
    return results


def train_and_evaluate(
    pipe: Pipeline,
    train_inst: pd.DataFrame,
    eval_inst: pd.DataFrame,
    out_dir: Path,
    params: dict | None = None,
) -> dict:
    X_train, y_train = prepare_features_and_labels(train_inst)
    X_eval, y_eval = prepare_features_and_labels(eval_inst)

    pipe.fit(X_train, y_train)
    train_proba = pipe.predict_proba(X_train)[:, 1]
    eval_proba = pipe.predict_proba(X_eval)[:, 1]

    metrics = {
        "train": _compute_metrics(y_train, train_proba),
        "eval": _compute_metrics(y_eval, eval_proba),
    }
    print_metrics(metrics)

    plot_all_roc(
        train_inst,
        X_train,
        y_train,
        train_proba,
        eval_inst,
        X_eval,
        y_eval,
        eval_proba,
        out_dir,
    )
    threshold_tables = compute_threshold_tables(train_inst, y_train, train_proba)

    joblib.dump(pipe, out_dir / "deletion_model.joblib")
    summary = {
        "params": params or {},
        "data": {
            "train_instances": len(train_inst),
            "eval_instances": len(eval_inst),
            "train_notes": int(train_inst["noteId"].nunique()),
            "eval_notes": int(eval_inst["noteId"].nunique()),
        },
        "metrics": metrics,
        "thresholds_by_num_ratings": threshold_tables,
    }
    with open(out_dir / "deletion_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return metrics


def _compute_metrics(y_true: pd.Series, y_proba: np.ndarray) -> dict:
    try:
        a = roc_auc_score(y_true, y_proba)
    except ValueError:
        a = float("nan")
    try:
        ap = average_precision_score(y_true, y_proba)
    except ValueError:
        ap = float("nan")
    result = {"auc": round(a, 4), "ap": round(ap, 4), "n": int(len(y_true))}
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    for tf in [0.05, 0.10, 0.20]:
        idx = max(np.searchsorted(fpr, tf, side="right") - 1, 0)
        result[f"recall_at_{int(tf * 100)}pct_fpr"] = round(float(tpr[idx]), 4)
    return result


def print_metrics(metrics: dict) -> None:
    for name, m in metrics.items():
        print(f"\n--- {name} (n={m['n']}) ---")
        print(f"  AUC: {m['auc']:.4f}  AP: {m['ap']:.4f}")
        for k in ["recall_at_5pct_fpr", "recall_at_10pct_fpr", "recall_at_20pct_fpr"]:
            if k in m:
                print(f"  {k}: {m[k]:.4f}")


FPR_TARGETS = [0.005, 0.01, 0.02]
FPR_TARGET_NAMES = ["0.5pct_fpr", "1pct_fpr", "2pct_fpr"]


def _combined_panel_arrays(
    inst: pd.DataFrame,
    note_ids: np.ndarray,
    num_rats: np.ndarray,
    y_vals: np.ndarray,
    proba: np.ndarray,
    eligible: set,
    thresholds: list[int],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | None:
    if not eligible:
        return None
    panel_max = max(thresholds)
    eligible_arr = np.array(list(eligible))
    in_panel = np.isin(num_rats, thresholds)
    in_eligible = np.isin(note_ids, eligible_arr)
    mask = in_panel & in_eligible
    if mask.sum() == 0:
        return None

    df = pd.DataFrame(
        {
            "noteId": note_ids[mask],
            "y": y_vals[mask],
            "p": proba[mask],
        }
    )
    grouped = df.groupby("noteId", sort=False).agg(
        y=("y", "first"),
        p=("p", "max"),
    )

    ci_mask = (num_rats == panel_max) & in_eligible
    ci = (
        inst[ci_mask]
        .drop_duplicates(subset=["noteId"], keep="first")
        .set_index("noteId")
    )
    ci = ci.reindex(grouped.index)
    return grouped["y"].to_numpy(), grouped["p"].to_numpy(), ci


def compute_threshold_tables(
    inst: pd.DataFrame,
    y: pd.Series,
    proba: np.ndarray,
) -> dict:
    result = {}
    note_ids = inst["noteId"].values
    num_rats = inst["num_ratings"].values
    y_vals = y.values if hasattr(y, "values") else np.asarray(y)

    for key, _label, thresholds in PANELS:
        gmax = max(thresholds)
        eligible = set(note_ids[num_rats == gmax])

        arrays = _combined_panel_arrays(
            inst,
            note_ids,
            num_rats,
            y_vals,
            proba,
            eligible,
            thresholds,
        )
        if arrays is None:
            continue
        cy, cp, ci = arrays

        neg_scores = cp[cy == 0]
        pos_scores = cp[cy == 1]
        pos_i = ci[cy == 1]
        if len(neg_scores) == 0 or len(pos_scores) == 0:
            continue

        total_h = (
            pos_i["negative_helpful"].values
            + pos_i["neutral_helpful"].values
            + pos_i["positive_helpful"].values
        )
        total_nh = (
            pos_i["negative_not_helpful"].values
            + pos_i["neutral_not_helpful"].values
            + pos_i["positive_not_helpful"].values
        )

        panel_data = {
            "n_crh": int((cy == 0).sum()),
            "n_non_crh": int(len(pos_scores)),
        }
        for ft, fn in zip(FPR_TARGETS, FPR_TARGET_NAMES):
            threshold = float(np.percentile(neg_scores, (1 - ft) * 100))
            actual_fpr = float((neg_scores > threshold).mean())
            detected = pos_scores > threshold
            det_nh, det_h = total_nh[detected], total_h[detected]
            panel_data[fn] = {
                "threshold": round(threshold, 6),
                "actual_crh_fpr": round(actual_fpr, 4),
                "recall": round(float(detected.mean()), 4),
                "n_detected": int(detected.sum()),
                "detected_nh_lt_h": int((det_nh < det_h).sum()),
                "detected_nh_gte_h": int((det_nh >= det_h).sum()),
            }

        result[f"train_{key}"] = panel_data
    return result


def _combined_max_curve(
    note_ids: np.ndarray,
    num_rats: np.ndarray,
    yv: np.ndarray,
    proba: np.ndarray,
    eligible: set,
    thresholds: list[int],
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    if not eligible:
        return np.array([]), np.array([]), 0.0, 0, 0
    in_panel = np.isin(num_rats, thresholds)
    in_eligible = np.isin(note_ids, list(eligible))
    mask = in_panel & in_eligible
    if mask.sum() == 0:
        return np.array([]), np.array([]), 0.0, 0, 0

    df = pd.DataFrame(
        {
            "noteId": note_ids[mask],
            "y": yv[mask],
            "p": proba[mask],
        }
    )

    grouped = df.groupby("noteId", sort=False).agg(y=("y", "first"), p=("p", "max"))
    cy = grouped["y"].to_numpy()
    cp = grouped["p"].to_numpy()
    if len(np.unique(cy)) < 2:
        return (
            np.array([]),
            np.array([]),
            0.0,
            int((cy == 0).sum()),
            int((cy == 1).sum()),
        )
    fpr, tpr, _ = roc_curve(cy, cp)
    return fpr, tpr, auc(fpr, tpr), int((cy == 0).sum()), int((cy == 1).sum())


def plot_all_roc(
    train_inst: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_proba: np.ndarray,
    eval_inst: pd.DataFrame,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    eval_proba: np.ndarray,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))

    for row, (sname, inst, y, proba) in enumerate(
        [
            ("Train", train_inst, y_train, train_proba),
            ("Eval", eval_inst, y_eval, eval_proba),
        ]
    ):
        yv = y.values if hasattr(y, "values") else np.asarray(y)
        nids = inst["noteId"].values
        nr = inst["num_ratings"].values

        for ci, (_key, label, thresholds) in enumerate(PANELS):
            ax = axes[row][ci]
            gmax = max(thresholds)
            eligible = set(nids[nr == gmax])

            cmap = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
            note_mask = (
                np.isin(nids, list(eligible))
                if eligible
                else np.zeros(
                    len(nids),
                    dtype=bool,
                )
            )
            for ti, t in enumerate(thresholds):
                cm = (nr == t) & note_mask
                if cm.sum() < 2:
                    continue
                cy, cp = yv[cm], proba[cm]
                if len(np.unique(cy)) < 2:
                    continue
                f, tp, _ = roc_curve(cy, cp)
                ra = auc(f, tp)
                n_crh_t = int((cy == 0).sum())
                n_non_crh_t = int((cy == 1).sum())
                ax.plot(
                    f,
                    tp,
                    color=cmap[ti],
                    label=f"ratings={t} (AUC={ra:.3f}, "
                    f"n_crh={n_crh_t}, n_non_crh={n_non_crh_t})",
                )

            cf, ct, cauc, n_neg, n_pos = _combined_max_curve(
                nids,
                nr,
                yv,
                proba,
                eligible,
                thresholds,
            )
            if cf.size > 0:
                ax.plot(
                    cf,
                    ct,
                    color="black",
                    linewidth=2.0,
                    label=f"combined (max, AUC={cauc:.3f}, "
                    f"n_crh={n_neg}, n_non_crh={n_pos})",
                )

            ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title(f"{sname} -- {label}")
            ax.legend(loc="lower right", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_dir / "deletion_roc.png", dpi=150)
    plt.close()
