from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from notable_post_model import features
from notable_post_model.model import NotablePostMLP
from utils.url_utils import remove_urls


FPR_CAP = 0.10
TARGET_LANG = "en"
TARGET_SIZE = "xxl"
TARGET = "is_not_crh"


MAX_NA_DROP_FRACTION = 0.001


_FEED_COLUMNS = [
    "note_id",
    "post_id",
    "enqueued_at",
    "lang",
    "api_feed",
    "retweet_count",
    "reply_count",
    "like_count",
    "quote_count",
    "bookmark_count",
    "impression_count",
    "author_followers_count",
    "author_following_count",
    "author_verified_type",
    "author_tweet_count",
    "author_listed_count",
    "author_like_count",
    "author_media_count",
    "author_parody",
    "num_unique_sources",
    "total_source_suggestions",
    "has_photo",
    "has_video",
    "hist_note_count",
    "hist_crh_count",
    "hist_crnh_count",
    "hist_total_ratings",
]


_REQUIRED_FEATURE_COLUMNS = [c for c in _FEED_COLUMNS if c != "note_id"]


_TRIPLE_KEYS = ["post_id", "enqueued_at", "api_feed"]


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_feed_events(feeds_path: str) -> pd.DataFrame:
    if not os.path.isfile(feeds_path):
        raise FileNotFoundError(f"Feed parquet not found: {feeds_path!r}")

    print(f"  Feed file: {feeds_path}")

    dataset = pa_ds.dataset(feeds_path, format="parquet")
    table = dataset.to_table(columns=_FEED_COLUMNS)
    df = table.to_pandas()

    print(
        f"  Loaded {len(df):,} feed event rows "
        f"({df.memory_usage(deep=True).sum() / (1024**2):.1f} MB)"
    )
    if not df.empty:
        triples = df[_TRIPLE_KEYS].drop_duplicates()
        print(
            f"  Unique post_ids: {df['post_id'].nunique():,}; "
            f"unique triples: {len(triples):,}; "
            f"rows with note_id: {df['note_id'].notna().sum():,}"
        )
    return df


def _load_post_text(feeds_path: str) -> pd.DataFrame:
    text_cols = ["post_id", "enqueued_at", "api_feed", "post_text"]
    available = pa_ds.dataset(feeds_path, format="parquet").schema.names
    if "post_text" not in available:
        raise ValueError(
            f"Feed parquet {feeds_path!r} does not contain a 'post_text' column. "
            f"Available columns: {sorted(available)}"
        )
    dataset = pa_ds.dataset(feeds_path, format="parquet")
    table = dataset.to_table(columns=text_cols)
    df = table.to_pandas()
    df = df.drop_duplicates(
        subset=_TRIPLE_KEYS,
        keep="first",
    ).reset_index(drop=True)
    print(f"  Loaded post_text for {len(df):,} triples")
    return df


def preprocess_text(text) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text)
    if pd.isna(text):
        return ""
    return remove_urls(text).strip()


def compute_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int = 256,
    device: torch.device | None = None,
    cache_path: str | None = None,
) -> np.ndarray:
    cache_key = f"{model_name}_{len(texts)}"

    if cache_path and os.path.isfile(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            if (
                "embeddings" in data
                and "cache_key" in data
                and str(data["cache_key"]) == cache_key
                and len(data["embeddings"]) == len(texts)
            ):
                print(f"  Loaded cached embeddings from {cache_path}")
                return data["embeddings"]
            print(f"  Cache exists but key/length mismatch; recomputing")
        except Exception as e:
            print(f"  Could not load cache ({e}); recomputing")

    from sentence_transformers import SentenceTransformer

    if device is None:
        device = select_device()

    print(f"  Loading embedding model: {model_name}")
    st_model = SentenceTransformer(model_name)
    print(
        f"  Embedding {len(texts):,} texts (batch_size={batch_size}, device={device})..."
    )
    embeddings = st_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=str(device),
    )

    if cache_path:
        np.savez(cache_path, embeddings=embeddings, cache_key=np.array(cache_key))
        print(f"  Saved embeddings to cache: {cache_path}")

    return embeddings


def _build_label_lookups(
    notes_path: str,
    nsh_path: str,
) -> tuple[set[int], pd.Series]:
    nsh = pd.read_parquet(
        nsh_path, columns=["noteId", "firstNonNMRStatus", "createdAtMillis"]
    )
    notes = pd.read_parquet(notes_path, columns=["noteId", "tweetId"])

    crh_in_nsh = nsh.loc[
        nsh["firstNonNMRStatus"] == "CURRENTLY_RATED_HELPFUL",
        ["noteId", "createdAtMillis"],
    ]
    crh_note_id_set = set(crh_in_nsh["noteId"].astype("int64").to_numpy())

    crh_with_post = crh_in_nsh.merge(notes, on="noteId", how="inner")
    post_max_crh = (
        crh_with_post.groupby("tweetId")["createdAtMillis"].max().astype("float64")
    )

    print(
        f"  Label lookups: {len(crh_note_id_set):,} CRH noteIds in nsh; "
        f"{len(post_max_crh):,} posts with at least one CRH note "
        f"(after notes <-> nsh join)"
    )
    return crh_note_id_set, post_max_crh


def _label_and_dedup(
    feed_df: pd.DataFrame,
    notes_path: str,
    nsh_path: str,
    prune_recent_hours: int = 24,
) -> pd.DataFrame:
    df = feed_df.copy()

    crh_note_id_set, post_max_crh = _build_label_lookups(notes_path, nsh_path)

    note_id_int = df["note_id"]
    path_a = note_id_int.notna() & note_id_int.fillna(-1).astype("int64").isin(
        crh_note_id_set
    )

    post_max = df["post_id"].map(post_max_crh)
    path_b = post_max.notna() & (
        post_max.astype("float64") > df["enqueued_at"].astype("float64")
    )

    df["is_positive"] = (path_a | path_b).to_numpy()

    n_path_a = int(path_a.sum())
    n_path_b = int(path_b.sum())
    n_a_only = int((path_a & ~path_b).sum())
    n_b_only = int((path_b & ~path_a).sum())
    n_both = int((path_a & path_b).sum())
    print(
        f"  Row-level labels: path_A={n_path_a:,}, path_B={n_path_b:,} "
        f"(A-only={n_a_only:,}, B-only={n_b_only:,}, both={n_both:,}, "
        f"positive={int(df['is_positive'].sum()):,} / {len(df):,})"
    )

    if prune_recent_hours > 0 and not df.empty:
        max_enq = int(df["enqueued_at"].max())
        cutoff_enq = max_enq - prune_recent_hours * 3600 * 1000
        n_before = len(df)
        keep = (df["enqueued_at"] <= cutoff_enq) | df["is_positive"]
        df = df[keep].copy()
        print(
            f"  Recent-prune: dropped {n_before - len(df):,} negative rows with "
            f"enqueued_at > {cutoff_enq} (= max - {prune_recent_hours}h); "
            f"positives kept regardless of recency"
        )

    df = df.drop(columns=["note_id"])

    _assert_features_constant_within_triple(df)

    feature_cols = [c for c in df.columns if c not in (_TRIPLE_KEYS + ["is_positive"])]
    agg_spec: dict[str, str] = {c: "first" for c in feature_cols}
    agg_spec["is_positive"] = "max"
    triples = df.groupby(_TRIPLE_KEYS, sort=False, as_index=False).agg(agg_spec)
    print(
        f"  Dedup: {len(df):,} rows -> {len(triples):,} triples; "
        f"positive triples = {int(triples['is_positive'].sum()):,} "
        f"({triples['is_positive'].mean():.4%})"
    )

    n_before = len(triples)
    triples = triples.dropna(subset=_REQUIRED_FEATURE_COLUMNS).copy()
    n_dropped = n_before - len(triples)
    drop_frac = n_dropped / n_before if n_before > 0 else 0.0
    assert drop_frac < MAX_NA_DROP_FRACTION, (
        f"Dropped {n_dropped}/{n_before} ({drop_frac:.2%}) triples due to NA "
        f"feature columns; threshold is {MAX_NA_DROP_FRACTION:.1%}."
    )
    print(
        f"  NA filter: kept {len(triples):,} triples "
        f"(dropped {n_dropped:,} for NA features = {drop_frac:.2%})"
    )

    return triples


def _assert_features_constant_within_triple(df: pd.DataFrame) -> None:
    if df.empty:
        return
    feature_cols = [c for c in df.columns if c not in (_TRIPLE_KEYS + ["is_positive"])]
    n_unique_full = df.drop_duplicates(subset=_TRIPLE_KEYS + feature_cols).shape[0]
    n_unique_triples = df.drop_duplicates(subset=_TRIPLE_KEYS).shape[0]
    excess = n_unique_full - n_unique_triples
    excess_frac = excess / n_unique_triples if n_unique_triples > 0 else 0.0
    assert excess_frac <= 0.001, (
        f"Feature columns vary within triples: {excess_frac:.2%} excess; "
        f"threshold is 0.1%."
    )


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    features_df = features.compute_model_features(df)
    for col in features_df.columns:
        df[col] = features_df[col].values
    df["is_not_crh"] = (~df["is_positive"].astype(bool)).astype("int8")
    return df


def prepare_data(
    feeds_path: str,
    notes_path: str,
    nsh_path: str,
    test_size: float = 0.2,
    prune_recent_hours: int = 24,
):
    feed_df = _load_feed_events(feeds_path)
    text_df = _load_post_text(feeds_path)

    triples = _label_and_dedup(feed_df, notes_path, nsh_path, prune_recent_hours)
    feat = _extract_features(triples)

    feat = feat.merge(
        text_df[["post_id", "enqueued_at", "api_feed", "post_text"]],
        on=_TRIPLE_KEYS,
        how="left",
    )
    n_with_text = feat["post_text"].notna().sum()
    print(
        f"  Post text coverage: {n_with_text:,} / {len(feat):,} "
        f"({n_with_text / len(feat):.1%})"
    )

    feat = feat.sort_values("enqueued_at", kind="stable").reset_index(drop=True)
    n_train = int(len(feat) * (1.0 - test_size))
    train_feat = feat.iloc[:n_train].reset_index(drop=True)
    eval_feat = feat.iloc[n_train:].reset_index(drop=True)

    if len(train_feat) > 0 and len(eval_feat) > 0:
        split_ms = int(eval_feat["enqueued_at"].iloc[0])
        split_dt = pd.to_datetime(split_ms, unit="ms")
        print(
            f"  Temporal split: train n={len(train_feat):,}  eval n={len(eval_feat):,}  "
            f"|  split at enqueued_at={split_ms} ({split_dt} UTC)"
        )

    def _split(part: pd.DataFrame):
        X = part[features.ALL_FEATURES].reset_index(drop=True)
        y = part[TARGET].astype(int).to_numpy()
        meta = part[["api_feed", "lang_top", "enqueued_at"]].reset_index(drop=True)
        text = part["post_text"].reset_index(drop=True)
        return X, y, meta, text

    X_tr, y_tr, meta_tr, text_tr = _split(train_feat)
    X_ev, y_ev, meta_ev, text_ev = _split(eval_feat)
    return X_tr, y_tr, meta_tr, text_tr, X_ev, y_ev, meta_ev, text_ev


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    meta_eval: pd.DataFrame,
    input_dim: int,
    hidden_dim: int = 512,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 2048,
    epochs: int = 50,
    patience: int = 5,
    pos_weight_ratio: float = 10.0,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[NotablePostMLP, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    if device is None:
        device = select_device()
    print(f"  Device: {device}")

    mlp = NotablePostMLP(input_dim, hidden_dim, dropout).to(device)
    print(f"  Model params: {sum(p.numel() for p in mlp.parameters()):,}")

    sample_weights = np.where(y_train == 0, pos_weight_ratio, 1.0).astype(np.float32)

    X_tr_t = torch.from_numpy(X_train).to(device)
    y_tr_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    w_tr_t = torch.from_numpy(sample_weights).to(device)
    X_ev_t = torch.from_numpy(X_eval).to(device)

    train_ds = TensorDataset(X_tr_t, y_tr_t, w_tr_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_auc = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        mlp.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb, wb in train_dl:
            logits = mlp(xb)
            raw_loss = loss_fn(logits, yb)
            loss = (raw_loss * wb).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        mlp.eval()
        with torch.no_grad():
            eval_logits = mlp(X_ev_t)
            eval_probs = torch.sigmoid(eval_logits).cpu().numpy()
        eval_auc = roc_auc_score(y_eval, eval_probs)
        eval_en_xxl = en_xxl_recall_at_fpr(y_eval, eval_probs, meta_eval["api_feed"])

        avg_loss = epoch_loss / n_batches
        improved = eval_auc > best_auc
        marker = " *" if improved else ""
        print(
            f"  Epoch {epoch:3d}/{epochs}: "
            f"loss={avg_loss:.4f}  eval_auc={eval_auc:.4f}  "
            f"en_xxl@FPR={eval_en_xxl:.3f}{marker}"
        )

        if improved:
            best_auc = eval_auc
            best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    mlp.load_state_dict(best_state)
    mlp.eval()
    with torch.no_grad():
        train_scores = torch.sigmoid(mlp(X_tr_t)).cpu().numpy()
        eval_scores = torch.sigmoid(mlp(X_ev_t)).cpu().numpy()

    return mlp, train_scores, eval_scores


def run_mlp_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    meta_train: pd.DataFrame,
    input_dim: int,
    grid: list[dict],
    fixed: dict,
    seed: int = 42,
    device: torch.device | None = None,
) -> list[tuple[float, dict]]:
    order = np.argsort(meta_train["enqueued_at"].values, kind="stable")
    n = len(order)
    n_val = max(1, n // 5)
    val_idx = order[-n_val:]
    fit_idx = order[:-n_val]

    X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    meta_val = meta_train.iloc[val_idx].reset_index(drop=True)

    print()
    print("=" * 72)
    print(
        f"MLP Sweep: {len(grid)} configs, temporal holdout "
        f"(fit={len(fit_idx):,}, val={len(val_idx):,})"
    )
    print(f"Target: en_xxl recall @ CRH FPR <= {FPR_CAP}")
    print(f"Fixed: {fixed}")
    print("=" * 72)

    results: list[tuple[float, dict]] = []
    for i, params in enumerate(grid, 1):
        t0 = time.time()
        full = {**fixed, **params}
        _, _, val_scores = train_mlp(
            X_train=X_fit,
            y_train=y_fit,
            X_eval=X_val,
            y_eval=y_val,
            meta_eval=meta_val,
            input_dim=input_dim,
            device=device,
            **full,
        )
        score = en_xxl_recall_at_fpr(y_val, val_scores, meta_val["api_feed"])
        elapsed = time.time() - t0
        results.append((score, params))
        tag = " ".join(f"{k}={v}" for k, v in params.items())
        print(
            f"  [{i:3d}/{len(grid)}] en_xxl@FPR={score:.3f}  ({elapsed:4.1f}s)  {tag}"
        )

    results.sort(key=lambda r: r[0], reverse=True)
    print()
    print("Top 10 configs:")
    for score, params in results[:10]:
        print(f"  {score:.3f}  {params}")
    return results


def _feed_parts(api_feed: pd.Series) -> tuple[pd.Series, pd.Series]:
    filled = api_feed.fillna("unknown")
    parts = filled.str.split("_", expand=True)
    if 1 not in parts.columns:
        parts[1] = None
    lang = parts[0].fillna("unknown")
    size = parts[1].fillna("unknown")
    return lang, size


def en_xxl_recall_at_fpr(
    y: np.ndarray,
    scores: np.ndarray,
    api_feed: pd.Series,
    fpr_cap: float = FPR_CAP,
) -> float:
    lang, size = _feed_parts(api_feed)
    in_en = lang.values == TARGET_LANG
    y_en = y[in_en]
    scores_en = scores[in_en]
    size_en = size.values[in_en]

    crh_scores = scores_en[y_en == 0]
    if len(crh_scores) == 0:
        return 0.0
    k = max(1, int(np.ceil(fpr_cap * len(crh_scores))))
    threshold = np.partition(crh_scores, -k)[-k]

    xxl_mask = (size_en == TARGET_SIZE) & (y_en == 1)
    xxl_ncrh_scores = scores_en[xxl_mask]
    if len(xxl_ncrh_scores) == 0:
        return 0.0
    return float((xxl_ncrh_scores >= threshold).mean())


def _thresholds_at_fpr(
    y: np.ndarray,
    scores: np.ndarray,
    api_feed: pd.Series,
    lang_group: str,
    fpr_targets: list[float],
) -> dict[str, float]:
    feed_lang, _ = _feed_parts(api_feed)
    crh_scores = 1.0 - scores[(feed_lang.values == lang_group) & (y == 0)]
    if len(crh_scores) == 0:
        return {str(fpr): None for fpr in fpr_targets}
    sorted_asc = np.sort(crh_scores)
    result = {}
    for fpr in fpr_targets:
        k = int(np.floor(fpr * len(crh_scores)))
        result[str(fpr)] = float(sorted_asc[k])
    return result


def print_metrics(name: str, y: np.ndarray, scores: np.ndarray, split: str) -> None:
    auc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)
    crh_scores = scores[y == 0]
    ncrh_scores = scores[y == 1]
    print(f"  [{name}] {split:5s}: AUC={auc:.4f}  AP={ap:.4f}")
    for fpr_target in (0.05, 0.10, 0.20):
        if len(crh_scores) == 0:
            continue
        k = max(1, int(np.ceil(fpr_target * len(crh_scores))))
        threshold = np.partition(crh_scores, -k)[-k]
        recall = float((ncrh_scores >= threshold).mean())
        print(f"           non-CRH recall @ CRH FPR<={fpr_target:.2f}: {recall:.3f}")


def _roc_with_shared_negatives(
    pos_scores: np.ndarray, neg_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    pos_sorted = np.sort(pos_scores)
    neg_sorted = np.sort(neg_scores)
    thresholds = np.unique(np.concatenate([pos_sorted, neg_sorted]))[::-1]
    pos_ge = len(pos_sorted) - np.searchsorted(pos_sorted, thresholds, side="left")
    neg_ge = len(neg_sorted) - np.searchsorted(neg_sorted, thresholds, side="left")
    tpr = pos_ge / len(pos_sorted)
    fpr = neg_ge / len(neg_sorted)
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    return fpr, tpr


def plot_feed_roc(
    y: np.ndarray,
    scores: np.ndarray,
    api_feed: pd.Series,
    lang_group: str,
    ax,
    title: str,
) -> None:
    feed_lang, feed_size = _feed_parts(api_feed)
    in_lang = feed_lang.values == lang_group
    y_lg = y[in_lang]
    scores_lg = scores[in_lang]
    size_lg = feed_size.values[in_lang]

    crh_scores = scores_lg[y_lg == 0]
    ncrh_scores = scores_lg[y_lg == 1]
    n_crh_total = len(crh_scores)
    n_ncrh_total = len(ncrh_scores)
    if n_crh_total == 0 or n_ncrh_total == 0:
        ax.text(0.5, 0.5, f"no {lang_group} data", ha="center", va="center")
        ax.set_title(title)
        return

    size_order = ["small", "large", "xl", "xxl"]
    color_map = {"small": "C0", "large": "C1", "xl": "C2", "xxl": "C3"}
    for size in size_order:
        mask = (size_lg == size) & (y_lg == 1)
        n = int(mask.sum())
        n_crh_size = int(((size_lg == size) & (y_lg == 0)).sum())
        if n < 5:
            continue
        fpr, tpr = _roc_with_shared_negatives(scores_lg[mask], crh_scores)
        ax.plot(
            fpr,
            tpr,
            color=color_map[size],
            label=f"{size} (n_ncrh={n}, n_crh={n_crh_size})",
        )

    fpr, tpr = _roc_with_shared_negatives(ncrh_scores, crh_scores)
    auc = roc_auc_score(y_lg, scores_lg)
    ax.plot(
        fpr,
        tpr,
        color="black",
        linewidth=2.0,
        label=f"combined (n_ncrh={n_ncrh_total}, n_crh={n_crh_total}, AUC={auc:.3f})",
    )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR on CRH samples (shared pool)")
    ax.set_ylabel("Recall on non-CRH samples")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_ablation_roc(
    ablation_results: list[
        tuple[str, np.ndarray, np.ndarray, pd.Series, np.ndarray, np.ndarray, pd.Series]
    ],
    out_path: Path,
) -> None:
    n_ablations = len(ablation_results)
    fig, axes = plt.subplots(n_ablations, 4, figsize=(28, 6 * n_ablations))
    if n_ablations == 1:
        axes = axes.reshape(1, -1)

    for row, (name, y_tr, s_tr, f_tr, y_ev, s_ev, f_ev) in enumerate(ablation_results):
        plot_feed_roc(y_tr, s_tr, f_tr, "en", axes[row, 0], f"{name} -- English, TRAIN")
        plot_feed_roc(y_ev, s_ev, f_ev, "en", axes[row, 1], f"{name} -- English, EVAL")
        plot_feed_roc(y_tr, s_tr, f_tr, "intl", axes[row, 2], f"{name} -- Intl, TRAIN")
        plot_feed_roc(y_ev, s_ev, f_ev, "intl", axes[row, 3], f"{name} -- Intl, EVAL")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
