from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from data_models.writer_data_models import AuthorHistory, PostWithContext
from notable_post_model.features import (
    HISTORY_COLUMNS,
    compute_author_history_from_cache,
    compute_model_features,
)
from notable_post_model.model import (
    NotablePostModelBundle,
    apply_tabular_preprocessing,
    load_model_bundle,
)
from notable_post_model.train import preprocess_text


def load_notable_post_model(
    model_dir: str,
    embedding_model: str | None = None,
) -> NotablePostModelBundle:
    return load_model_bundle(model_dir, embedding_model_override=embedding_model)


def _posts_to_raw_df(posts: list[PostWithContext]) -> pd.DataFrame:
    rows = []
    for p in posts:
        pm = p.post.public_metrics
        apm = p.post.author_public_metrics
        if pm is None:
            raise ValueError(f"Post {p.post.post_id} is missing public_metrics")
        if apm is None:
            raise ValueError(f"Post {p.post.post_id} is missing author_public_metrics")
        if p.post.author_verified_type is None:
            raise ValueError(f"Post {p.post.post_id} is missing author_verified_type")
        if p.post.author_parody is None:
            raise ValueError(f"Post {p.post.post_id} is missing author_parody")
        if p.post.lang is None:
            raise ValueError(f"Post {p.post.post_id} is missing lang")
        rows.append(
            {
                "post_id": p.post.post_id,
                "enqueued_at": p.enqueued_at,
                "author_id": p.post.author_id,
                "post_text": p.post.text,
                "retweet_count": pm.retweet_count,
                "reply_count": pm.reply_count,
                "like_count": pm.like_count,
                "quote_count": pm.quote_count,
                "bookmark_count": pm.bookmark_count,
                "impression_count": pm.impression_count,
                "author_followers_count": apm.followers_count,
                "author_following_count": apm.following_count,
                "author_tweet_count": apm.tweet_count,
                "author_listed_count": apm.listed_count,
                "author_like_count": apm.like_count,
                "author_media_count": apm.media_count,
                "author_verified_type": p.post.author_verified_type,
                "author_parody": p.post.author_parody,
                "lang": p.post.lang,
                "num_unique_sources": len(p.suggested_sources),
                "total_source_suggestions": sum(s.count for s in p.suggested_sources),
                "has_photo": any(m.media_type == "photo" for m in p.post.media),
                "has_video": any(m.media_type == "video" for m in p.post.media),
            }
        )
    return pd.DataFrame(rows)


def compute_notable_post_predictions(
    bundle: NotablePostModelBundle,
    posts: list[PostWithContext],
    cache_df: pd.DataFrame,
) -> None:
    if not posts:
        return

    raw_df = _posts_to_raw_df(posts)

    author_hist = compute_author_history_from_cache(
        cache_df,
        author_ids=raw_df["author_id"],
    )
    raw_df = raw_df.merge(author_hist, on="author_id", how="left")
    for col in HISTORY_COLUMNS:
        raw_df[col] = raw_df[col].fillna(0)

    features_df = compute_model_features(raw_df)

    tabular_arr = apply_tabular_preprocessing(features_df, bundle.scaler, bundle.ohe)

    post_ids = raw_df["post_id"].tolist()
    texts = raw_df["post_text"].tolist()
    n = len(post_ids)
    cache = bundle.embedding_cache

    uncached_indices = [i for i in range(n) if cache.get(post_ids[i]) is None]

    if uncached_indices:
        uncached_texts = [preprocess_text(texts[i]) for i in uncached_indices]
        new_embeddings = bundle.embedding_model.encode(
            uncached_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            device="cpu",
        ).astype(np.float32)
        for j, idx in enumerate(uncached_indices):
            cache.put(post_ids[idx], new_embeddings[j])

    emb_dim = bundle.config["embedding_dim"]
    text_embeddings = np.empty((n, emb_dim), dtype=np.float32)
    for i in range(n):
        text_embeddings[i] = cache.get(post_ids[i])

    combined = np.hstack([tabular_arr, text_embeddings])
    combined_t = torch.from_numpy(combined)

    bundle.mlp.eval()
    with torch.no_grad():
        logits = bundle.mlp(combined_t)
        raw_scores = torch.sigmoid(logits).numpy()

    reversed_scores = 1.0 - raw_scores

    for i, (post, score) in enumerate(zip(posts, reversed_scores)):
        post.notable_post_prediction = float(score)
        post.author_history = AuthorHistory(
            hist_note_count=int(raw_df.at[i, "hist_note_count"]),
            hist_crh_count=int(raw_df.at[i, "hist_crh_count"]),
            hist_crnh_count=int(raw_df.at[i, "hist_crnh_count"]),
            hist_total_ratings=int(raw_df.at[i, "hist_total_ratings"]),
        )
