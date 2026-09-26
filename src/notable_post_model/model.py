from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from notable_post_model import features


class NotablePostMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class EmbeddingCache:
    ttl_seconds: float = 5 * 3600
    _entries: dict[int, tuple[float, np.ndarray]] = field(
        default_factory=dict, repr=False
    )
    _last_purge: float = field(default=0.0, repr=False)
    _PURGE_INTERVAL: float = 300.0

    def get(self, post_id: int) -> np.ndarray | None:
        entry = self._entries.get(post_id)
        if entry is None:
            return None
        ts, vec = entry
        if time.monotonic() - ts > self.ttl_seconds:
            del self._entries[post_id]
            return None
        return vec

    def put(self, post_id: int, embedding: np.ndarray) -> None:
        now = time.monotonic()
        self._entries[post_id] = (now, embedding)
        if now - self._last_purge > self._PURGE_INTERVAL:
            self._purge(now)

    def _purge(self, now: float) -> None:
        expired = [
            k for k, (ts, _) in self._entries.items() if now - ts > self.ttl_seconds
        ]
        for k in expired:
            del self._entries[k]
        self._last_purge = now

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class NotablePostModelBundle:
    mlp: NotablePostMLP
    scaler: StandardScaler
    ohe: OneHotEncoder
    embedding_model: object
    config: dict
    embedding_cache: EmbeddingCache = field(default_factory=EmbeddingCache)


def build_tabular_features(
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray | None, int, StandardScaler, OneHotEncoder]:
    num_cols = features.NUMERIC_FEATURES
    cat_cols = features.CATEGORICAL_FEATURES

    scaler = StandardScaler()
    num_train = scaler.fit_transform(X_train[num_cols].astype("float64").fillna(0))

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_train = ohe.fit_transform(X_train[cat_cols].astype(str).fillna("missing"))

    train_arr = np.hstack([num_train, cat_train]).astype(np.float32)
    tabular_dim = train_arr.shape[1]

    eval_arr = None
    if X_eval is not None:
        num_eval = scaler.transform(X_eval[num_cols].astype("float64").fillna(0))
        cat_eval = ohe.transform(X_eval[cat_cols].astype(str).fillna("missing"))
        eval_arr = np.hstack([num_eval, cat_eval]).astype(np.float32)

    n_num = num_train.shape[1]
    n_cat = cat_train.shape[1]
    print(
        f"  Tabular dims: {n_num} numeric + {n_cat} one-hot categorical "
        f"= {tabular_dim} total"
    )
    return train_arr, eval_arr, tabular_dim, scaler, ohe


def apply_tabular_preprocessing(
    X: pd.DataFrame,
    scaler: StandardScaler,
    ohe: OneHotEncoder,
) -> np.ndarray:
    num_cols = features.NUMERIC_FEATURES
    cat_cols = features.CATEGORICAL_FEATURES
    num = scaler.transform(X[num_cols].astype("float64").fillna(0))
    cat = ohe.transform(X[cat_cols].astype(str).fillna("missing"))
    return np.hstack([num, cat]).astype(np.float32)


_CONFIG_FILE = "config.json"
_MODEL_FILE = "model.pt"
_PREPROCESSING_FILE = "preprocessing.joblib"


def save_model_bundle(
    mlp: NotablePostMLP,
    scaler: StandardScaler,
    ohe: OneHotEncoder,
    config: dict,
    out_dir: Path | str,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / _CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2, default=str)
    torch.save(mlp.state_dict(), out_dir / _MODEL_FILE)
    joblib.dump({"scaler": scaler, "ohe": ohe}, out_dir / _PREPROCESSING_FILE)
    print(f"  Saved model bundle to {out_dir}")


def load_model_bundle(
    model_dir: str | Path,
    embedding_model_override: str | None = None,
) -> NotablePostModelBundle:
    from sentence_transformers import SentenceTransformer

    model_dir = Path(model_dir)

    with open(model_dir / _CONFIG_FILE) as f:
        config = json.load(f)

    preprocessing = joblib.load(model_dir / _PREPROCESSING_FILE)
    scaler = preprocessing["scaler"]
    ohe = preprocessing["ohe"]

    input_dim = config["tabular_dim"] + config["embedding_dim"]
    mlp = NotablePostMLP(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        dropout=config.get("dropout", 0.1),
    )
    mlp.load_state_dict(
        torch.load(model_dir / _MODEL_FILE, map_location="cpu", weights_only=True)
    )
    mlp.eval()

    emb_model_name = embedding_model_override or config["embedding_model"]
    print(f"  Loading embedding model: {emb_model_name}")
    embedding_model = SentenceTransformer(emb_model_name)

    return NotablePostModelBundle(
        mlp=mlp,
        scaler=scaler,
        ohe=ohe,
        embedding_model=embedding_model,
        config=config,
    )
