from __future__ import annotations

import argparse
import itertools
import json
import random
import warnings
from pathlib import Path

from notable_post_model import train
from notable_post_model.model import (
    NotablePostMLP,
    build_tabular_features,
    save_model_bundle,
)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _default_sweep_grid() -> list[dict]:
    return [
        dict(hidden_dim=h, lr=lr, dropout=d, weight_decay=wd, pos_weight_ratio=pw)
        for h, lr, d, wd, pw in itertools.product(
            [128, 256, 384],
            [1e-4, 3e-4, 1e-3, 3e-3],
            [0.0, 0.05, 0.1, 0.2],
            [3e-4, 1e-3, 3e-3],
            [7.0, 10.0, 15.0, 25.0],
        )
    ]


def _parse_class_weight(s: str) -> float:
    s = s.strip()
    if s.lower() == "none":
        return 1.0
    if ":" in s:
        parts = s.split(":")
        return float(parts[0]) / float(parts[1])
    raise argparse.ArgumentTypeError(f"unrecognized class_weight: {s!r}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--feeds", required=True, help="path to the consolidated feed parquet"
    )
    p.add_argument(
        "--notes", required=True, help="path to combined_notes_by_minute_parquet"
    )
    p.add_argument(
        "--nsh", required=True, help="path to birdwatch_note_status_history parquet"
    )
    p.add_argument("--out-dir", required=True)

    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--prune-recent-hours", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="HuggingFace sentence-transformers model name "
        "(default: all-mpnet-base-v2)",
    )
    p.add_argument("--embedding-batch-size", type=int, default=256)
    p.add_argument(
        "--embedding-cache",
        type=str,
        default=None,
        help="path to save/load precomputed embeddings (.npz)",
    )

    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument(
        "--class-weight",
        type=_parse_class_weight,
        default=15.0,
        help="CRH upweight ratio as 'w_crh:w_ncrh' (e.g. '15:1', "
        "default) or 'none' for equal weighting",
    )

    p.add_argument(
        "--sweep-mode",
        choices=["grid", "random"],
        default=None,
        help="if set, run a hparam sweep before training",
    )
    p.add_argument(
        "--sweep-size",
        type=int,
        default=50,
        help="number of random configs to sample when --sweep-mode random (default 50)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = train.select_device()
    print(f"Device: {device}")

    print("=" * 72)
    print("Loading data and building features")
    print("=" * 72)
    print(f"  feeds:              {args.feeds}")
    print(f"  notes:              {args.notes}")
    print(f"  nsh:                {args.nsh}")
    print(f"  prune_recent_hours: {args.prune_recent_hours}")
    X_tr, y_tr, meta_tr, text_tr, X_ev, y_ev, meta_ev, text_ev = train.prepare_data(
        feeds_path=args.feeds,
        notes_path=args.notes,
        nsh_path=args.nsh,
        test_size=args.test_size,
        prune_recent_hours=args.prune_recent_hours,
    )
    print(f"  train: n={len(y_tr):,}  CRH={int((y_tr == 0).sum()):,}")
    print(f"  eval:  n={len(y_ev):,}  CRH={int((y_ev == 0).sum()):,}")

    print()
    print("=" * 72)
    print("Computing embeddings")
    print("=" * 72)

    all_texts = pd.concat([text_tr, text_ev], ignore_index=True)
    preprocessed = [train.preprocess_text(t) for t in all_texts]
    n_empty = sum(1 for t in preprocessed if t == "")
    print(f"  Preprocessed {len(preprocessed):,} texts ({n_empty:,} empty)")

    embeddings = train.compute_embeddings(
        texts=preprocessed,
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
        device=device,
        cache_path=args.embedding_cache,
    )
    n_emb_dims = embeddings.shape[1]
    print(f"  Embedding shape: {embeddings.shape}")

    emb_train = embeddings[: len(y_tr)].astype(np.float32)
    emb_eval = embeddings[len(y_tr) :].astype(np.float32)

    print()
    print("=" * 72)
    print("Building tabular features")
    print("=" * 72)

    tab_train, tab_eval, tab_dim, scaler, ohe = build_tabular_features(X_tr, X_ev)

    combined_train = np.hstack([tab_train, emb_train])
    combined_eval = np.hstack([tab_eval, emb_eval])
    combined_dim = tab_dim + n_emb_dims

    print(
        f"  Combined dim: {tab_dim} tabular + {n_emb_dims} embedding = {combined_dim}"
    )

    if args.sweep_mode is not None:
        full_grid = _default_sweep_grid()
        if args.sweep_mode == "grid":
            grid = full_grid
            print(f"\n  Sweep mode: grid ({len(grid)} configs)")
        else:
            sweep_size = min(args.sweep_size, len(full_grid))
            grid = random.Random(args.seed).sample(full_grid, k=sweep_size)
            print(
                f"\n  Sweep mode: random; sampled {sweep_size} of "
                f"{len(full_grid)} configs (seed={args.seed})"
            )
        sweep_fixed = dict(
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
        )
        sweep_results = train.run_mlp_sweep(
            X_train=combined_train,
            y_train=y_tr,
            meta_train=meta_tr,
            input_dim=combined_dim,
            grid=grid,
            fixed=sweep_fixed,
            seed=args.seed,
            device=device,
        )
        sweep_out = [{"score": s, "params": p} for s, p in sweep_results]
        sweep_path = out_dir / "sweep_results.json"
        with open(sweep_path, "w") as fh:
            json.dump(sweep_out, fh, indent=2, default=str)
        print(f"  Sweep results saved: {sweep_path}")

        best_score, best_params = sweep_results[0]
        print(f"  Best config (en_xxl@FPR={best_score:.3f}): {best_params}")
        params = {**sweep_fixed, **best_params}
    else:
        params = dict(
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            pos_weight_ratio=args.class_weight,
            seed=args.seed,
        )

    print()
    print("=" * 72)
    print("Training final model (combined features)")
    print(f"Params: {params}")
    print("=" * 72)

    mlp, comb_scores_train, comb_scores_eval = train.train_mlp(
        X_train=combined_train,
        y_train=y_tr,
        X_eval=combined_eval,
        y_eval=y_ev,
        meta_eval=meta_ev,
        input_dim=combined_dim,
        device=device,
        **params,
    )
    train.print_metrics("Combined", y_tr, comb_scores_train, split="train")
    train.print_metrics("Combined", y_ev, comb_scores_eval, split="eval")
    en_xxl_train = train.en_xxl_recall_at_fpr(
        y_tr, comb_scores_train, meta_tr["api_feed"]
    )
    en_xxl_eval = train.en_xxl_recall_at_fpr(
        y_ev, comb_scores_eval, meta_ev["api_feed"]
    )
    print(f"  en_xxl recall @ CRH FPR<={train.FPR_CAP} (train): {en_xxl_train:.3f}")
    print(f"  en_xxl recall @ CRH FPR<={train.FPR_CAP} (eval):  {en_xxl_eval:.3f}")

    print()
    print("=" * 72)
    print("Ablation: text-only features")
    print("=" * 72)
    _, text_scores_train, text_scores_eval = train.train_mlp(
        X_train=emb_train,
        y_train=y_tr,
        X_eval=emb_eval,
        y_eval=y_ev,
        meta_eval=meta_ev,
        input_dim=n_emb_dims,
        device=device,
        **params,
    )

    print()
    print("=" * 72)
    print("Ablation: tabular-only features")
    print("=" * 72)
    _, tab_scores_train, tab_scores_eval = train.train_mlp(
        X_train=tab_train,
        y_train=y_tr,
        X_eval=tab_eval,
        y_eval=y_ev,
        meta_eval=meta_ev,
        input_dim=tab_dim,
        device=device,
        **params,
    )

    config = {
        "embedding_model": args.embedding_model,
        "embedding_dim": n_emb_dims,
        "tabular_dim": tab_dim,
        "hidden_dim": params["hidden_dim"],
        "dropout": params["dropout"],
    }
    save_model_bundle(mlp, scaler, ohe, config, out_dir)

    plot_path = out_dir / "notable_post_roc.png"
    train.plot_ablation_roc(
        ablation_results=[
            (
                "Combined (tabular + text)",
                y_tr,
                comb_scores_train,
                meta_tr["api_feed"],
                y_ev,
                comb_scores_eval,
                meta_ev["api_feed"],
            ),
            (
                "Text only",
                y_tr,
                text_scores_train,
                meta_tr["api_feed"],
                y_ev,
                text_scores_eval,
                meta_ev["api_feed"],
            ),
            (
                "Tabular only",
                y_tr,
                tab_scores_train,
                meta_tr["api_feed"],
                y_ev,
                tab_scores_eval,
                meta_ev["api_feed"],
            ),
        ],
        out_path=plot_path,
    )
    print(f"  Saved ROC plot: {plot_path}")

    fpr_levels = [0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.40, 0.60, 0.80]
    summary = {
        "params": params,
        "embedding_model": args.embedding_model,
        "metrics": {
            "train_auc": float(roc_auc_score(y_tr, comb_scores_train)),
            "eval_auc": float(roc_auc_score(y_ev, comb_scores_eval)),
            "train_en_xxl_recall_at_fpr": en_xxl_train,
            "eval_en_xxl_recall_at_fpr": en_xxl_eval,
            "thresholds_at_crh_loss_rate": {
                "en": train._thresholds_at_fpr(
                    y_ev, comb_scores_eval, meta_ev["api_feed"], "en", fpr_levels
                ),
                "intl": train._thresholds_at_fpr(
                    y_ev, comb_scores_eval, meta_ev["api_feed"], "intl", fpr_levels
                ),
            },
        },
    }
    summary_path = out_dir / "notable_post_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"  Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
