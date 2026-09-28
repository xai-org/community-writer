from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import dotenv

from deletion_model.model import DISCRETIZER_STRATEGIES, build_model
from deletion_model.train import (
    generate_training_instances,
    load_and_prepare_data,
    prepare_features_and_labels,
    run_sweep,
    split_data,
    train_and_evaluate,
)


SWEEP_DIMENSIONS = ("n_bins_fine", "C", "select_k", "discretizer_strategy")


def _parse_optional_int(s: str):
    if s is None:
        return None
    s = s.strip()
    if s.lower() in ("none", "null"):
        return None
    return int(s)


def _default_sweep_grid() -> list[dict]:
    return [
        {
            "n_bins_fine": 8,
            "C": 0.001,
            "select_k": 1000,
            "discretizer_strategy": strategy,
        }
        for strategy in DISCRETIZER_STRATEGIES
    ]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--status", required=True, help="Path to note status history parquet file"
    )
    p.add_argument(
        "--raters", required=True, help="Path to rater model output parquet file"
    )
    p.add_argument(
        "--ratings", required=True, help="Path to combined ratings parquet directory"
    )
    p.add_argument(
        "--enrollment",
        required=True,
        help="Path to the user enrollment TSV (headerless; used to "
        "derive the other-bot author pool)",
    )
    p.add_argument(
        "--out-dir", required=True, help="Directory to write model artifacts to"
    )

    p.add_argument(
        "--C",
        type=float,
        default=0.001,
        help="Inverse L2 regularization strength (default 0.001). "
        "Ignored when --sweep-mode is set.",
    )
    p.add_argument(
        "--n-bins-fine",
        type=int,
        default=8,
        help="Number of fine-grained bins for feature discretization. "
        "Ignored when --sweep-mode is set.",
    )
    p.add_argument(
        "--select-k",
        type=_parse_optional_int,
        default=1000,
        help="Number of top features to select ('none' to skip). "
        "Ignored when --sweep-mode is set.",
    )
    p.add_argument(
        "--discretizer-strategy",
        choices=list(DISCRETIZER_STRATEGIES),
        default="quantile",
        help="KBinsDiscretizer strategy for the fine, coarse, and "
        "cross branches. Ignored when --sweep-mode is set "
        "(the grid sweeps over all strategies).",
    )

    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of notes to hold out for evaluation (random "
        "split grouped by noteId)",
    )

    p.add_argument(
        "--sweep-mode",
        choices=["grid", "random"],
        default=None,
        help="If set, run a hparam sweep before training. 'grid' "
        "tries every config in the default grid. 'random' "
        "samples --sweep-size configs from the grid. If unset "
        "(default), skip the sweep and train using the hparam "
        "values supplied on the CLI.",
    )
    p.add_argument(
        "--sweep-size",
        type=int,
        default=50,
        help="Number of random configs to sample when "
        "--sweep-mode random (ignored otherwise). Capped at "
        "the grid size. Default: 50.",
    )
    p.add_argument(
        "--sweep-cv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set (default), score each sweep config via 5-fold "
        "GroupKFold CV. Pass --no-sweep-cv to score on a "
        "single grouped 1/5 holdout instead (5x faster, but "
        "no fold-to-fold variance).",
    )

    return p.parse_args(argv)


def _load_account_ids_from_env() -> tuple[set[str], set[str]]:
    dotenv.load_dotenv(override=True)
    missing = []
    writer_ids_raw = os.getenv("WRITER_ACCOUNT_IDS")
    if not writer_ids_raw:
        missing.append("WRITER_ACCOUNT_IDS")
    live_writer_id = os.getenv("LIVE_WRITER_ACCOUNT_ID")
    if not live_writer_id:
        missing.append("LIVE_WRITER_ACCOUNT_ID")
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
    writer_ids = {s.strip() for s in writer_ids_raw.split(",") if s.strip()}
    return writer_ids, {live_writer_id.strip()}


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writer_account_ids, excluded_account_ids = _load_account_ids_from_env()

    print("Loading and preparing data...")
    notes, ratings = load_and_prepare_data(
        args.status,
        args.raters,
        args.ratings,
        args.enrollment,
        writer_account_ids=writer_account_ids,
        excluded_account_ids=excluded_account_ids,
        seed=args.seed,
    )
    print(f"  Notes: {len(notes)}, Ratings: {len(ratings)}")

    print("Generating training instances...")
    instances = generate_training_instances(notes, ratings, seed=args.seed)
    print(f"  {len(instances)} instances from {instances['noteId'].nunique()} notes")
    crh = (instances["currentStatus"] == "CURRENTLY_RATED_HELPFUL").sum()
    print(f"  CRH: {crh}, Not CRH: {len(instances) - crh}")

    print("Splitting data...")
    train_inst, eval_inst = split_data(
        instances,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(f"  Train: {len(train_inst)}, Eval: {len(eval_inst)}")

    X_train, y_train = prepare_features_and_labels(train_inst)

    if args.sweep_mode is None:
        params = {
            "n_bins_fine": args.n_bins_fine,
            "C": args.C,
            "select_k": args.select_k,
            "discretizer_strategy": args.discretizer_strategy,
        }
    else:
        full_grid = _default_sweep_grid()
        if args.sweep_mode == "grid":
            grid = full_grid
            print(f"  Sweep mode: grid ({len(grid)} configs)")
        else:
            sweep_size = min(args.sweep_size, len(full_grid))
            grid = random.Random(args.seed).sample(full_grid, k=sweep_size)
            print(
                f"  Sweep mode: random; sampled {sweep_size} of "
                f"{len(full_grid)} configs (seed={args.seed})"
            )
        sweep_results = run_sweep(
            grid=grid,
            fixed_params={},
            X_train=X_train,
            y_train=y_train,
            groups=train_inst["noteId"].values,
            cv=args.sweep_cv,
            seed=args.seed,
        )
        with open(out_dir / "sweep_results.json", "w") as f:
            json.dump(sweep_results, f, indent=2, default=str)
        best = sweep_results[0]
        print(f"\nBest config: {best}")
        params = {k: best[k] for k in SWEEP_DIMENSIONS}

    print(f"\nTraining final model with {params}")
    pipe = build_model(**params)
    all_params = {**params, "seed": args.seed, "test_size": args.test_size}
    train_and_evaluate(pipe, train_inst, eval_inst, out_dir, params=all_params)

    print(f"\nArtifacts saved to {out_dir}/")
    print("  deletion_model.joblib, deletion_roc.png, deletion_summary.json")


if __name__ == "__main__":
    main()
