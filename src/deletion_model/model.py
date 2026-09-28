import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures

warnings.filterwarnings("ignore", message="Bins whose width are too small")

from deletion_model.features import (
    ALL_CROSS_PAIRS,
    CROSS_INPUT_COLUMNS,
    INTERACTION_FEATURES,
    NUMERIC_FEATURES,
)

DISCRETIZER_STRATEGIES = ("quantile", "uniform", "kmeans")


def _disc_kwargs(strategy: str) -> dict:
    if strategy == "quantile":
        return {"quantile_method": "averaged_inverted_cdf"}
    return {}


class CrossFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, pairs, n_bins=4, strategy="quantile"):
        self.pairs = pairs
        self.n_bins = n_bins
        self.strategy = strategy

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        col_indices = sorted({idx for pair in self.pairs for idx in pair})
        self.discretizers_ = {}
        for col_idx in col_indices:
            disc = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode="ordinal",
                strategy=self.strategy,
                subsample=None,
                **_disc_kwargs(self.strategy),
            )
            try:
                disc.fit(X[:, col_idx : col_idx + 1])
            except ValueError:
                disc = None
            self.discretizers_[col_idx] = disc
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_bins_sq = self.n_bins**2

        binned = {}
        for col_idx, disc in self.discretizers_.items():
            if disc is None:
                binned[col_idx] = np.zeros(n_samples, dtype=int)
            else:
                vals = disc.transform(X[:, col_idx : col_idx + 1])
                binned[col_idx] = vals.ravel().astype(int)

        result = np.zeros((n_samples, len(self.pairs) * n_bins_sq))
        rows = np.arange(n_samples)
        for pair_idx, (col_a, col_b) in enumerate(self.pairs):
            combined = binned[col_a] * self.n_bins + binned[col_b]
            result[rows, pair_idx * n_bins_sq + combined] = 1.0

        return result


def build_model(
    n_bins_fine: int = 8,
    n_bins_coarse: int = 4,
    n_bins_cross: int = 4,
    C: float = 0.001,
    class_weight: str | dict | None = "balanced",
    select_k: int | None = 1000,
    discretizer_strategy: str = "quantile",
) -> Pipeline:
    if discretizer_strategy not in DISCRETIZER_STRATEGIES:
        raise ValueError(
            f"discretizer_strategy={discretizer_strategy!r} not in "
            f"{DISCRETIZER_STRATEGIES}"
        )

    fine_branch = Pipeline(
        [
            (
                "discretize",
                KBinsDiscretizer(
                    n_bins=n_bins_fine,
                    encode="onehot-dense",
                    strategy=discretizer_strategy,
                    **_disc_kwargs(discretizer_strategy),
                ),
            ),
        ]
    )

    coarse_branch = Pipeline(
        [
            (
                "discretize",
                KBinsDiscretizer(
                    n_bins=n_bins_coarse,
                    encode="onehot-dense",
                    strategy=discretizer_strategy,
                    **_disc_kwargs(discretizer_strategy),
                ),
            ),
            (
                "poly",
                PolynomialFeatures(
                    degree=2,
                    interaction_only=True,
                    include_bias=False,
                ),
            ),
        ]
    )

    cross_branch = CrossFeatureTransformer(
        pairs=ALL_CROSS_PAIRS,
        n_bins=n_bins_cross,
        strategy=discretizer_strategy,
    )

    preprocessor = ColumnTransformer(
        [
            ("fine", fine_branch, NUMERIC_FEATURES),
            ("coarse", coarse_branch, INTERACTION_FEATURES),
            ("crosses", cross_branch, CROSS_INPUT_COLUMNS),
        ]
    )

    steps = [("preprocessor", preprocessor)]

    if select_k is not None:
        steps.append(("drop_constant", VarianceThreshold(threshold=0)))
        steps.append(("select", SelectKBest(f_classif, k=select_k)))

    steps.append(
        (
            "classifier",
            LogisticRegression(
                C=C,
                class_weight=class_weight,
                solver="liblinear",
                max_iter=1000,
            ),
        )
    )

    return Pipeline(steps)
