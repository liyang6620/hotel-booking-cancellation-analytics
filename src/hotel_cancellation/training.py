"""Temporal model selection, calibration, evaluation, and artifact generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .calibration import PlattCalibrator
from .config import (
    ARTIFACTS_DIR,
    CALIBRATOR_PATH,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    DIRECT_LEAKAGE_COLUMNS,
    FEATURE_CONTRACT_PATH,
    FIGURES_DIR,
    METADATA_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    POST_BOOKING_COLUMNS,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TABLES_DIR,
    TARGET,
    TRAIN_END,
    VALIDATION_END,
)
from .features import arrival_dates, build_features


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f2: float
    balanced_accuracy: float
    specificity: float
    false_positives: int
    false_negatives: int


def _preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=25,
                    sparse_output=True,
                ),
            ),
        ]
    )
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ("numeric", numeric, list(NUMERIC_FEATURES)),
        ],
        verbose_feature_names_out=True,
    )


def _candidate_estimators() -> dict[str, object]:
    return {
        "Logistic regression": LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost (shallow)": XGBClassifier(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.04,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost (regularised)": XGBClassifier(
            n_estimators=450,
            max_depth=2,
            learning_rate=0.035,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.15,
            reg_lambda=5.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def _pipeline(estimator: object) -> Pipeline:
    return Pipeline([("preprocess", _preprocessor()), ("model", estimator)])


def _temporal_folds(dates: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    boundaries = (
        ("2015-12-01", "2016-02-01"),
        ("2016-02-01", "2016-04-01"),
        ("2016-04-01", "2016-06-01"),
        ("2016-06-01", TRAIN_END),
    )
    folds = []
    for validation_start, validation_end in boundaries:
        train_indices = np.flatnonzero(dates.lt(validation_start).to_numpy())
        validation_indices = np.flatnonzero(
            (dates.ge(validation_start) & dates.lt(validation_end)).to_numpy()
        )
        folds.append((train_indices, validation_indices))
    return folds


def _compare_models(
    features: pd.DataFrame, labels: pd.Series, dates: pd.Series
) -> tuple[pd.DataFrame, str, Pipeline]:
    rows: list[dict[str, float | str]] = []
    candidates = _candidate_estimators()
    for model_name, estimator in candidates.items():
        fold_scores = []
        for fold_number, (train_indices, validation_indices) in enumerate(
            _temporal_folds(dates), start=1
        ):
            candidate = _pipeline(clone(estimator))
            candidate.fit(features.iloc[train_indices], labels.iloc[train_indices])
            probabilities = candidate.predict_proba(features.iloc[validation_indices])[:, 1]
            y_fold = labels.iloc[validation_indices]
            fold_scores.append(
                {
                    "fold": fold_number,
                    "roc_auc": roc_auc_score(y_fold, probabilities),
                    "pr_auc": average_precision_score(y_fold, probabilities),
                    "brier": brier_score_loss(y_fold, probabilities),
                }
            )
        fold_frame = pd.DataFrame(fold_scores)
        rows.append(
            {
                "model": model_name,
                "temporal_roc_auc_mean": fold_frame["roc_auc"].mean(),
                "temporal_roc_auc_std": fold_frame["roc_auc"].std(ddof=1),
                "temporal_pr_auc_mean": fold_frame["pr_auc"].mean(),
                "temporal_pr_auc_std": fold_frame["pr_auc"].std(ddof=1),
                "temporal_brier_mean": fold_frame["brier"].mean(),
                "temporal_brier_std": fold_frame["brier"].std(ddof=1),
            }
        )
    comparison = pd.DataFrame(rows).sort_values(
        ["temporal_pr_auc_mean", "temporal_roc_auc_mean"], ascending=False
    )
    selected_name = str(comparison.iloc[0]["model"])
    selected_estimator = candidates[selected_name]
    if not isinstance(selected_estimator, XGBClassifier):
        raise RuntimeError(
            f"{selected_name} won temporal validation; update model persistence before release."
        )
    return comparison, selected_name, _pipeline(selected_estimator)


def _choose_threshold(labels: pd.Series, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.03, 0.8, 300)
    truth = labels.to_numpy()
    return float(
        min(
            candidates,
            key=lambda threshold: (
                4 * np.sum((probabilities < threshold) & (truth == 1))
                + np.sum((probabilities >= threshold) & (truth == 0)),
                threshold,
            ),
        )
    )


def _threshold_metrics(
    labels: pd.Series, probabilities: np.ndarray, threshold: float
) -> ThresholdMetrics:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return ThresholdMetrics(
        threshold=threshold,
        precision=precision_score(labels, predictions, zero_division=0),
        recall=recall_score(labels, predictions, zero_division=0),
        f2=fbeta_score(labels, predictions, beta=2, zero_division=0),
        balanced_accuracy=balanced_accuracy_score(labels, predictions),
        specificity=tn / (tn + fp),
        false_positives=int(fp),
        false_negatives=int(fn),
    )


def _bootstrap_interval(
    labels: pd.Series, probabilities: np.ndarray, metric, repeats: int = 500
) -> list[float]:
    rng = np.random.default_rng(RANDOM_STATE)
    truth = labels.to_numpy()
    values: list[float] = []
    for _ in range(repeats):
        indices = rng.integers(0, len(truth), len(truth))
        if np.unique(truth[indices]).size == 2:
            values.append(float(metric(truth[indices], probabilities[indices])))
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _shap_table(
    pipeline: Pipeline, features: pd.DataFrame, sample_size: int = 5_000
) -> pd.DataFrame:
    sample = features.sample(min(sample_size, len(features)), random_state=RANDOM_STATE)
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(sample)
    feature_names = list(preprocessor.get_feature_names_out())
    contributions = model.get_booster().predict(
        xgb.DMatrix(transformed, feature_names=feature_names), pred_contribs=True
    )[:, :-1]
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(contributions).mean(axis=0),
                "mean_shap": contributions.mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )


def _monthly_stability(
    dates: pd.Series, labels: pd.Series, probabilities: np.ndarray
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"month": dates.dt.to_period("M").astype(str), "label": labels, "score": probabilities}
    )
    rows = []
    for month, group in frame.groupby("month"):
        if group["label"].nunique() < 2:
            continue
        rows.append(
            {
                "month": month,
                "bookings": len(group),
                "cancellation_rate": group["label"].mean(),
                "roc_auc": roc_auc_score(group["label"], group["score"]),
                "pr_auc": average_precision_score(group["label"], group["score"]),
                "brier": brier_score_loss(group["label"], group["score"]),
            }
        )
    return pd.DataFrame(rows)


def _save_figures(
    comparison: pd.DataFrame,
    labels: pd.Series,
    raw_probabilities: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    shap_table: pd.DataFrame,
    monthly: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    ordered = comparison.sort_values("temporal_pr_auc_mean")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(
        ordered["model"],
        ordered["temporal_pr_auc_mean"],
        xerr=ordered["temporal_pr_auc_std"],
        color="#176b5b",
    )
    ax.set(
        xlabel="Expanding-window PR-AUC (mean +/- 1 SD)",
        xlim=(0.3, 1),
        title="Temporal model selection",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model-comparison.png", dpi=180)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].plot(fpr, tpr, color="#176b5b", lw=2)
    axes[0].plot([0, 1], [0, 1], "--", color="#8b9691")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="2017 ROC")
    axes[1].plot(recall, precision, color="#c4573f", lw=2)
    axes[1].axhline(labels.mean(), ls="--", color="#8b9691")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="2017 precision-recall")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "test-curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for name, scores, colour in (
        ("Raw XGBoost", raw_probabilities, "#8b9691"),
        ("Platt calibrated", probabilities, "#176b5b"),
    ):
        observed, predicted = calibration_curve(labels, scores, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=name, color=colour)
    ax.plot([0, 1], [0, 1], "--", color="#c4573f", label="Perfect calibration")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed cancellation rate",
        title="Probability calibration on future bookings",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=180)
    plt.close(fig)

    predictions = (probabilities >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        display_labels=["Kept", "Cancelled"],
        cmap="Greens",
        ax=ax,
    )
    ax.set_title(f"2017 confusion matrix at threshold {threshold:.3f}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion-matrix.png", dpi=180)
    plt.close(fig)

    ordered_shap = shap_table.head(15).sort_values("mean_abs_shap").copy()
    ordered_shap["display_feature"] = (
        ordered_shap["feature"]
        .str.replace(r"^(categorical|numeric)__", "", regex=True)
        .str.replace("_", " ")
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.barh(
        ordered_shap["display_feature"],
        ordered_shap["mean_abs_shap"],
        color="#176b5b",
    )
    ax.set(xlabel="Mean absolute SHAP value", title="Global influence on 2017 predictions")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature-importance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    axes[0].plot(monthly["month"], monthly["roc_auc"], marker="o", label="ROC-AUC")
    axes[0].plot(monthly["month"], monthly["pr_auc"], marker="o", label="PR-AUC")
    axes[0].set(ylabel="Discrimination", ylim=(0.5, 1))
    axes[0].legend()
    axes[1].plot(monthly["month"], monthly["brier"], marker="o", color="#c4573f")
    axes[1].set(xlabel="2017 arrival month", ylabel="Brier score")
    for axis in axes:
        axis.tick_params(axis="x", rotation=45)
    fig.suptitle("Out-of-time performance stability")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "temporal-stability.png", dpi=180)
    plt.close(fig)


def train_and_evaluate() -> dict[str, object]:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    raw_source = pd.read_csv(DATA_PATH)
    source_rows = len(raw_source)
    raw = raw_source.reset_index(names="source_row").drop_duplicates(
        subset=list(raw_source.columns), keep="first"
    )
    duplicate_rows_removed = source_rows - len(raw)
    dates = arrival_dates(raw)
    raw = raw.assign(arrival_date=dates).sort_values("arrival_date").reset_index(drop=True)
    dates = raw["arrival_date"]
    labels = raw[TARGET].astype(int)

    train_mask = dates.lt(TRAIN_END)
    validation_mask = dates.ge(TRAIN_END) & dates.lt(VALIDATION_END)
    test_mask = dates.ge(VALIDATION_END)
    train_raw = raw.loc[train_mask]
    validation_raw = raw.loc[validation_mask]
    test_raw = raw.loc[test_mask]
    train_features = build_features(train_raw)
    validation_features = build_features(validation_raw)
    test_features = build_features(test_raw)
    train_labels = labels.loc[train_mask].reset_index(drop=True)
    validation_labels = labels.loc[validation_mask].reset_index(drop=True)
    test_labels = labels.loc[test_mask].reset_index(drop=True)
    train_features = train_features.reset_index(drop=True)
    validation_features = validation_features.reset_index(drop=True)
    test_features = test_features.reset_index(drop=True)
    train_dates = dates.loc[train_mask].reset_index(drop=True)
    test_dates = dates.loc[test_mask].reset_index(drop=True)

    comparison, selected_name, selected_pipeline = _compare_models(
        train_features, train_labels, train_dates
    )
    selected_pipeline.fit(train_features, train_labels)
    raw_validation = selected_pipeline.predict_proba(validation_features)[:, 1]
    calibrator = PlattCalibrator().fit(raw_validation, validation_labels.to_numpy())
    validation_probabilities = calibrator.predict(raw_validation)
    threshold = _choose_threshold(validation_labels, validation_probabilities)

    raw_test_probabilities = selected_pipeline.predict_proba(test_features)[:, 1]
    test_probabilities = calibrator.predict(raw_test_probabilities)
    threshold_metrics = _threshold_metrics(test_labels, test_probabilities, threshold)
    test_metrics = {
        "roc_auc": float(roc_auc_score(test_labels, test_probabilities)),
        "roc_auc_95_ci": _bootstrap_interval(test_labels, test_probabilities, roc_auc_score),
        "pr_auc": float(average_precision_score(test_labels, test_probabilities)),
        "pr_auc_95_ci": _bootstrap_interval(
            test_labels, test_probabilities, average_precision_score
        ),
        "brier_raw": float(brier_score_loss(test_labels, raw_test_probabilities)),
        "brier_calibrated": float(brier_score_loss(test_labels, test_probabilities)),
        "log_loss": float(log_loss(test_labels, test_probabilities)),
        **asdict(threshold_metrics),
    }

    shap_table = _shap_table(selected_pipeline, test_features)
    monthly = _monthly_stability(test_dates, test_labels, test_probabilities)

    preprocessor = selected_pipeline.named_steps["preprocess"]
    model = selected_pipeline.named_steps["model"]
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    model.save_model(MODEL_PATH)
    CALIBRATOR_PATH.write_text(
        json.dumps(calibrator.to_dict(), indent=2), encoding="utf-8"
    )
    FEATURE_CONTRACT_PATH.write_text(
        json.dumps(
            {
                "prediction_time": "booking creation",
                "categorical_features": list(CATEGORICAL_FEATURES),
                "numeric_features": list(NUMERIC_FEATURES),
                "direct_leakage_columns": list(DIRECT_LEAKAGE_COLUMNS),
                "post_booking_columns": list(POST_BOOKING_COLUMNS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    comparison.to_csv(TABLES_DIR / "model-comparison.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(TABLES_DIR / "test-metrics.csv", index=False)
    shap_table.to_csv(TABLES_DIR / "top-features.csv", index=False)
    monthly.to_csv(TABLES_DIR / "monthly-stability.csv", index=False)
    split_manifest = raw[["source_row", "arrival_date", TARGET]].copy()
    split_manifest["split"] = np.select(
        [train_mask, validation_mask, test_mask],
        ["train", "validation", "test"],
        default="unknown",
    )
    split_manifest.to_csv(ARTIFACTS_DIR / "split_assignments.csv", index=False)
    regression = test_raw[["source_row"]].copy()
    regression["cancellation_probability"] = test_probabilities
    regression.head(100).to_csv(ARTIFACTS_DIR / "score_regression.csv", index=False)

    metadata: dict[str, object] = {
        "model_version": "2.0.0",
        "prediction_time": "booking creation",
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "source_rows": source_rows,
        "exact_duplicate_rows_removed": duplicate_rows_removed,
        "model_rows": len(raw),
        "split": {
            "train_rows": len(train_raw),
            "validation_rows": len(validation_raw),
            "test_rows": len(test_raw),
            "train_period": [str(train_dates.min().date()), "2016-08-31"],
            "validation_period": ["2016-09-01", "2016-12-31"],
            "test_period": ["2017-01-01", str(test_dates.max().date())],
        },
        "selection": {
            "method": "four expanding-window temporal folds on training data",
            "primary_metric": "PR-AUC",
            "selected_model": selected_name,
            "candidates": comparison.to_dict(orient="records"),
        },
        "calibration": "Platt scaling fitted on the future validation period",
        "threshold_policy": "Validation cost: false negative = 4, false positive = 1",
        "test_metrics": test_metrics,
        "top_features": shap_table.to_dict(orient="records"),
        "direct_leakage_columns": list(DIRECT_LEAKAGE_COLUMNS),
        "post_booking_columns": list(POST_BOOKING_COLUMNS),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _save_figures(
        comparison,
        test_labels,
        raw_test_probabilities,
        test_probabilities,
        threshold,
        shap_table,
        monthly,
    )
    return metadata


if __name__ == "__main__":
    result = train_and_evaluate()
    print(json.dumps(result["test_metrics"], indent=2))
