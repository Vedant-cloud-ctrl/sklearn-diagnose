"""
Deterministic signal extraction for sklearn-diagnose.

This module computes quantitative statistics from the evidence.
All computations are deterministic and reproducible.

Signal extractors are organized by category:
- Performance signals (train/val scores, gaps)
- CV signals (mean, std, fold analysis)
- Residual signals (for regression)
- Classification signals (class distribution, per-class metrics)
- Feature signals (correlations, redundancy)
- Leakage signals (suspicious patterns)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from .schemas import Evidence, Signals, TaskType


def extract_all_signals(evidence: Evidence) -> Signals:
    """
    Extract all signals from the provided evidence.
    
    This is the main entry point for signal extraction.
    
    Args:
        evidence: Evidence object with all diagnostic inputs
        
    Returns:
        Signals object with all computed statistics
    """
    signals = Signals()
    
    # Basic data characteristics
    signals.n_samples_train = evidence.n_samples_train
    signals.n_samples_val = evidence.n_samples_val
    signals.n_features = evidence.n_features
    
    if evidence.n_samples_train > 0 and evidence.n_features > 0:
        signals.feature_to_sample_ratio = evidence.n_features / evidence.n_samples_train
    
    # Extract performance signals
    _extract_performance_signals(evidence, signals)
    
    # Extract CV signals if available
    if evidence.has_cv_results:
        _extract_cv_signals(evidence, signals)
    
    # Extract task-specific signals
    if evidence.task == TaskType.CLASSIFICATION:
        _extract_classification_signals(evidence, signals)
    else:
        _extract_regression_signals(evidence, signals)
    
    # Extract feature signals
    _extract_feature_signals(evidence, signals)
    
    # Extract leakage indicators
    _extract_leakage_signals(evidence, signals)
    
    return signals


def _extract_performance_signals(evidence: Evidence, signals: Signals) -> None:
    """Extract basic performance metrics."""
    
    if evidence.task == TaskType.CLASSIFICATION:
        # Training score
        if evidence.y_pred_train is not None:
            signals.train_score = accuracy_score(evidence.y_train, evidence.y_pred_train)
        
        # Validation score
        if evidence.has_validation_set and evidence.y_pred_val is not None:
            signals.val_score = accuracy_score(evidence.y_val, evidence.y_pred_val)
    
    else:  # Regression
        # Training score (R²)
        if evidence.y_pred_train is not None:
            signals.train_score = r2_score(evidence.y_train, evidence.y_pred_train)
        
        # Validation score
        if evidence.has_validation_set and evidence.y_pred_val is not None:
            signals.val_score = r2_score(evidence.y_val, evidence.y_pred_val)
    
    # Train-val gap
    if signals.train_score is not None and signals.val_score is not None:
        signals.train_val_gap = signals.train_score - signals.val_score


def _extract_cv_signals(evidence: Evidence, signals: Signals) -> None:
    """
    Extract cross-validation signals.
    
    CV interpretation is a core signal extractor within sklearn-diagnose,
    used to detect instability, overfitting, and potential data leakage.
    """
    cv = evidence.cv_results
    
    # Test scores
    if "test_score" in cv:
        test_scores = np.asarray(cv["test_score"])
        signals.cv_fold_scores = test_scores.tolist()
        signals.cv_mean = float(np.mean(test_scores))
        signals.cv_std = float(np.std(test_scores))
        signals.cv_min = float(np.min(test_scores))
        signals.cv_max = float(np.max(test_scores))
        signals.cv_range = signals.cv_max - signals.cv_min
    
    # Train scores (if available)
    if "train_score" in cv:
        train_scores = np.asarray(cv["train_score"])
        signals.cv_train_mean = float(np.mean(train_scores))
        
        # CV train-test gap (overfitting signal)
        if signals.cv_mean is not None:
            signals.cv_train_val_gap = signals.cv_train_mean - signals.cv_mean
    
    # CV vs holdout comparison (leakage signal)
    if signals.cv_mean is not None and signals.val_score is not None:
        signals.cv_holdout_gap = signals.cv_mean - signals.val_score


def _extract_classification_signals(evidence: Evidence, signals: Signals) -> None:
    """Extract classification-specific signals."""
    
    # Class distribution
    unique, counts = np.unique(evidence.y_train, return_counts=True)
    total = len(evidence.y_train)
    signals.class_distribution = {
        str(cls): count / total 
        for cls, count in zip(unique, counts)
    }
    
    # Minority class ratio
    if len(counts) > 1:
        signals.minority_class_ratio = float(np.min(counts) / total)
    
    # Confusion matrix and per-class metrics (if predictions available)
    if evidence.y_pred_val is not None and evidence.y_val is not None:
        try:
            signals.confusion_matrix = confusion_matrix(evidence.y_val, evidence.y_pred_val)
            
            # Per-class recall
            recalls = recall_score(
                evidence.y_val, evidence.y_pred_val, 
                average=None, zero_division=0
            )
            signals.per_class_recall = {
                str(cls): float(rec) 
                for cls, rec in zip(unique, recalls)
            }
            
            # Per-class precision
            precisions = precision_score(
                evidence.y_val, evidence.y_pred_val,
                average=None, zero_division=0
            )
            signals.per_class_precision = {
                str(cls): float(prec)
                for cls, prec in zip(unique, precisions)
            }
        except Exception:
            pass  # Handle edge cases gracefully


def _extract_regression_signals(evidence: Evidence, signals: Signals) -> None:
    """Extract regression-specific signals (residual analysis)."""
    
    if evidence.y_pred_train is None:
        return
    
    # Training residuals
    residuals = evidence.y_train - evidence.y_pred_train
    
    signals.residual_mean = float(np.mean(residuals))
    signals.residual_std = float(np.std(residuals))
    
    # Skewness and kurtosis for residual distribution analysis
    if len(residuals) > 3:
        try:
            signals.residual_skew = float(stats.skew(residuals))
            signals.residual_kurtosis = float(stats.kurtosis(residuals))
        except Exception:
            pass


def _extract_feature_signals(evidence: Evidence, signals: Signals) -> None:
    """Extract feature-level signals."""
    
    X = evidence.X_train
    
    if len(X.shape) != 2 or X.shape[1] < 2:
        return  # Need at least 2 features
    
    try:
        # Feature correlation matrix
        # Handle potential NaN/inf values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Only compute if we have enough variance
        variances = np.var(X_clean, axis=0)
        if np.all(variances > 1e-10):
            corr_matrix = np.corrcoef(X_clean, rowvar=False)
            signals.feature_correlations = corr_matrix
            
            # Find highly correlated feature pairs
            high_corr_pairs = []
            n_features = corr_matrix.shape[0]
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    corr = abs(corr_matrix[i, j])
                    if corr > 0.9:  # Threshold for high correlation
                        high_corr_pairs.append((i, j, float(corr)))
            
            if high_corr_pairs:
                signals.high_correlation_pairs = sorted(
                    high_corr_pairs, 
                    key=lambda x: x[2], 
                    reverse=True
                )
    except Exception:
        pass  # Handle numerical issues gracefully
    
    # Feature-target correlations
    try:
        y = evidence.y_train.astype(float)
        feature_target_corr = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            if not np.isnan(corr):
                feature_target_corr.append(corr)
            else:
                feature_target_corr.append(0.0)
        signals.feature_target_correlations = np.array(feature_target_corr)
    except Exception:
        pass


def _extract_leakage_signals(evidence: Evidence, signals: Signals) -> None:
    """Extract signals that may indicate data leakage."""
    
    # Already computed CV vs holdout gap in CV signals
    
    # Look for suspiciously high feature-target correlations
    if signals.feature_target_correlations is not None:
        suspicious = []
        for i, corr in enumerate(signals.feature_target_correlations):
            if abs(corr) > 0.95:  # Threshold for suspicious correlation
                suspicious.append((i, float(corr)))
        
        if suspicious:
            signals.suspicious_feature_correlations = sorted(
                suspicious,
                key=lambda x: abs(x[1]),
                reverse=True
            )


def compute_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: TaskType,
    metric: str = "default"
) -> float:
    """
    Compute a single score for the given predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        task: Classification or regression
        metric: Metric name or "default"
        
    Returns:
        Computed score
    """
    if task == TaskType.CLASSIFICATION:
        if metric == "default" or metric == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif metric == "balanced_accuracy":
            return balanced_accuracy_score(y_true, y_pred)
        elif metric == "f1":
            return f1_score(y_true, y_pred, average="weighted")
        else:
            return accuracy_score(y_true, y_pred)
    
    else:  # Regression
        if metric == "default" or metric == "r2":
            return r2_score(y_true, y_pred)
        elif metric == "mse":
            return -mean_squared_error(y_true, y_pred)  # Negative for consistency
        elif metric == "mae":
            return -mean_absolute_error(y_true, y_pred)
        else:
            return r2_score(y_true, y_pred)


def analyze_cv_stability(cv_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze cross-validation result stability.
    
    This provides additional detail for CV interpretation.
    
    Args:
        cv_results: Dictionary from cross_validate()
        
    Returns:
        Dictionary with stability analysis
    """
    if "test_score" not in cv_results:
        return {"error": "No test_score in cv_results"}
    
    test_scores = np.asarray(cv_results["test_score"])
    n_folds = len(test_scores)
    
    analysis = {
        "n_folds": n_folds,
        "mean": float(np.mean(test_scores)),
        "std": float(np.std(test_scores)),
        "cv": float(np.std(test_scores) / np.mean(test_scores)) if np.mean(test_scores) > 0 else None,
        "range": float(np.max(test_scores) - np.min(test_scores)),
        "min_fold": int(np.argmin(test_scores)),
        "max_fold": int(np.argmax(test_scores)),
    }
    
    # Stability assessment
    cv = analysis["cv"]
    if cv is not None:
        if cv < 0.05:
            analysis["stability"] = "high"
        elif cv < 0.10:
            analysis["stability"] = "medium"
        elif cv < 0.20:
            analysis["stability"] = "low"
        else:
            analysis["stability"] = "very_low"
    
    # Detect outlier folds (more than 2 std from mean)
    mean = analysis["mean"]
    std = analysis["std"]
    outliers = []
    for i, score in enumerate(test_scores):
        if abs(score - mean) > 2 * std:
            outliers.append({
                "fold": i,
                "score": float(score),
                "deviation": float((score - mean) / std) if std > 0 else 0
            })
    analysis["outlier_folds"] = outliers
    
    # Train-test gap analysis if train scores available
    if "train_score" in cv_results:
        train_scores = np.asarray(cv_results["train_score"])
        gaps = train_scores - test_scores
        analysis["train_test_gaps"] = {
            "mean": float(np.mean(gaps)),
            "std": float(np.std(gaps)),
            "max": float(np.max(gaps)),
        }
    
    return analysis

# New signal extraction 
def _extract_calibration_signals(evidence: Evidence, signals: Signals) -> None:
    """
    Calibration signal — is model overconfident or underconfident?

    Uses predicted probabilities to detect:
    - Overconfidence: model always outputs very high probability (> 0.95)
    - Underconfidence: model always outputs near-uniform probability
    - Brier score: overall calibration quality (binary only)
    """
    if evidence.y_proba_val is None or evidence.y_val is None:
        signals.calibration_note = "Skipped — no predicted probabilities available (model may lack predict_proba)"
        return

    try:
        proba = np.array(evidence.y_proba_val)
        n_classes = proba.shape[1] if proba.ndim > 1 else 2
        max_proba = np.max(proba, axis=1) if proba.ndim > 1 else proba

        # Overconfidence: predictions clustering near 1.0
        overconfident_ratio = float(np.mean(max_proba > 0.95))
        signals.calibration_overconfidence_ratio = overconfident_ratio

        # Underconfidence: predictions clustering near uniform (1/n_classes)
        uniform_threshold = (1.0 / n_classes) + 0.05
        underconfident_ratio = float(np.mean(max_proba < uniform_threshold + 0.10))
        signals.calibration_underconfidence_ratio = underconfident_ratio

        # Brier score (binary classification only)
        if n_classes == 2:
            from sklearn.metrics import brier_score_loss
            pos_proba = proba[:, 1] if proba.ndim > 1 else proba
            signals.brier_score = float(brier_score_loss(evidence.y_val, pos_proba))

        # Human-readable note
        if overconfident_ratio > 0.60:
            signals.calibration_note = (
                f"Model is overconfident — {overconfident_ratio:.0%} of predictions "
                f"have >95% confidence. Consider calibration (Platt scaling / isotonic regression)."
            )
        elif underconfident_ratio > 0.50:
            signals.calibration_note = (
                f"Model is underconfident — {underconfident_ratio:.0%} of predictions "
                f"are near uniform probability. Model may be underfitting."
            )
        else:
            signals.calibration_note = "Calibration appears reasonable — no extreme overconfidence or underconfidence detected."

    except Exception as e:
        signals.calibration_note = f"Calibration signal skipped — {str(e)}"


def _extract_learning_curve_signals(evidence: Evidence, signals: Signals) -> None:
    """
    Learning curve shape — did the model actually converge?

    Trains on increasing fractions of data and checks:
    - Did validation score plateau? (convergence)
    - Is there still a large train-val gap at full data? (overfitting)
    - Is val score still rising? (needs more data)
    """
    try:
        from sklearn.model_selection import learning_curve as sk_learning_curve

        # We need the original model — stored in evidence if available
        # Use a proxy: recompute from evidence data
        if not evidence.has_validation_set:
            signals.learning_curve_note = "Skipped — no validation set provided."
            return

        # Can't refit without model object — check if evidence carries it
        # sklearn-diagnose passes model internally; we use CV scores as proxy
        if evidence.cv_results is None:
            signals.learning_curve_note = "Skipped — no CV results to approximate learning curve."
            return

        # Approximate learning curve from CV fold scores
        cv_scores = np.array(evidence.cv_results.get("test_score", []))
        if len(cv_scores) < 3:
            signals.learning_curve_note = "Skipped — too few CV folds to assess convergence."
            return

        # Use fold index as proxy for "data seen"
        signals.learning_curve_val_scores = cv_scores.tolist()

        # Check convergence: std of last half vs first half
        mid = len(cv_scores) // 2
        first_half_std = float(np.std(cv_scores[:mid]))
        second_half_std = float(np.std(cv_scores[mid:]))
        converged = second_half_std <= first_half_std * 1.2
        signals.learning_curve_converged = converged

        # Gap at full data
        if evidence.cv_results.get("train_score") is not None:
            train_scores = np.array(evidence.cv_results["train_score"])
            signals.learning_curve_gap_at_full = float(
                np.mean(train_scores) - np.mean(cv_scores)
            )

        # Human-readable note
        if not converged:
            signals.learning_curve_note = (
                "CV scores are still varying significantly across folds — "
                "model may benefit from more training data or tuning."
            )
        elif signals.learning_curve_gap_at_full and signals.learning_curve_gap_at_full > 0.15:
            signals.learning_curve_note = (
                f"Model converged but train-val gap is {signals.learning_curve_gap_at_full:.1%} — "
                f"likely overfitting even with full data."
            )
        else:
            signals.learning_curve_note = "Model appears to have converged — learning curve is stable."

    except Exception as e:
        signals.learning_curve_note = f"Learning curve signal skipped — {str(e)}"


def _extract_prediction_distribution_signals(evidence: Evidence, signals: Signals) -> None:
    """
    Prediction distribution — are outputs spread or clustered?

    Checks if model is predicting one class far too often,
    which may indicate majority class bias or a collapsed model.
    """
    if evidence.y_pred_val is None:
        signals.pred_distribution_note = "Skipped — no validation predictions available."
        return

    try:
        preds = np.array(evidence.y_pred_val)
        classes, counts = np.unique(preds, return_counts=True)
        total = len(preds)
        pred_dist = counts / total

        # Entropy of prediction distribution (higher = more spread)
        entropy = float(-np.sum(pred_dist * np.log(pred_dist + 1e-9)))
        signals.pred_distribution_entropy = entropy

        # How often does model predict the majority class?
        majority_ratio = float(np.max(counts) / total)
        signals.pred_majority_class_ratio = majority_ratio

        # Human-readable note
        max_entropy = float(np.log(len(classes)))
        relative_entropy = entropy / (max_entropy + 1e-9)

        if majority_ratio > 0.90:
            signals.pred_distribution_note = (
                f"Model predicts one class {majority_ratio:.0%} of the time — "
                f"likely collapsed due to class imbalance or poor training."
            )
        elif relative_entropy < 0.5:
            signals.pred_distribution_note = (
                f"Prediction distribution is skewed (entropy = {entropy:.2f}) — "
                f"model may be biased toward certain classes."
            )
        else:
            signals.pred_distribution_note = (
                f"Prediction distribution looks healthy (entropy = {entropy:.2f})."
            )

    except Exception as e:
        signals.pred_distribution_note = f"Prediction distribution signal skipped — {str(e)}"


def _extract_threshold_sensitivity_signals(evidence: Evidence, signals: Signals) -> None:
    """
    Threshold sensitivity — how fragile is the decision boundary?

    For binary classifiers: checks what % of predictions would flip
    if decision threshold moved from 0.5 → 0.4 or 0.5 → 0.6.
    High sensitivity = fragile boundary = unreliable predictions.
    """
    if evidence.y_proba_val is None:
        signals.threshold_sensitivity_note = "Skipped — no predicted probabilities available."
        return

    try:
        proba = np.array(evidence.y_proba_val)

        # Only meaningful for binary classification
        if proba.ndim < 2 or proba.shape[1] != 2:
            signals.threshold_sensitivity_note = "Skipped — threshold sensitivity only applies to binary classification."
            return

        pos_proba = proba[:, 1]

        # Predictions at three thresholds
        preds_05 = (pos_proba >= 0.50).astype(int)
        preds_04 = (pos_proba >= 0.40).astype(int)
        preds_06 = (pos_proba >= 0.60).astype(int)

        # What fraction of predictions flip?
        flip_down = float(np.mean(preds_05 != preds_04))  # threshold lowered
        flip_up = float(np.mean(preds_05 != preds_06))    # threshold raised
        sensitivity = max(flip_down, flip_up)

        signals.threshold_sensitivity_score = sensitivity

        # Human-readable note
        if sensitivity > 0.20:
            signals.threshold_sensitivity_note = (
                f"{sensitivity:.0%} of predictions flip when threshold shifts by ±0.1 — "
                f"decision boundary is fragile. Consider optimising threshold on validation set."
            )
        elif sensitivity > 0.10:
            signals.threshold_sensitivity_note = (
                f"{sensitivity:.0%} of predictions are threshold-sensitive — "
                f"moderate fragility. Monitor on new data."
            )
        else:
            signals.threshold_sensitivity_note = (
                f"Decision boundary is stable — only {sensitivity:.0%} of predictions "
                f"shift with threshold changes."
            )

    except Exception as e:
        signals.threshold_sensitivity_note = f"Threshold sensitivity signal skipped — {str(e)}"


def _extract_feature_drift_signals(evidence: Evidence, signals: Signals) -> None:
    """
    Feature drift — which features are potentially problematic?

    Checks for:
    - Extreme variance differences across features (scale issues)
    - Features with near-zero variance (useless features)
    - Features with suspiciously high variance (outlier-driven)
    """
    try:
        X = np.array(evidence.X_train)

        if X.ndim != 2 or X.shape[1] < 2:
            signals.feature_drift_note = "Skipped — need at least 2 features."
            return

        variances = np.var(X, axis=0)

        # Variance ratio: max / min (scale issue detector)
        min_var = np.min(variances)
        max_var = np.max(variances)
        ratio = float(max_var / (min_var + 1e-9))
        signals.feature_variance_ratio = ratio

        # Flag problematic features
        drift_flags = []

        # Near-zero variance → useless features
        zero_var_threshold = np.percentile(variances, 10) * 0.1
        for i, var in enumerate(variances):
            if var <= zero_var_threshold and zero_var_threshold > 0:
                drift_flags.append(i)

        # Extremely high variance vs others → outlier-driven
        high_var_threshold = np.mean(variances) + 3 * np.std(variances)
        for i, var in enumerate(variances):
            if var > high_var_threshold and i not in drift_flags:
                drift_flags.append(i)

        signals.feature_drift_flags = drift_flags if drift_flags else []

        # Human-readable note
        if ratio > 1000:
            signals.feature_drift_note = (
                f"Feature scales vary by {ratio:.0f}x — strong indicator of missing normalisation. "
                f"Apply StandardScaler or MinMaxScaler before training."
            )
        elif ratio > 100:
            signals.feature_drift_note = (
                f"Feature scales vary by {ratio:.0f}x — moderate scale issue detected. "
                f"Consider feature scaling."
            )
        elif drift_flags:
            signals.feature_drift_note = (
                f"{len(drift_flags)} feature(s) flagged as potentially problematic "
                f"(near-zero or extreme variance): indices {drift_flags}."
            )
        else:
            signals.feature_drift_note = "Feature variances look healthy — no obvious scale or drift issues."

    except Exception as e:
        signals.feature_drift_note = f"Feature drift signal skipped — {str(e)}"
