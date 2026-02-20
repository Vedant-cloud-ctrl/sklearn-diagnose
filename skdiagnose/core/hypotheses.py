"""
Hypothesis generation rules for sklearn-diagnose (reference implementation).

This module provides rule-based hypothesis generation that can be used as:
- A reference implementation for understanding detection logic
- A fallback when LLM is unavailable
- Validation/comparison against LLM-generated hypotheses

Note: The primary hypothesis generation is now LLM-driven (see llm/client.py).
This module contains the deterministic rules that were the original implementation.

Rules are:
- Primarily domain-agnostic and generally accepted at the core
- Evidence-based, not heuristic hand-waving
- Expressed as confidence-weighted hypotheses, not hard truths

Each rule computes a confidence score based on quantitative evidence.
"""

from typing import List, Optional

import numpy as np

from .schemas import FailureMode, Hypothesis, Signals, TaskType


# Thresholds for hypothesis detection
# These are conservative, evidence-based thresholds
THRESHOLDS = {
    # Overfitting
    "train_val_gap_mild": 0.10,      # 10% gap
    "train_val_gap_moderate": 0.15,  # 15% gap
    "train_val_gap_severe": 0.25,    # 25% gap
    "train_score_high": 0.85,        # Training score suggesting possible overfitting
    
    # Underfitting
    "score_low": 0.60,               # Scores below this suggest underfitting
    "score_very_low": 0.50,          # Scores below this are clearly problematic
    
    # High variance
    "cv_std_mild": 0.05,             # 5% std
    "cv_std_moderate": 0.10,         # 10% std
    "cv_std_severe": 0.15,           # 15% std
    "cv_range_high": 0.20,           # 20% range between folds
    
    # Class imbalance
    "minority_ratio_mild": 0.20,     # 20% minority class
    "minority_ratio_moderate": 0.10, # 10% minority class
    "minority_ratio_severe": 0.05,   # 5% minority class
    "recall_disparity": 0.30,        # 30% difference in per-class recall
    
    # Feature redundancy
    "feature_corr_high": 0.90,       # 90% correlation between features
    "feature_corr_very_high": 0.95,  # 95% correlation
    
    # Label noise
    "train_ceiling": 0.98,           # Near-perfect training score
    "residual_skew_high": 1.5,       # High skewness in residuals
    
    # Data leakage
    "cv_holdout_gap": 0.10,          # 10% gap between CV and holdout
    "feature_target_corr": 0.95,     # 95% correlation with target
    
    # Data quantity
    "sample_to_feature_ratio_low": 10,  # Less than 10 samples per feature
}


def generate_hypotheses(
    signals: Signals,
    task: TaskType
) -> List[Hypothesis]:
    """
    Generate hypotheses using deterministic rules (reference implementation).
    
    Note: The main diagnose() function now uses LLM-based hypothesis generation.
    This function is kept as a reference implementation and potential fallback.
    
    Args:
        signals: Computed signals from the signal extraction layer
        task: Classification or regression
        
    Returns:
        List of confidence-weighted hypotheses
    """
    hypotheses = []
    
    # Check each failure mode
    hypotheses.extend(_check_overfitting(signals))
    hypotheses.extend(_check_underfitting(signals))
    hypotheses.extend(_check_high_variance(signals))
    hypotheses.extend(_check_class_imbalance(signals, task))
    hypotheses.extend(_check_feature_redundancy(signals))
    hypotheses.extend(_check_label_noise(signals, task))
    hypotheses.extend(_check_data_leakage(signals))
    
    # Filter out low-confidence hypotheses
    # (keep even low confidence for transparency, but mark appropriately)
    
    # Sort by confidence (highest first)
    hypotheses.sort(key=lambda h: h.confidence, reverse=True)
    
    return hypotheses


def _check_overfitting(signals: Signals) -> List[Hypothesis]:
    """
    Check for overfitting signals.
    
    Overfitting is characterized by:
    - High training score
    - Significant gap between train and validation scores
    - Model performs well on training data but poorly on unseen data
    """
    hypotheses = []
    evidence = []
    confidence = 0.0
    severity = "low"
    
    # Check train-val gap (primary signal)
    if signals.train_val_gap is not None:
        gap = signals.train_val_gap
        
        if gap >= THRESHOLDS["train_val_gap_severe"]:
            confidence = min(0.95, 0.7 + gap)
            severity = "high"
            evidence.append(f"Train-val gap of {gap:.1%} is severe (>{THRESHOLDS['train_val_gap_severe']:.0%})")
        elif gap >= THRESHOLDS["train_val_gap_moderate"]:
            confidence = min(0.85, 0.5 + gap)
            severity = "medium"
            evidence.append(f"Train-val gap of {gap:.1%} is moderate")
        elif gap >= THRESHOLDS["train_val_gap_mild"]:
            confidence = min(0.60, 0.3 + gap)
            severity = "low"
            evidence.append(f"Train-val gap of {gap:.1%} suggests mild overfitting")
    
    # Check CV train-test gap (if CV available)
    if signals.cv_train_val_gap is not None:
        cv_gap = signals.cv_train_val_gap
        
        if cv_gap >= THRESHOLDS["train_val_gap_moderate"]:
            # Boost confidence if CV also shows gap
            confidence = min(0.95, confidence + 0.15)
            evidence.append(f"CV train-test gap of {cv_gap:.1%} confirms overfitting pattern")
    
    # High training score as supporting evidence
    if signals.train_score is not None and signals.train_score >= THRESHOLDS["train_score_high"]:
        evidence.append(f"High training score ({signals.train_score:.1%}) with lower validation score")
        if confidence > 0:
            confidence = min(0.95, confidence + 0.05)
    
    if confidence >= 0.20 and evidence:
        hypotheses.append(Hypothesis(
            name=FailureMode.OVERFITTING,
            confidence=round(confidence, 2),
            evidence=evidence,
            severity=severity
        ))
    
    return hypotheses


def _check_underfitting(signals: Signals) -> List[Hypothesis]:
    """
    Check for underfitting signals.
    
    Underfitting is characterized by:
    - Low training score
    - Low validation score
    - Small gap between train and val (both are bad)
    """
    hypotheses = []
    evidence = []
    confidence = 0.0
    severity = "low"
    
    # Both scores must be low
    train_low = (signals.train_score is not None and 
                 signals.train_score < THRESHOLDS["score_low"])
    val_low = (signals.val_score is not None and 
               signals.val_score < THRESHOLDS["score_low"])
    cv_low = (signals.cv_mean is not None and 
              signals.cv_mean < THRESHOLDS["score_low"])
    
    # Primary check: low training score
    if train_low:
        confidence = 0.5 + (THRESHOLDS["score_low"] - signals.train_score)
        evidence.append(f"Training score ({signals.train_score:.1%}) is low, suggesting model cannot learn the pattern")
        
        if signals.train_score < THRESHOLDS["score_very_low"]:
            severity = "high"
            confidence = min(0.95, confidence + 0.15)
        else:
            severity = "medium"
    
    # Supporting: low val score without large gap
    if val_low and signals.train_val_gap is not None:
        if signals.train_val_gap < THRESHOLDS["train_val_gap_moderate"]:
            confidence = min(0.95, confidence + 0.15)
            evidence.append(f"Validation score ({signals.val_score:.1%}) is also low with small train-val gap")
    
    # Supporting: low CV mean
    if cv_low:
        confidence = min(0.95, confidence + 0.10)
        evidence.append(f"CV mean ({signals.cv_mean:.1%}) confirms poor learning across folds")
    
    if confidence >= 0.25 and evidence:
        hypotheses.append(Hypothesis(
            name=FailureMode.UNDERFITTING,
            confidence=round(confidence, 2),
            evidence=evidence,
            severity=severity
        ))
    
    return hypotheses


def _check_high_variance(signals: Signals) -> List[Hypothesis]:
    """
    Check for high variance signals.
    
    High variance is characterized by:
    - Large standard deviation in CV fold scores
    - Large range between best and worst folds
    - Unstable model that's sensitive to data splits
    """
    hypotheses = []
    evidence = []
    confidence = 0.0
    severity = "low"
    
    # Primary: CV standard deviation
    if signals.cv_std is not None:
        std = signals.cv_std
        
        if std >= THRESHOLDS["cv_std_severe"]:
            confidence = min(0.90, 0.6 + std)
            severity = "high"
            evidence.append(f"CV std of {std:.1%} indicates highly unstable performance")
        elif std >= THRESHOLDS["cv_std_moderate"]:
            confidence = min(0.75, 0.4 + std)
            severity = "medium"
            evidence.append(f"CV std of {std:.1%} indicates moderate instability")
        elif std >= THRESHOLDS["cv_std_mild"]:
            confidence = min(0.50, 0.2 + std)
            severity = "low"
            evidence.append(f"CV std of {std:.1%} suggests some variance sensitivity")
    
    # Supporting: large CV range
    if signals.cv_range is not None and signals.cv_range >= THRESHOLDS["cv_range_high"]:
        confidence = min(0.95, confidence + 0.10)
        evidence.append(f"CV range of {signals.cv_range:.1%} between folds")
    
    # Supporting: fold outliers
    if signals.cv_fold_scores is not None:
        scores = np.array(signals.cv_fold_scores)
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std > 0:
            outliers = np.abs(scores - mean) > 2 * std
            n_outliers = np.sum(outliers)
            if n_outliers > 0:
                confidence = min(0.95, confidence + 0.05 * n_outliers)
                evidence.append(f"{n_outliers} fold(s) are outliers (>2 std from mean)")
    
    if confidence >= 0.20 and evidence:
        hypotheses.append(Hypothesis(
            name=FailureMode.HIGH_VARIANCE,
            confidence=round(confidence, 2),
            evidence=evidence,
            severity=severity
        ))
    
    return hypotheses


def _check_class_imbalance(signals: Signals, task: TaskType) -> List[Hypothesis]:
    """
    Check for class imbalance issues.
    
    Class imbalance is characterized by:
    - Skewed class distribution
    - Poor recall on minority class
    - Large disparity in per-class metrics
    """
    if task != TaskType.CLASSIFICATION:
        return []
    
    hypotheses = []
    evidence = []
    confidence = 0.0
    severity = "low"
    
    # Primary: minority class ratio
    if signals.minority_class_ratio is not None:
        ratio = signals.minority_class_ratio
        
        if ratio <= THRESHOLDS["minority_ratio_severe"]:
            confidence = min(0.90, 0.7 + (THRESHOLDS["minority_ratio_severe"] - ratio) * 5)
            severity = "high"
            evidence.append(f"Minority class is only {ratio:.1%} of data (severe imbalance)")
        elif ratio <= THRESHOLDS["minority_ratio_moderate"]:
            confidence = min(0.70, 0.5 + (THRESHOLDS["minority_ratio_moderate"] - ratio) * 3)
            severity = "medium"
            evidence.append(f"Minority class is {ratio:.1%} of data (moderate imbalance)")
        elif ratio <= THRESHOLDS["minority_ratio_mild"]:
            confidence = min(0.50, 0.3)
            severity = "low"
            evidence.append(f"Minority class is {ratio:.1%} of data (mild imbalance)")
    
    # Supporting: per-class recall disparity
    if signals.per_class_recall is not None and len(signals.per_class_recall) >= 2:
        recalls = list(signals.per_class_recall.values())
        disparity = max(recalls) - min(recalls)
        
        if disparity >= THRESHOLDS["recall_disparity"]:
            confidence = min(0.95, confidence + 0.20)
            evidence.append(f"Recall disparity of {disparity:.1%} across classes")
            
            # Identify which class is struggling
            min_class = min(signals.per_class_recall, key=signals.per_class_recall.get)
            min_recall = signals.per_class_recall[min_class]
            evidence.append(f"Class '{min_class}' has only {min_recall:.1%} recall")
    
    if confidence >= 0.25 and evidence:
        hypotheses.append(Hypothesis(
            name=FailureMode.CLASS_IMBALANCE,
            confidence=round(confidence, 2),
            evidence=evidence,
            severity=severity
        ))
    
    return hypotheses


def _check_feature_redundancy(signals: Signals) -> List[Hypothesis]:
    """
    Check for feature redundancy signals.
    
    Feature redundancy is characterized by:
    - High correlations between features
    - Multiple features providing similar information
    """
    hypotheses = []
    evidence = []
    confidence = 0.0
    severity = "low"
    
    if signals.high_correlation_pairs is not None and len(signals.high_correlation_pairs) > 0:
        n_pairs = len(signals.high_correlation_pairs)
        max_corr = signals.high_correlation_pairs[0][2]  # Highest correlation
        
        # Base confidence on number and strength of correlations
        if max_corr >= THRESHOLDS["feature_corr_very_high"]:
            confidence = min(0.85, 0.6 + max_corr - 0.9)
            severity = "medium"
            evidence.append(f"Features have extremely high correlation ({max_corr:.1%})")
        elif max_corr >= THRESHOLDS["feature_corr_high"]:
            confidence = min(0.65, 0.4 + max_corr - 0.85)
            severity = "low"
            evidence.append(f"Features have high correlation ({max_corr:.1%})")
        
        # More pairs = higher concern
        if n_pairs > 5:
            confidence = min(0.90, confidence + 0.10)
            evidence.append(f"{n_pairs} pairs of highly correlated features found")
        elif n_pairs > 1:
            confidence = min(0.85, confidence + 0.05)
            evidence.append(f"{n_pairs} pairs of highly correlated features")
    
    # Check feature-to-sample ratio
    if signals.feature_to_sample_ratio is not None:
        ratio = 1 / signals.feature_to_sample_ratio  # samples per feature
        if ratio < THRESHOLDS["sample_to_feature_ratio_low"]:
            confidence = min(0.90, confidence + 0.15)
            evidence.append(f"Only {ratio:.1f} samples per feature (risk of spurious correlations)")
            severity = "medium"
    
    if confidence >= 0.30 and evidence:
        hypotheses.append(Hypothesis(
            name=FailureMode.FEATURE_REDUNDANCY,
            confidence=round(confidence, 2),
            evidence=evidence,
            severity=severity
        ))
    
    return hypotheses


def _check_label_noise(signals: Signals, task: TaskType) -> List[Hypothesis]:
    """
    Check for label noise signals.
    
    Label noise is characterized by:
    - Near-perfect training score but scattered errors
    - High training score but lower-than-expected validation
    - For regression: high residual variance, non-normal residuals
    """
    hypotheses = []
    evidence = []
    confidence = 0.0
    severity = "low"
    
    # Training score near ceiling but not perfect
    if signals.train_score is not None:
        train = signals.train_score
        
        if train >= THRESHOLDS["train_ceiling"] and signals.val_score is not None:
            val_gap = train - signals.val_score
            if val_gap > THRESHOLDS["train_val_gap_mild"]:
                confidence = 0.45
                evidence.append(
                    f"Near-perfect training ({train:.1%}) with validation gap may indicate noise in labels"
                )
                severity = "low"
    
    # Regression-specific: residual analysis
    if task == TaskType.REGRESSION:
        if signals.residual_skew is not None:
            skew = abs(signals.residual_skew)
            if skew > THRESHOLDS["residual_skew_high"]:
                confidence = min(0.75, confidence + 0.20)
                evidence.append(f"Residual skewness of {skew:.2f} suggests non-random errors")
                severity = "medium"
        
        if signals.residual_kurtosis is not None:
            kurt = signals.residual_kurtosis
            if abs(kurt) > 3:  # Excess kurtosis
                confidence = min(0.80, confidence + 0.10)
                evidence.append(f"Residual kurtosis of {kurt:.2f} suggests heavy-tailed errors")
    
    if confidence >= 0.30 and evidence:
        hypotheses.append(Hypothesis(
            name=FailureMode.LABEL_NOISE,
            confidence=round(confidence, 2),
            evidence=evidence,
            severity=severity
        ))
    
    return hypotheses


def _check_data_leakage(signals: Signals) -> List[Hypothesis]:
    """
    Check for data leakage signals.
    
    Data leakage is characterized by:
    - CV performance much higher than holdout
    - Suspiciously high feature-target correlations
    - Perfect or near-perfect scores
    """
    hypotheses = []
    evidence = []
    confidence = 0.0
    severity = "low"
    
    # CV vs holdout discrepancy
    if signals.cv_holdout_gap is not None:
        gap = signals.cv_holdout_gap
        
        if gap > THRESHOLDS["cv_holdout_gap"] * 2:
            confidence = min(0.80, 0.5 + gap)
            severity = "high"
            evidence.append(
                f"CV performance ({signals.cv_mean:.1%}) significantly exceeds holdout ({signals.val_score:.1%})"
            )
        elif gap > THRESHOLDS["cv_holdout_gap"]:
            confidence = min(0.55, 0.3 + gap)
            severity = "medium"
            evidence.append(
                f"CV-holdout gap of {gap:.1%} may indicate leakage or distribution shift"
            )
    
    # Suspiciously high feature-target correlations
    if signals.suspicious_feature_correlations is not None:
        n_suspicious = len(signals.suspicious_feature_correlations)
        max_corr = signals.suspicious_feature_correlations[0][1]
        
        if abs(max_corr) >= THRESHOLDS["feature_target_corr"]:
            confidence = min(0.85, confidence + 0.25)
            severity = "high"
            evidence.append(
                f"{n_suspicious} feature(s) have suspiciously high correlation with target "
                f"(max: {max_corr:.1%})"
            )
    
    # Perfect training score is always suspicious
    if signals.train_score is not None and signals.train_score >= 0.99:
        confidence = min(0.75, confidence + 0.20)
        evidence.append(
            f"Perfect training score ({signals.train_score:.1%}) warrants leakage investigation"
        )
        severity = "high"
    
    if confidence >= 0.30 and evidence:
        hypotheses.append(Hypothesis(
            name=FailureMode.DATA_LEAKAGE,
            confidence=round(confidence, 2),
            evidence=evidence,
            severity=severity
        ))
    
    return hypotheses
