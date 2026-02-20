"""
Recommendation templates for sklearn-diagnose.

This module provides example recommendations for each failure mode.
These are suggested examples that the LLM uses as guidance when generating
the final recommendations.
"""

from typing import Dict, List, Optional

from .schemas import FailureMode, Signals


# Example recommendation templates organized by failure mode
# These are suggestions for the LLM, not exhaustive lists
RECOMMENDATION_TEMPLATES: Dict[FailureMode, List[dict]] = {
    FailureMode.OVERFITTING: [
        {
            "action": "Increase regularization strength",
            "rationale": "Stronger regularization penalizes model complexity and reduces memorization of training data",
        },
        {
            "action": "Reduce model complexity",
            "rationale": "Simpler models generalize better; consider fewer features, shallower trees, or fewer parameters",
        },
        {
            "action": "Apply feature selection to reduce dimensionality",
            "rationale": "Fewer features reduce the model's ability to memorize noise in the training data",
        },
        {
            "action": "Collect more training data",
            "rationale": "More data provides better coverage of the true distribution and reduces overfitting",
        },
        {
            "action": "Use early stopping",
            "rationale": "Early stopping prevents the model from fitting too closely to training data",
        },
    ],
    
    FailureMode.UNDERFITTING: [
        {
            "action": "Increase model complexity",
            "rationale": "The current model may be too simple to capture the underlying patterns",
        },
        {
            "action": "Create more informative features through feature engineering",
            "rationale": "Better features help the model learn meaningful patterns",
        },
        {
            "action": "Reduce regularization strength",
            "rationale": "Too much regularization prevents the model from learning complex patterns",
        },
        {
            "action": "Try a different algorithm better suited to the data",
            "rationale": "Different algorithms have different inductive biases; another may fit your data better",
        },
        {
            "action": "Ensure data is properly preprocessed and scaled",
            "rationale": "Many models perform poorly on unscaled or improperly encoded data",
        },
    ],
    
    FailureMode.HIGH_VARIANCE: [
        {
            "action": "Use ensemble methods for more stable predictions",
            "rationale": "Ensembles average multiple models, reducing sensitivity to data splits",
        },
        {
            "action": "Increase training data size",
            "rationale": "More data reduces the impact of individual samples on the model",
        },
        {
            "action": "Add regularization to stabilize the model",
            "rationale": "Regularization prevents the model from being too sensitive to specific training samples",
        },
        {
            "action": "Use cross-validation during model selection",
            "rationale": "CV provides more robust estimates of model performance than a single split",
        },
        {
            "action": "Check for data quality issues in specific folds",
            "rationale": "Outlier folds may indicate data quality problems in certain subsets",
        },
    ],
    
    FailureMode.CLASS_IMBALANCE: [
        {
            "action": "Use class weights to balance the training objective",
            "rationale": "Class weights make the model pay more attention to minority class errors",
        },
        {
            "action": "Use stratified sampling for train/test splits",
            "rationale": "Stratification ensures each split has representative class proportions",
        },
        {
            "action": "Consider resampling techniques (SMOTE, undersampling)",
            "rationale": "Resampling can create a more balanced training set",
        },
        {
            "action": "Use metrics appropriate for imbalanced data",
            "rationale": "Accuracy is misleading; use F1, precision-recall, or balanced accuracy",
        },
        {
            "action": "Adjust decision threshold for predictions",
            "rationale": "The default 0.5 threshold may not be optimal for imbalanced data",
        },
    ],
    
    FailureMode.FEATURE_REDUNDANCY: [
        {
            "action": "Apply dimensionality reduction (PCA, feature selection)",
            "rationale": "Reducing redundant features simplifies the model and can improve performance",
        },
        {
            "action": "Remove highly correlated features",
            "rationale": "Keeping only one of a pair of correlated features reduces redundancy",
        },
        {
            "action": "Use regularization methods that handle collinearity",
            "rationale": "Ridge regression and elastic net handle correlated features better",
        },
        {
            "action": "Consider feature clustering or grouping",
            "rationale": "Grouping similar features can reduce redundancy while preserving information",
        },
    ],
    
    FailureMode.LABEL_NOISE: [
        {
            "action": "Review and clean mislabeled samples",
            "rationale": "Identifying and correcting mislabeled samples improves model quality",
        },
        {
            "action": "Use robust loss functions or models",
            "rationale": "Robust methods are less sensitive to outliers and label noise",
        },
        {
            "action": "Consider label smoothing or soft labels",
            "rationale": "Soft labels reduce overconfidence and make the model more robust to noise",
        },
        {
            "action": "Use cross-validation to identify samples with consistently wrong predictions",
            "rationale": "Samples that are always misclassified across folds are likely mislabeled",
        },
    ],
    
    FailureMode.DATA_LEAKAGE: [
        {
            "action": "Review feature engineering for temporal leakage",
            "rationale": "Features computed from future data can cause leakage",
        },
        {
            "action": "Check for target leakage in features",
            "rationale": "Features that directly encode the target cause leakage",
        },
        {
            "action": "Ensure preprocessing is fit only on training data",
            "rationale": "Fitting scalers/encoders on the full dataset leaks validation information",
        },
        {
            "action": "Verify train/validation data are properly separated",
            "rationale": "Overlapping samples between train and validation sets cause leakage",
        },
        {
            "action": "Use holdout set that was never seen during development",
            "rationale": "A truly independent holdout set provides unbiased performance estimates",
        },
    ],
}


def get_example_recommendations_for_failure_mode(failure_mode: FailureMode) -> List[dict]:
    """
    Get example recommendations for a given failure mode.
    
    These are suggestions for the LLM, not exhaustive lists.
    
    Args:
        failure_mode: The failure mode to get recommendations for
        
    Returns:
        List of example recommendation dictionaries
    """
    return RECOMMENDATION_TEMPLATES.get(failure_mode, [])


def get_all_failure_modes_with_examples() -> Dict[str, List[dict]]:
    """
    Get all failure modes with their example recommendations.
    
    Returns:
        Dictionary mapping failure mode names to example recommendations
    """
    return {
        mode.value: templates 
        for mode, templates in RECOMMENDATION_TEMPLATES.items()
    }


def get_insufficient_evidence_message(signals: Signals) -> Optional[str]:
    """
    Generate a message when there's insufficient evidence for diagnosis.
    
    Args:
        signals: Computed signals
        
    Returns:
        Message string if evidence is insufficient, None otherwise
    """
    issues = []
    
    if signals.train_score is None:
        issues.append("Unable to compute training score")
    
    if signals.val_score is None and signals.cv_mean is None:
        issues.append("No validation or CV scores available for comparison")
    
    if signals.n_samples_train is not None and signals.n_samples_train < 50:
        issues.append(f"Very small training set ({signals.n_samples_train} samples)")
    
    if issues:
        return (
            "**Insufficient Evidence Warning**\n\n"
            "The diagnosis may be limited due to:\n"
            + "\n".join(f"- {issue}" for issue in issues)
            + "\n\nConsider providing more data or cross-validation results for better diagnosis."
        )
    
    return None
