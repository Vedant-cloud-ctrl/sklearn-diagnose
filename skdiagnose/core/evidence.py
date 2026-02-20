"""
Evidence collection and validation for sklearn-diagnose.

This module handles:
- Validating that estimators are fitted
- Ensuring read-only behavior
- Collecting predictions from estimators
- Validating dataset integrity
- Preventing accidental data leakage between train/val
"""

import hashlib
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .schemas import Evidence, TaskType, ValidationResult


def validate_estimator(estimator: BaseEstimator) -> ValidationResult:
    """
    Validate that the estimator is properly fitted and compatible.
    
    Args:
        estimator: A scikit-learn estimator or Pipeline
        
    Returns:
        ValidationResult with any errors or warnings
    """
    errors = []
    warnings_list = []
    
    # Check if estimator is a valid sklearn object
    if not isinstance(estimator, BaseEstimator):
        errors.append(
            f"Estimator must be a scikit-learn BaseEstimator, got {type(estimator).__name__}"
        )
        return ValidationResult(is_valid=False, errors=errors)
    
    # Check if estimator is fitted (fail fast)
    try:
        check_is_fitted(estimator)
    except Exception as e:
        errors.append(
            f"Estimator is not fitted. Call estimator.fit(X, y) first. "
            f"sklearn-diagnose only analyzes fitted models. Error: {str(e)}"
        )
        return ValidationResult(is_valid=False, errors=errors)
    
    # Check for predict method
    if not hasattr(estimator, "predict"):
        errors.append("Estimator must have a predict() method")
        return ValidationResult(is_valid=False, errors=errors)
    
    # Check if it's a Pipeline and note for diagnostics
    if isinstance(estimator, Pipeline):
        warnings_list.append(
            "Pipeline detected. Additional preprocessing diagnostics available."
        )
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings_list
    )


def validate_datasets(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    cv_results: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """
    Validate dataset integrity and check for potential issues.
    
    Args:
        datasets: Dictionary with "train" and optionally "val" keys
        cv_results: Optional cross-validation results
        
    Returns:
        ValidationResult with any errors or warnings
    """
    errors = []
    warnings_list = []
    
    # Check required train set
    if "train" not in datasets:
        errors.append("datasets must contain 'train' key with (X_train, y_train)")
        return ValidationResult(is_valid=False, errors=errors)
    
    X_train, y_train = datasets["train"]
    
    # Validate training data
    if X_train is None or y_train is None:
        errors.append("X_train and y_train cannot be None")
        return ValidationResult(is_valid=False, errors=errors)
    
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    
    if len(X_train) != len(y_train):
        errors.append(
            f"X_train and y_train must have same length. "
            f"Got {len(X_train)} and {len(y_train)}"
        )
    
    if len(X_train) == 0:
        errors.append("Training data cannot be empty")
    
    # Check validation set if provided
    if "val" in datasets:
        X_val, y_val = datasets["val"]
        
        if X_val is None or y_val is None:
            errors.append("If 'val' key is provided, both X_val and y_val must be non-None")
        else:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)
            
            if len(X_val) != len(y_val):
                errors.append(
                    f"X_val and y_val must have same length. "
                    f"Got {len(X_val)} and {len(y_val)}"
                )
            
            if len(X_val) == 0:
                errors.append("Validation data cannot be empty")
            
            # Check for same number of features
            if X_train.shape[1:] != X_val.shape[1:]:
                errors.append(
                    f"X_train and X_val must have same feature dimensions. "
                    f"Got {X_train.shape[1:]} and {X_val.shape[1:]}"
                )
            
            # Check for potential data leakage (same data in train and val)
            leakage_check = _check_data_leakage(X_train, X_val)
            if leakage_check:
                warnings_list.append(leakage_check)
    
    # Validate evidence sources
    # Note: If no validation/CV data is provided, a recommendation will be added
    # in the diagnose() function instead of showing a warning here
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings_list
    )


def _check_data_leakage(X_train: np.ndarray, X_val: np.ndarray) -> Optional[str]:
    """
    Check for potential data leakage between train and validation sets.
    
    Returns warning message if leakage suspected, None otherwise.
    """
    # Check if shapes match exactly
    if X_train.shape == X_val.shape:
        # Quick hash comparison for identical data
        try:
            train_hash = hashlib.md5(X_train.tobytes()).hexdigest()
            val_hash = hashlib.md5(X_val.tobytes()).hexdigest()
            
            if train_hash == val_hash:
                return (
                    "WARNING: X_train and X_val appear to be identical! "
                    "This indicates a serious data leakage issue."
                )
        except Exception:
            pass
    
    # Check for high overlap using sampling
    if len(X_train) > 0 and len(X_val) > 0:
        try:
            # Sample and check for exact row matches
            n_check = min(100, len(X_val))
            sample_idx = np.random.choice(len(X_val), n_check, replace=False)
            
            matches = 0
            for idx in sample_idx:
                if any(np.allclose(X_val[idx], row) for row in X_train[:1000]):
                    matches += 1
            
            overlap_ratio = matches / n_check
            if overlap_ratio > 0.1:
                return (
                    f"WARNING: {overlap_ratio:.0%} of validation samples appear in training data. "
                    "This may indicate data leakage."
                )
        except Exception:
            pass
    
    return None


def validate_cv_results(cv_results: Dict[str, Any]) -> ValidationResult:
    """
    Validate cross-validation results structure.
    
    Args:
        cv_results: Dictionary from cross_validate()
        
    Returns:
        ValidationResult with any errors or warnings
    """
    errors = []
    warnings_list = []
    
    if not isinstance(cv_results, dict):
        errors.append(f"cv_results must be a dictionary, got {type(cv_results).__name__}")
        return ValidationResult(is_valid=False, errors=errors)
    
    # Check for test scores
    if "test_score" not in cv_results:
        errors.append(
            "cv_results must contain 'test_score' key. "
            "Use cross_validate() with default parameters."
        )
    
    # Check for train scores (recommended)
    if "train_score" not in cv_results:
        warnings_list.append(
            "cv_results does not contain 'train_score'. "
            "For better diagnosis, use cross_validate(..., return_train_score=True)"
        )
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings_list
    )


def collect_evidence(
    estimator: BaseEstimator,
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    task: Union[str, TaskType],
    cv_results: Optional[Dict[str, Any]] = None
) -> Evidence:
    """
    Collect all evidence needed for diagnosis.
    
    This function:
    1. Validates all inputs
    2. Extracts predictions from the fitted estimator
    3. Collects feature metadata if available
    4. Returns a structured Evidence object
    
    Args:
        estimator: A fitted scikit-learn estimator or Pipeline
        datasets: Dictionary with "train" and optionally "val" keys
        task: Either "classification" or "regression"
        cv_results: Optional cross-validation results
        
    Returns:
        Evidence object with all collected data
        
    Raises:
        ValueError: If validation fails
    """
    # Validate inputs
    est_validation = validate_estimator(estimator)
    est_validation.raise_if_invalid()
    
    data_validation = validate_datasets(datasets, cv_results)
    data_validation.raise_if_invalid()
    
    if cv_results is not None:
        cv_validation = validate_cv_results(cv_results)
        cv_validation.raise_if_invalid()
    
    # Issue any warnings
    for warning in est_validation.warnings + data_validation.warnings:
        warnings.warn(warning)
    
    # Parse task type
    if isinstance(task, str):
        task = TaskType(task.lower())
    
    # Extract data
    X_train, y_train = datasets["train"]
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    
    X_val = None
    y_val = None
    if "val" in datasets and datasets["val"][0] is not None:
        X_val, y_val = datasets["val"]
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
    
    # Generate predictions (read-only - no fitting)
    y_pred_train = estimator.predict(X_train)
    y_pred_val = estimator.predict(X_val) if X_val is not None else None
    
    # Get probability predictions if available
    y_proba_train = None
    y_proba_val = None
    if task == TaskType.CLASSIFICATION and hasattr(estimator, "predict_proba"):
        try:
            y_proba_train = estimator.predict_proba(X_train)
            if X_val is not None:
                y_proba_val = estimator.predict_proba(X_val)
        except Exception:
            pass  # Some estimators may not support predict_proba in all cases
    
    # Extract feature names if available
    feature_names = _extract_feature_names(estimator, X_train)
    
    # Extract feature importances if available
    feature_importances = _extract_feature_importances(estimator)
    
    return Evidence(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        y_pred_train=y_pred_train,
        y_pred_val=y_pred_val,
        y_proba_train=y_proba_train,
        y_proba_val=y_proba_val,
        cv_results=cv_results,
        task=task,
        feature_names=feature_names,
        feature_importances=feature_importances
    )


def _extract_feature_names(
    estimator: BaseEstimator,
    X: np.ndarray
) -> Optional[list]:
    """
    Attempt to extract feature names from estimator or data.
    
    Falls back gracefully if not available.
    """
    # Try to get from Pipeline preprocessor
    if isinstance(estimator, Pipeline):
        for name, step in estimator.named_steps.items():
            if hasattr(step, "get_feature_names_out"):
                try:
                    return list(step.get_feature_names_out())
                except Exception:
                    pass
            if hasattr(step, "feature_names_in_"):
                try:
                    return list(step.feature_names_in_)
                except Exception:
                    pass
    
    # Try to get from estimator directly
    if hasattr(estimator, "feature_names_in_"):
        try:
            return list(estimator.feature_names_in_)
        except Exception:
            pass
    
    # Generate default names based on shape
    if len(X.shape) > 1:
        return [f"feature_{i}" for i in range(X.shape[1])]
    
    return None


def get_estimator_type(estimator: BaseEstimator) -> str:
    """
    Get a human-readable description of the estimator type.
    
    Args:
        estimator: A scikit-learn estimator
        
    Returns:
        String description of the estimator type
    """
    if isinstance(estimator, Pipeline):
        # Get the final step (actual model)
        final_step = estimator.steps[-1][1]
        return f"Pipeline[{type(final_step).__name__}]"
    else:
        return type(estimator).__name__


def is_pipeline(estimator: BaseEstimator) -> bool:
    """Check if estimator is a Pipeline."""
    return isinstance(estimator, Pipeline)


def _extract_feature_importances(estimator: BaseEstimator) -> Optional[np.ndarray]:
    """
    Extract feature importances from estimator if available.
    
    Supports:
    - Tree-based models (feature_importances_)
    - Linear models (coef_ as proxy)
    - Pipelines (extracts from final estimator)
    
    Args:
        estimator: A fitted scikit-learn estimator
        
    Returns:
        Array of feature importances, or None if not available
    """
    try:
        # For tree-based models (RandomForest, GradientBoosting, etc.)
        if hasattr(estimator, "feature_importances_"):
            return np.array(estimator.feature_importances_)
        
        # For linear models (use absolute coefficients as importance proxy)
        if hasattr(estimator, "coef_"):
            coefs = np.array(estimator.coef_)
            if coefs.ndim > 1:
                # Multi-class: average absolute coefficients across classes
                coefs = np.abs(coefs).mean(axis=0)
            else:
                coefs = np.abs(coefs)
            return coefs
        
        # For pipelines, look at the final estimator
        if isinstance(estimator, Pipeline):
            final_step = estimator.steps[-1][1]
            return _extract_feature_importances(final_step)
            
    except Exception:
        pass
    
    return None
