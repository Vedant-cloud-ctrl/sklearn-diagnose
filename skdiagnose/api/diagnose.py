"""
Main public API for sklearn-diagnose.

This module provides the `diagnose()` function, the primary entry point
for model diagnosis.

IMPORTANT: You must set up an LLM provider before using diagnose().
Call setup_llm() first:

    from skdiagnose import setup_llm, diagnose
    
    setup_llm(provider="openai", model="gpt-4o", api_key="your-key")
    
    report = diagnose(
        estimator=model,
        datasets={
            "train": (X_train, y_train),
            "val": (X_val, y_val)
        },
        task="classification"
    )
    
    print(report.summary())
    print(report.recommendations)
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator

from ..core import (
    DiagnosisReport,
    TaskType,
    collect_evidence,
    extract_all_signals,
    get_all_failure_modes_with_examples,
    get_estimator_type,
    get_insufficient_evidence_message,
    is_pipeline,
)
from ..llm import (
    generate_llm_hypotheses,
    generate_llm_recommendations,
    generate_llm_summary,
    _get_global_client,
)


def diagnose(
    estimator: BaseEstimator,
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    task: Union[str, TaskType],
    # cv_results: Optional[Dict[str, Any]] = None,
    max_recommendations: int = 5
) -> DiagnosisReport:
    """
    Diagnose a fitted scikit-learn estimator for common issues.
    
    This function analyzes a fitted model to detect potential problems like
    overfitting, underfitting, high variance, class imbalance, and more.
    
    **IMPORTANT**: You must call setup_llm() before using this function.
    sklearn-diagnose uses an LLM for hypothesis detection, recommendation
    generation, and summaries.
    
    The diagnosis follows an LLM-driven architecture:
    1. Signal extraction: Compute deterministic statistics from the model
    2. LLM hypothesis generation: Detect failure modes with confidence/severity
    3. LLM recommendation generation: Generate actionable recommendations
    4. LLM summary: Generate human-readable summary
    
    **Read-Only Guarantee**: This function never modifies the estimator,
    never calls fit(), and never mutates the input data.
    
    Args:
        estimator: A fitted scikit-learn estimator or Pipeline.
            Must already be fitted (will raise ValueError if not).
            Can be any sklearn estimator: LogisticRegression, RandomForest,
            Pipeline, GridSearchCV, etc.
        
        datasets: Dictionary with training and optional validation data.
            Required keys:
                - "train": Tuple of (X_train, y_train)
            Optional keys:
                - "val": Tuple of (X_val, y_val) for holdout validation
            
            Example:
                datasets={
                    "train": (X_train, y_train),
                    "val": (X_val, y_val)  # optional
                }
        
        task: The type of ML task. Either:
            - "classification" 
            - "regression"
            Or a TaskType enum value.
        
        cv_results: Optional cross-validation results from cross_validate().
            Provides richer diagnostic signals than holdout validation alone.
            
            Example:
                from sklearn.model_selection import cross_validate
                cv_results = cross_validate(
                    model, X, y, cv=5, 
                    return_train_score=True
                )
        
        max_recommendations: Maximum number of recommendations to return.
            The LLM will generate up to this many recommendations.
            Default is 5.
    
    Returns:
        DiagnosisReport: A report containing:
            - hypotheses: List of detected issues with confidence scores
            - recommendations: List of actionable recommendations
            - signals: Raw computed statistics
            - summary(): Method to get LLM-generated summary
    
    Raises:
        RuntimeError: If no LLM provider is configured. Call setup_llm() first.
        ValueError: If estimator is not fitted, if required data is missing,
            or if input validation fails.
    
    Examples:
        Basic usage with holdout validation:
        
            >>> from sklearn.linear_model import LogisticRegression
            >>> from skdiagnose import setup_llm, diagnose
            >>> 
            >>> # Set up LLM (required)
            >>> setup_llm(provider="openai", model="gpt-4o", api_key="sk-...")
            >>> 
            >>> model = LogisticRegression()
            >>> model.fit(X_train, y_train)
            >>> 
            >>> report = diagnose(
            ...     estimator=model,
            ...     datasets={
            ...         "train": (X_train, y_train),
            ...         "val": (X_val, y_val)
            ...     },
            ...     task="classification"
            ... )
            >>> print(report.summary())
    
    Notes:
        - You MUST call setup_llm() before using diagnose()
        - The LLM determines failure modes, confidence scores, and severity
        - Recommendations are generated by the LLM based on detected issues
        - All hypotheses include confidence scores (0.0 - 0.95)
        
    See Also:
        - setup_llm(): For setting up the required LLM provider
        - DiagnosisReport: For understanding the report structure
    """
    # Verify LLM is configured (fail fast)
    if _get_global_client() is None:
        raise RuntimeError(
            "No LLM provider configured. Call setup_llm() first.\n"
            "Example: setup_llm(provider='openai', model='gpt-4o', api_key='sk-...')"
        )
    
    # Parse task type
    if isinstance(task, str):
        task = TaskType(task.lower())
    
    # Layer 1: Collect and validate evidence
    evidence = collect_evidence(
        estimator=estimator,
        datasets=datasets,
        task=task,
        # cv_results=cv_results
    )
    
    # Layer 2: Extract signals (deterministic statistics)
    signals = extract_all_signals(evidence)
    signals_dict = signals.to_dict()
    
    # Layer 3: LLM generates hypotheses from signals
    hypotheses = generate_llm_hypotheses(
        signals=signals_dict,
        task=task.value
    )
    
    # Layer 4a: LLM generates recommendations from hypotheses
    # Provide example recommendations as guidance
    example_recommendations = get_all_failure_modes_with_examples()
    recommendations = generate_llm_recommendations(
        hypotheses=hypotheses,
        example_recommendations=example_recommendations,
        max_recommendations=max_recommendations
    )
    
    # Add CV recommendation if no validation/CV data was provided
    if signals.val_score is None and signals.cv_mean is None:
        from skdiagnose.core.schemas import Recommendation
        cv_recommendation = Recommendation(
            action="Use cross-validation for robust model evaluation and hyperparameter tuning",
            rationale="Cross-validation (e.g., sklearn.model_selection.cross_val_score or GridSearchCV) "
                      "provides more reliable performance estimates by testing on multiple data splits. "
                      "This helps detect overfitting and ensures hyperparameters generalize well to unseen data.",
            related_hypothesis=None
        )
        recommendations.append(cv_recommendation)
    
    # Layer 4b: Generate LLM summary (includes hypotheses and recommendations)
    llm_summary = generate_llm_summary(
        hypotheses=hypotheses,
        recommendations=recommendations,
        signals=signals_dict,
        task=task.value
    )
    
    # Check for insufficient evidence
    insufficient_msg = get_insufficient_evidence_message(signals)
    if insufficient_msg:
        llm_summary = insufficient_msg + "\n\n" + llm_summary
    
    # Build the report
    report = DiagnosisReport(
        hypotheses=hypotheses,
        recommendations=recommendations,
        signals=signals,
        task=task,
        estimator_type=get_estimator_type(estimator),
        has_pipeline=is_pipeline(estimator)
    )
    
    # Attach LLM summary
    report._llm_summary = llm_summary
    
    return report
