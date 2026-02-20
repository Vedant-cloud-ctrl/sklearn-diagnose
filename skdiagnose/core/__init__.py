"""
Core module for sklearn-diagnose.

This module provides the foundational components:
- schemas: Data structures and type definitions
- evidence: Evidence collection and validation
- signals: Deterministic signal extraction
- hypotheses: Reference rule-based hypothesis generation (LLM is primary)
- recommendations: Example recommendation templates for LLM guidance
"""

from .evidence import (
    collect_evidence,
    get_estimator_type,
    is_pipeline,
    validate_cv_results,
    validate_datasets,
    validate_estimator,
)
from .hypotheses import generate_hypotheses
from .recommendations import (
    get_example_recommendations_for_failure_mode,
    get_all_failure_modes_with_examples,
    get_insufficient_evidence_message,
    RECOMMENDATION_TEMPLATES,
)
from .schemas import (
    ConfidenceLevel,
    DiagnosisReport,
    Evidence,
    FailureMode,
    Hypothesis,
    Recommendation,
    Signals,
    TaskType,
    ValidationResult,
)
from .signals import analyze_cv_stability, compute_score, extract_all_signals

__all__ = [
    # Schemas
    "TaskType",
    "FailureMode",
    "ConfidenceLevel",
    "Evidence",
    "Signals",
    "Hypothesis",
    "Recommendation",
    "DiagnosisReport",
    "ValidationResult",
    # Evidence
    "validate_estimator",
    "validate_datasets",
    "validate_cv_results",
    "collect_evidence",
    "get_estimator_type",
    "is_pipeline",
    # Signals
    "extract_all_signals",
    "compute_score",
    "analyze_cv_stability",
    # Hypotheses
    "generate_hypotheses",
    # Recommendations
    "get_example_recommendations_for_failure_mode",
    "get_all_failure_modes_with_examples",
    "get_insufficient_evidence_message",
    "RECOMMENDATION_TEMPLATES",
]
