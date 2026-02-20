"""
Type definitions and data structures for sklearn-diagnose.

This module defines the core data structures used throughout the library
for representing evidence, hypotheses, and diagnosis reports.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class TaskType(str, Enum):
    """Supported ML task types."""
    
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class FailureMode(str, Enum):
    """Recognized model failure modes."""
    
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    HIGH_VARIANCE = "high_variance"
    LABEL_NOISE = "label_noise"
    FEATURE_REDUNDANCY = "feature_redundancy"
    CLASS_IMBALANCE = "class_imbalance"
    DATA_LEAKAGE = "data_leakage"
    DISTRIBUTION_SHIFT = "distribution_shift"
    MULTICOLLINEARITY = "multicollinearity"
    TARGET_LEAKAGE = "target_leakage"
    FEATURE_IMPORTANCE_SKEW = "feature_importance_skew"


class ConfidenceLevel(str, Enum):
    """Human-readable confidence categories."""
    
    HIGH = "high"          # >= 0.75
    MEDIUM = "medium"      # >= 0.50
    LOW = "low"            # >= 0.25
    INSUFFICIENT = "insufficient"  # < 0.25
    
    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Convert numeric confidence to categorical level."""
        if score >= 0.75:
            return cls.HIGH
        elif score >= 0.50:
            return cls.MEDIUM
        elif score >= 0.25:
            return cls.LOW
        else:
            return cls.INSUFFICIENT


@dataclass
class Evidence:
    """
    Container for all evidence used in diagnosis.
    
    This is the input to the signal extraction layer.
    """
    
    # Core data
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    
    # Predictions (computed internally)
    y_pred_train: Optional[np.ndarray] = None
    y_pred_val: Optional[np.ndarray] = None
    y_proba_train: Optional[np.ndarray] = None
    y_proba_val: Optional[np.ndarray] = None
    
    # Cross-validation results
    cv_results: Optional[Dict[str, Any]] = None
    
    # Task metadata
    task: TaskType = TaskType.CLASSIFICATION
    
    # Feature information
    feature_names: Optional[List[str]] = None
    feature_importances: Optional[np.ndarray] = None  # From estimator
    
    @property
    def has_validation_set(self) -> bool:
        """Check if explicit validation set is provided."""
        return self.X_val is not None and self.y_val is not None
    
    @property
    def has_cv_results(self) -> bool:
        """Check if cross-validation results are provided."""
        return self.cv_results is not None
    
    @property
    def n_samples_train(self) -> int:
        """Number of training samples."""
        return len(self.X_train)
    
    @property
    def n_samples_val(self) -> Optional[int]:
        """Number of validation samples."""
        return len(self.X_val) if self.X_val is not None else None
    
    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.X_train.shape[1] if len(self.X_train.shape) > 1 else 1


@dataclass
class Signals:
    """
    Computed statistics from evidence.
    
    This is the output of the signal extraction layer and input to the LLM
    for hypothesis generation. All values are deterministic computations.
    """
    
    # Basic performance metrics
    train_score: Optional[float] = None
    val_score: Optional[float] = None
    train_val_gap: Optional[float] = None
    
    # Cross-validation signals
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    cv_min: Optional[float] = None
    cv_max: Optional[float] = None
    cv_range: Optional[float] = None
    cv_fold_scores: Optional[List[float]] = None
    cv_train_mean: Optional[float] = None
    cv_train_val_gap: Optional[float] = None
    
    # Residual analysis (regression)
    residual_mean: Optional[float] = None
    residual_std: Optional[float] = None
    residual_skew: Optional[float] = None
    residual_kurtosis: Optional[float] = None
    
    # Classification-specific
    class_distribution: Optional[Dict[Any, float]] = None
    minority_class_ratio: Optional[float] = None
    per_class_recall: Optional[Dict[Any, float]] = None
    per_class_precision: Optional[Dict[Any, float]] = None
    confusion_matrix: Optional[np.ndarray] = None
    
    # Feature analysis
    feature_correlations: Optional[np.ndarray] = None
    high_correlation_pairs: Optional[List[Tuple[int, int, float]]] = None
    feature_importances: Optional[np.ndarray] = None
    feature_target_correlations: Optional[np.ndarray] = None
    feature_importance_top_n: Optional[List[Tuple[int, float]]] = None  # Top N features by importance
    feature_importance_concentration: Optional[float] = None  # % of importance in top 3 features
    
    # Distribution shift indicators
    feature_distribution_divergence: Optional[Dict[int, float]] = None  # KL divergence per feature
    distribution_shift_detected: Optional[bool] = None
    
    # Multicollinearity
    max_vif: Optional[float] = None  # Max Variance Inflation Factor
    high_vif_features: Optional[List[Tuple[int, float]]] = None
    
    # Target leakage indicators
    perfect_correlation_features: Optional[List[int]] = None  # Features with |corr| > 0.95 with target
    
    # Data quality
    n_samples_train: Optional[int] = None
    n_samples_val: Optional[int] = None
    n_features: Optional[int] = None
    feature_to_sample_ratio: Optional[float] = None
    
    # Leakage indicators
    cv_holdout_gap: Optional[float] = None
    suspicious_feature_correlations: Optional[List[Tuple[int, float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signals to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, np.ndarray):
                    result[key] = value.tolist()
                else:
                    result[key] = value
        return result


@dataclass
class Hypothesis:
    """
    A confidence-weighted hypothesis about a potential issue.
    
    This is generated by the LLM based on computed signals.
    """
    
    name: FailureMode
    confidence: float  # 0.0 to 1.0
    evidence: List[str]  # Human-readable evidence descriptions
    severity: str = "medium"  # "low", "medium", "high"
    
    def __post_init__(self):
        """Validate hypothesis after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.severity not in ("low", "medium", "high"):
            raise ValueError(f"Severity must be low/medium/high, got {self.severity}")
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get categorical confidence level."""
        return ConfidenceLevel.from_score(self.confidence)
    
    @property
    def is_actionable(self) -> bool:
        """Check if hypothesis has sufficient confidence to act on."""
        return self.confidence >= 0.25
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"{self.name.value.replace('_', ' ').title()} "
            f"(confidence: {self.confidence:.0%}, severity: {self.severity})\n"
            f"  Evidence:\n" + 
            "\n".join(f"    - {e}" for e in self.evidence)
        )
    
    def __repr__(self) -> str:
        return f"Hypothesis({self.name.value}, confidence={self.confidence:.2f}, severity={self.severity})"


@dataclass
class Recommendation:
    """
    An actionable recommendation to address an issue.
    """
    
    action: str  # What to do
    rationale: str  # Why it helps
    related_hypothesis: Optional[FailureMode] = None
    
    def __str__(self) -> str:
        result = f"{self.action}\n  Rationale: {self.rationale}"
        if self.related_hypothesis:
            result += f"\n  Addresses: {self.related_hypothesis.value}"
        return result
    
    def __repr__(self) -> str:
        return f"Recommendation({self.action[:50]}...)"


@dataclass  
class DiagnosisReport:
    """
    Complete diagnosis report output.
    
    This is the final output of the diagnose() function.
    """
    
    # Core results
    hypotheses: List[Hypothesis] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)
    signals: Signals = field(default_factory=Signals)
    
    # Metadata
    task: TaskType = TaskType.CLASSIFICATION
    estimator_type: str = ""
    has_pipeline: bool = False
    
    # LLM-generated summary (optional)
    _llm_summary: Optional[str] = None
    
    def summary(self, use_llm: bool = True) -> str:
        """
        Generate a human-readable summary of the diagnosis.
        
        Args:
            use_llm: If True and LLM is configured, use LLM for summary.
                    Otherwise, use template-based summary.
        
        Returns:
            Human-readable summary string.
        """
        if use_llm and self._llm_summary:
            return self._llm_summary
        
        return self._template_summary()
    
    def _template_summary(self) -> str:
        """Generate template-based summary without LLM."""
        if not self.hypotheses:
            return "No significant issues detected. Your model appears to be performing reasonably."
        
        lines = ["## Diagnosis Summary\n"]
        
        # Sort hypotheses by confidence
        sorted_hypotheses = sorted(
            self.hypotheses, 
            key=lambda h: h.confidence, 
            reverse=True
        )
        
        for hyp in sorted_hypotheses:
            if hyp.confidence < 0.25:
                continue
            
            level = hyp.confidence_level.value.upper()
            lines.append(f"### {hyp.name.value.replace('_', ' ').title()} ({level} confidence: {hyp.confidence:.0%})")
            lines.append("")
            lines.append("**Evidence:**")
            for ev in hyp.evidence:
                lines.append(f"- {ev}")
            lines.append("")
        
        if self.recommendations:
            lines.append("### Recommendations\n")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. **{rec.action}**")
                lines.append(f"   - {rec.rationale}")
                lines.append("")
        
        return "\n".join(lines)
    
    @property
    def top_issue(self) -> Optional[Hypothesis]:
        """Get the highest-confidence hypothesis."""
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.confidence)
    
    @property
    def actionable_issues(self) -> List[Hypothesis]:
        """Get hypotheses with sufficient confidence to act on."""
        return [h for h in self.hypotheses if h.is_actionable]
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if any high-severity, high-confidence issues exist."""
        return any(
            h.confidence >= 0.75 and h.severity == "high"
            for h in self.hypotheses
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "hypotheses": [
                {
                    "name": h.name.value,
                    "confidence": h.confidence,
                    "confidence_level": h.confidence_level.value,
                    "evidence": h.evidence,
                    "severity": h.severity
                }
                for h in self.hypotheses
            ],
            "recommendations": [
                {
                    "action": r.action,
                    "rationale": r.rationale,
                    "related_hypothesis": r.related_hypothesis.value if r.related_hypothesis else None
                }
                for r in self.recommendations
            ],
            "signals": self.signals.to_dict(),
            "task": self.task.value,
            "estimator_type": self.estimator_type,
            "has_pipeline": self.has_pipeline
        }


@dataclass
class ValidationResult:
    """Result of input validation."""
    
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def raise_if_invalid(self) -> None:
        """Raise ValueError if validation failed."""
        if not self.is_valid:
            raise ValueError(
                "Validation failed:\n" + "\n".join(f"  - {e}" for e in self.errors)
            )
