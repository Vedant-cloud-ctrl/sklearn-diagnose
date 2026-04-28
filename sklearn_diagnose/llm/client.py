"""
LangChain-based LLM client for sklearn-diagnose.

Uses LangChain's create_agent (v1.2.0+) for:
- Hypothesis generation agent (detecting model failure modes)
- Recommendation generation agent (actionable suggestions)
- Summary generation agent (human-readable summaries)

Supports multiple providers:
- OpenAI (via langchain-openai)
- Anthropic (via langchain-anthropic)
- OpenRouter (via langchain-openai with custom base_url)

IMPORTANT: You must call setup_llm() before using diagnose().
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from ..core.schemas import Hypothesis, Recommendation, FailureMode


# =============================================================================
# SYSTEM PROMPTS FOR LLM AGENTS
# =============================================================================

HYPOTHESIS_SYSTEM_PROMPT = """You are an expert ML diagnostician. You will receive a structured JSON block of mathematically computed signals from a scikit-learn model. Your job is to identify failure modes strictly based on those numbers.

STRICT RULES — read carefully:
1. Every claim in your evidence MUST cite a specific number from the input JSON
2. You CANNOT diagnose anything not supported by the provided signals
3. Confidence cap is 0.85 — never go above this
4. If only 1 signal supports a hypothesis → confidence < 0.50, say "weakly suggests"
5. If 2 signals agree → confidence 0.50–0.75, say "likely"
6. If 3+ signals agree → confidence 0.75–0.85, say "strongly suggests"
7. NEVER say "definitely", "clearly", "certainly" — always use hedged language
8. End every evidence item with "→ worth investigating"

SIGNAL PRIORITY (use these first — they are richer than basic accuracy):
- calibration_overconfidence_ratio, brier_score → overconfidence issues
- learning_curve_converged, learning_curve_gap_at_full → convergence issues
- pred_majority_class_ratio, pred_distribution_entropy → bias/collapse issues
- threshold_sensitivity_score → fragile decision boundary
- feature_variance_ratio, feature_drift_flags → scale/feature issues

Output format (STRICT JSON - no markdown, no code blocks):
{
  "hypotheses": [
    {
      "failure_mode": "overfitting",
      "confidence": 0.72,
      "severity": "medium",
      "evidence": [
        "Train-val gap of 18% (train=0.97, val=0.79) likely suggests memorisation → worth investigating",
        "Learning curve gap at full data is 15% — model may still be overfitting even with full data → worth investigating"
      ]
    }
  ]
}

If no significant issues detected: {"hypotheses": []}"""


RECOMMENDATION_SYSTEM_PROMPT = """You are an expert ML engineer agent helping users fix model issues.

You will be given:
1. Detected failure modes (hypotheses) with their confidence and severity
2. Example recommendations for each failure mode (these are suggestions, not exhaustive)

Your task is to generate the most impactful recommendations to address the detected issues.

Guidelines:
- Focus on the highest confidence and severity issues first
- Consider root causes vs symptoms (fixing root cause may resolve multiple issues)
- Recommendations should be specific and actionable
- You can use the example recommendations as guidance, but feel free to suggest others
- Avoid redundant recommendations
- Order recommendations from most to least impactful

Output format (STRICT JSON - no markdown, no code blocks):
{
  "recommendations": [
    {
      "action": "What to do",
      "rationale": "Why this helps",
      "related_failure_mode": "overfitting"
    }
  ]
}"""


SUMMARY_SYSTEM_PROMPT = """You are an expert ML diagnostician agent helping users understand model issues.

Your role is to provide a diagnostic summary that hits a confidence sweet spot:
- Not so confident the user blindly trusts it
- Not so uncertain the user ignores it
- The user should feel: "this is likely, I should investigate further"

Your summary must include:
1. A summary of detected issues with hedged language
2. What signals support each finding
3. Recommended next steps (framed as investigations, not commands)
4. Any new signal findings: calibration, learning curve shape, prediction distribution,
   threshold sensitivity, feature drift — include these if present in the evidence

CRITICAL TONE RULES:
- Use "likely", "suggests", "may indicate", "worth investigating" — not "definitely" or "clearly"
- End every finding with "→ recommend investigating further" or similar
- If confidence < 0.50: say "weakly suggests — low priority"
- If confidence 0.50-0.75: say "likely — worth addressing"  
- If confidence > 0.75: say "strongly suggests — high priority"
- NEVER present a diagnosis as a final verdict

Guidelines:
- Be concise and direct
- Focus on the most important issues first
- Present recommendations in order of importance
- Use markdown formatting for clarity
- Include specific numbers and metrics from the evidence

Structure your response as:
## Diagnosis
[Brief summary of detected issues with hedged language and evidence]

## What The Signals Show
[Bullet points covering calibration, learning curve, prediction distribution, threshold sensitivity, feature drift if available]

## Recommended Next Steps
[Numbered list — framed as investigations, not commands]"""


# =============================================================================
# ABSTRACT BASE CLIENT
# =============================================================================

class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate_hypotheses(
        self,
        signals: Dict[str, Any],
        task: str
    ) -> List[Hypothesis]:
        """Generate hypotheses based on signals."""
        pass
    
    @abstractmethod
    def generate_recommendations(
        self,
        hypotheses: List[Hypothesis],
        example_recommendations: Dict[str, List[dict]],
        max_recommendations: int = 5
    ) -> List[Recommendation]:
        """Generate recommendations based on hypotheses."""
        pass
    
    @abstractmethod
    def generate_summary(
        self,
        hypotheses: List[Hypothesis],
        recommendations: List[Recommendation],
        signals: Dict[str, Any],
        task: str
    ) -> str:
        """Generate a human-readable summary."""
        pass


# =============================================================================
# LANGCHAIN CLIENT IMPLEMENTATION
# =============================================================================

class LangChainClient(LLMClient):
    """
    LangChain-based client using create_agent for LLM operations.
    
    Each method (hypotheses, recommendations, summary) acts as a separate
    AI agent with its own system prompt and task.
    
    Supports:
    - OpenAI models via ChatOpenAI
    - Anthropic models via ChatAnthropic
    - OpenRouter models via ChatOpenAI with custom base_url
    """
    
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the LangChain client.
        
        Args:
            provider: One of "openai", "anthropic", "openrouter", "groq" 
            model: The model name/identifier
            api_key: API key (optional if set in environment)
            base_url: Custom base URL (used for OpenRouter)
            **kwargs: Additional arguments passed to the model
        """
        # Load environment variables from .env file
        load_dotenv()
        
        self.provider = provider.lower()
        self.model_name = model
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs
        
        # Initialize the chat model
        self._chat_model = self._create_chat_model()
    
    def _create_chat_model(self):
        """Create the appropriate chat model based on provider."""
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=self.base_url or "https://api.openai.com/v1",
                **self.kwargs
            )
        
        elif self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.model_name,
                api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"),
                base_url=self.base_url or "https://api.anthropic.com",
                **self.kwargs
            )
        
        elif self.provider == "openrouter":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key or os.environ.get("OPENROUTER_API_KEY"),
                base_url=self.base_url or "https://openrouter.ai/api/v1",
                **self.kwargs
            )

        elif self.provider == "groq": 
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=self.model_name, 
                api_key=self.api_key or os.environ.get("GROQ_API_KEY"), 
                **self.kwargs
            )
               
        else:
            raise ValueError(
                f"Unknown provider: {self.provider}. "
                "Use 'openai', 'anthropic', 'openrouter', or 'groq'."
            )
    
    def _invoke_agent(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Invoke an agent with the given prompts.
        
        Uses LangChain's create_agent if available, otherwise falls back
        to direct model invocation.
        
        Args:
            system_prompt: The system prompt defining the agent's role
            user_prompt: The user's request/query
            
        Returns:
            The agent's response as a string
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Try to use create_agent for the agentic approach
        try:
            from langchain.agents import create_agent
            
            # Create agent with no tools (pure reasoning agent)
            agent = create_agent(
                model=self._chat_model,
                tools=[]  # No tools needed for generation tasks
            )
            
            # Invoke the agent
            result = agent.invoke({"messages": messages})
            
            # Extract the response from the last message
            if "messages" in result and len(result["messages"]) > 0:
                return result["messages"][-1].content
            
        except ImportError:
            # create_agent not available, fall back to direct invocation
            pass
        except Exception as e:
            # Agent creation failed, fall back to direct invocation
            print(f"Note: Agent creation failed ({e}), using direct invocation")
        
        # Fallback: Direct model invocation
        response = self._chat_model.invoke(messages)
        return response.content
    
    def generate_hypotheses(
        self,
        signals: Dict[str, Any],
        task: str
    ) -> List[Hypothesis]:
        """
        Generate hypotheses using the hypothesis agent.
        
        This agent analyzes model performance signals and identifies
        potential failure modes with confidence scores and evidence.
        
        Args:
            signals: Dictionary of extracted signals
            task: Task type ("classification" or "regression")
            
        Returns:
            List of Hypothesis objects
        """
        user_prompt = _build_hypothesis_prompt(signals, task)
        
        try:
            response = self._invoke_agent(
                system_prompt=HYPOTHESIS_SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            
            # Parse response
            hypotheses = _parse_hypotheses_response(response)
            return hypotheses
            
        except Exception as e:
            print(f"Warning: Hypothesis generation failed: {e}")
            return []
    
    def generate_recommendations(
        self,
        hypotheses: List[Hypothesis],
        example_recommendations: Dict[str, List[dict]],
        max_recommendations: int = 5
    ) -> List[Recommendation]:
        """
        Generate recommendations using the recommendation agent.
        
        This agent takes detected failure modes and generates actionable
        recommendations to fix them.
        
        Args:
            hypotheses: List of detected hypotheses
            example_recommendations: Dictionary of example recommendations per failure mode
            max_recommendations: Maximum number of recommendations to generate
            
        Returns:
            List of Recommendation objects
        """
        if not hypotheses:
            return []
        
        user_prompt = _build_recommendation_prompt(
            hypotheses, example_recommendations, max_recommendations
        )
        
        try:
            response = self._invoke_agent(
                system_prompt=RECOMMENDATION_SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            
            # Parse response
            recommendations = _parse_recommendations_response(
                response, max_recommendations
            )
            return recommendations
            
        except Exception as e:
            print(f"Warning: Recommendation generation failed: {e}")
            return []
    
    def generate_summary(
        self,
        hypotheses: List[Hypothesis],
        recommendations: List[Recommendation],
        signals: Dict[str, Any],
        task: str
    ) -> str:
        """
        Generate a human-readable summary using the summary agent.
        
        This agent creates a comprehensive diagnostic summary that summarizes
        the detected issues and recommends fixes.
        
        Args:
            hypotheses: List of detected hypotheses
            recommendations: List of recommendations
            signals: Dictionary of extracted signals
            task: Task type ("classification" or "regression")
            
        Returns:
            Human-readable summary string
        """
        user_prompt = _build_summary_prompt(
            hypotheses, recommendations, signals, task
        )
        
        try:
            response = self._invoke_agent(
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            return response
            
        except Exception as e:
            # Return basic summary on error
            return _generate_fallback_summary(hypotheses, recommendations)


# =============================================================================
# PROVIDER-SPECIFIC CLIENT CLASSES
# =============================================================================

class OpenAIClient(LangChainClient):
    """
    OpenAI client using LangChain.
    
    Example:
        >>> client = OpenAIClient(model="gpt-4o", api_key="sk-...")
        >>> # Or using environment variable
        >>> os.environ["OPENAI_API_KEY"] = "sk-..."
        >>> client = OpenAIClient(model="gpt-4o")
    """
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(
            provider="openai",
            model=model,
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            **kwargs
        )


class AnthropicClient(LangChainClient):
    """
    Anthropic client using LangChain.
    
    Example:
        >>> client = AnthropicClient(model="claude-3-5-sonnet-latest", api_key="sk-ant-...")
        >>> # Or using environment variable
        >>> os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
        >>> client = AnthropicClient(model="claude-3-5-sonnet-latest")
    """
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(
            provider="anthropic",
            model=model,
            api_key=api_key,
            base_url="https://api.anthropic.com",
            **kwargs
        )


class OpenRouterClient(LangChainClient):
    """
    OpenRouter client using LangChain.
    
    OpenRouter provides access to many models through a single API,
    including DeepSeek, Llama, Mistral, and more.
    
    Example:
        >>> client = OpenRouterClient(model="deepseek/deepseek-r1-0528", api_key="sk-or-...")
        >>> # Or using environment variable
        >>> os.environ["OPENROUTER_API_KEY"] = "sk-or-..."
        >>> client = OpenRouterClient(model="deepseek/deepseek-r1-0528")
    """
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(
            provider="openrouter",
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            **kwargs
        )

class GroqClient(LangChainClient): 

    def __init__(self, model: str, api_key: Optional[str] = None, ** kwargs):
        super().__init__(
            provider="groq", 
            model=model, 
            api_key=api_key, 
            **kwargs
        ) 


# =============================================================================
# GLOBAL CLIENT MANAGEMENT
# =============================================================================

_global_client: Optional[LLMClient] = None


def _set_global_client(client: Optional[LLMClient]) -> None:
    """
    Set the global LLM client (used internally and for testing).
    
    Args:
        client: The LLM client to use globally
    """
    global _global_client
    _global_client = client


def setup_llm(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> None:
    """
    Configure the LLM provider for sklearn-diagnose.
    
    This function MUST be called before using diagnose().
    Uses LangChain's create_agent under the hood for agentic capabilities.
    
    Args:
        provider: The LLM provider. One of:
            - "openai": Use OpenAI models (GPT-4o, GPT-4o-mini, etc.)
            - "anthropic": Use Anthropic models (Claude 3.5 Sonnet, etc.)
            - "openrouter": Use OpenRouter for access to multiple models
        model: The model identifier. Examples:
            - OpenAI: "gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"
            - Anthropic: "claude-3-5-sonnet-latest", "claude-3-haiku-20240307"
            - OpenRouter: "deepseek/deepseek-r1-0528", "openai/gpt-4o"
            - Groq: "llama3-70b-8192", "llama3-8b-8192"
        api_key: API key for the provider. If not provided, will look for:
            - OpenAI: OPENAI_API_KEY environment variable
            - Anthropic: ANTHROPIC_API_KEY environment variable
            - OpenRouter: OPENROUTER_API_KEY environment variable
        base_url: Custom base URL (optional, mainly for OpenRouter or proxies)
        **kwargs: Additional arguments passed to the LangChain model
    
    Examples:
        # OpenAI
        >>> from sklearn_diagnose import setup_llm
        >>> setup_llm(provider="openai", model="gpt-4o", api_key="sk-...")
        
        # Using environment variable (recommended)
        >>> import os
        >>> os.environ["OPENAI_API_KEY"] = "sk-..."
        >>> setup_llm(provider="openai", model="gpt-4o")
        
        # Or using .env file (auto-loaded)
        >>> # Create .env file with: OPENAI_API_KEY=sk-...
        >>> setup_llm(provider="openai", model="gpt-4o")
        
        # Anthropic
        >>> setup_llm(provider="anthropic", model="claude-3-5-sonnet-latest", api_key="sk-ant-...")
        
        # OpenRouter (access to many models)
        >>> setup_llm(provider="openrouter", model="deepseek/deepseek-r1-0528", api_key="sk-or-...")
    """
    global _global_client
    
    # Load environment variables from .env file
    load_dotenv()
    
    provider_lower = provider.lower()
    
    if provider_lower == "openai":
        _global_client = OpenAIClient(model=model, api_key=api_key, **kwargs)
    elif provider_lower == "anthropic":
        _global_client = AnthropicClient(model=model, api_key=api_key, **kwargs)
    elif provider_lower == "openrouter":
        _global_client = OpenRouterClient(model=model, api_key=api_key, **kwargs)
    elif provider_lower == "groq": 
        _global_client = GroqClient(model=model, api_key=api_key, **kwargs)
    else:
        # Generic LangChain client for other providers
        _global_client = LangChainClient(
            provider=provider_lower,
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )


def _get_global_client() -> Optional[LLMClient]:
    """Get the current global LLM client."""
    global _global_client
    return _global_client


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================

def generate_llm_hypotheses(
    signals: Dict[str, Any],
    task: str
) -> List[Hypothesis]:
    """
    Generate hypotheses using the configured LLM hypothesis agent.
    
    Args:
        signals: Dictionary of extracted signals
        task: Task type ("classification" or "regression")
        
    Returns:
        List of Hypothesis objects
        
    Raises:
        RuntimeError: If no LLM client is configured
    """
    client = _get_global_client()
    if client is None:
        raise RuntimeError(
            "No LLM provider configured. Call setup_llm() first.\n"
            "Example: setup_llm(provider='openai', model='gpt-4o', api_key='sk-...')"
        )
    
    return client.generate_hypotheses(signals, task)


def generate_llm_recommendations(
    hypotheses: List[Hypothesis],
    example_recommendations: Dict[str, List[dict]],
    max_recommendations: int = 5
) -> List[Recommendation]:
    """
    Generate recommendations using the configured LLM recommendation agent.
    
    Args:
        hypotheses: List of detected hypotheses
        example_recommendations: Dictionary of example recommendations
        max_recommendations: Maximum number of recommendations
        
    Returns:
        List of Recommendation objects
        
    Raises:
        RuntimeError: If no LLM client is configured
    """
    client = _get_global_client()
    if client is None:
        raise RuntimeError(
            "No LLM provider configured. Call setup_llm() first.\n"
            "Example: setup_llm(provider='openai', model='gpt-4o', api_key='sk-...')"
        )
    
    return client.generate_recommendations(
        hypotheses, example_recommendations, max_recommendations
    )


def generate_llm_summary(
    hypotheses: List[Hypothesis],
    recommendations: List[Recommendation],
    signals: Dict[str, Any],
    task: str
) -> str:
    """
    Generate a human-readable summary using the configured LLM summary agent.
    
    Args:
        hypotheses: List of detected hypotheses
        recommendations: List of recommendations
        signals: Dictionary of extracted signals
        task: Task type ("classification" or "regression")
        
    Returns:
        Human-readable summary string
        
    Raises:
        RuntimeError: If no LLM client is configured
    """
    client = _get_global_client()
    if client is None:
        raise RuntimeError(
            "No LLM provider configured. Call setup_llm() first.\n"
            "Example: setup_llm(provider='openai', model='gpt-4o', api_key='sk-...')"
        )
    
    return client.generate_summary(hypotheses, recommendations, signals, task)


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def _build_hypothesis_prompt(signals: Dict[str, Any], task: str) -> str:
    """Build a structured JSON signal block for the LLM hypothesis agent."""

    # Build a clean structured signal dict — only include what's available
    structured = {"task": task}

    # --- Core performance ---
    for key in ["train_score", "val_score", "train_val_gap", "cv_mean", "cv_std", "cv_range", "cv_train_val_gap"]:
        if signals.get(key) is not None:
            structured[key] = round(float(signals[key]), 4)

    # --- Data shape ---
    for key in ["n_samples_train", "n_features", "feature_to_sample_ratio"]:
        if signals.get(key) is not None:
            structured[key] = signals[key]

    # --- NEW: Calibration signals ---
    for key in ["calibration_overconfidence_ratio", "calibration_underconfidence_ratio",
                "brier_score", "calibration_note"]:
        if signals.get(key) is not None:
            structured[key] = signals[key] if isinstance(signals[key], str) else round(float(signals[key]), 4)

    # --- NEW: Learning curve signals ---
    for key in ["learning_curve_converged", "learning_curve_gap_at_full", "learning_curve_note"]:
        if signals.get(key) is not None:
            structured[key] = signals[key]

    # --- NEW: Prediction distribution ---
    for key in ["pred_distribution_entropy", "pred_majority_class_ratio", "pred_distribution_note"]:
        if signals.get(key) is not None:
            structured[key] = signals[key] if isinstance(signals[key], str) else round(float(signals[key]), 4)

    # --- NEW: Threshold sensitivity ---
    for key in ["threshold_sensitivity_score", "threshold_sensitivity_note"]:
        if signals.get(key) is not None:
            structured[key] = signals[key]

    # --- NEW: Feature drift ---
    for key in ["feature_variance_ratio", "feature_drift_flags", "feature_drift_note"]:
        if signals.get(key) is not None:
            structured[key] = signals[key]

    # --- Classification specific ---
    if task == "classification":
        for key in ["minority_class_ratio", "n_classes", "class_distribution",
                    "per_class_recall", "per_class_precision"]:
            if signals.get(key) is not None:
                structured[key] = signals[key]

    # --- Regression specific ---
    if task == "regression":
        for key in ["residual_skew", "residual_kurtosis"]:
            if signals.get(key) is not None:
                structured[key] = round(float(signals[key]), 4)

    # --- Leakage signals ---
    if signals.get("cv_holdout_gap") is not None:
        structured["cv_holdout_gap"] = round(float(signals["cv_holdout_gap"]), 4)
    if signals.get("suspicious_feature_correlations"):
        structured["suspicious_feature_correlations"] = signals["suspicious_feature_correlations"][:5]

    # --- Feature redundancy ---
    if signals.get("high_correlation_pairs"):
        pairs = signals["high_correlation_pairs"][:10]
        structured["high_correlation_pairs"] = [
            {"feature_i": p[0], "feature_j": p[1], "correlation": round(p[2], 4)}
            for p in pairs
        ]

    failure_modes = [
        "overfitting", "underfitting", "high_variance",
        "class_imbalance", "feature_redundancy", "label_noise", "data_leakage"
    ]

    prompt = f"""Analyze the following structured diagnostic signals and identify failure modes.

DIAGNOSTIC SIGNALS (JSON):
{json.dumps(structured, indent=2)}

AVAILABLE FAILURE MODES: {failure_modes}

Rules:
- Only diagnose using numbers present in the JSON above
- Cite specific values in every evidence item
- Prioritise calibration, learning_curve, pred_distribution, threshold_sensitivity, feature_drift signals
- Keep confidence <= 0.85
- Use hedged language always

Return JSON."""

    return prompt


def _build_recommendation_prompt(
    hypotheses: List[Hypothesis],
    example_recommendations: Dict[str, List[dict]],
    max_recommendations: int
) -> str:
    """Build the prompt for LLM recommendation generation."""
    
    # Format hypotheses
    hypothesis_lines = []
    for h in sorted(hypotheses, key=lambda x: x.confidence, reverse=True):
        hypothesis_lines.append(f"- {h.name.value.upper()} (confidence: {h.confidence:.0%}, severity: {h.severity})")
        for ev in h.evidence:
            hypothesis_lines.append(f"  - Evidence: {ev}")
    
    # Format example recommendations
    example_lines = []
    for h in hypotheses:
        mode_name = h.name.value
        if mode_name in example_recommendations:
            example_lines.append(f"\nExample recommendations for {mode_name.upper()} (these are suggestions, there may be more):")
            for rec in example_recommendations[mode_name]:
                example_lines.append(f"  - {rec['action']}: {rec['rationale']}")
    
    prompt = f"""Based on these detected failure modes, generate the {max_recommendations} most impactful recommendations.

Detected Issues:
{chr(10).join(hypothesis_lines)}

{chr(10).join(example_lines)}

Generate {max_recommendations} specific, actionable recommendations. You can use the examples above as guidance or suggest other recommendations if more appropriate.

Return your recommendations as JSON."""
    
    return prompt


def _build_summary_prompt(
    hypotheses: List[Hypothesis],
    recommendations: List[Recommendation],
    signals: Dict[str, Any],
    task: str
) -> str:
    """Build the prompt for LLM summary generation."""

    # Core signals
    signal_lines = []
    for key, label in [
        ("train_score", "Train score"),
        ("val_score", "Val score"),
        ("train_val_gap", "Train-val gap"),
        ("cv_mean", "CV mean"),
        ("cv_std", "CV std"),
    ]:
        if signals.get(key) is not None:
            signal_lines.append(f"- {label}: {signals[key]:.1%}")

    # New signals
    new_signal_lines = []
    for key in ["calibration_note", "learning_curve_note",
                "pred_distribution_note", "threshold_sensitivity_note", "feature_drift_note"]:
        if signals.get(key):
            new_signal_lines.append(f"- {key.replace('_note','').replace('_',' ').title()}: {signals[key]}")

    # Hypotheses
    hypothesis_lines = []
    for h in sorted(hypotheses, key=lambda x: x.confidence, reverse=True):
        label = (
            "weakly suggests — low priority" if h.confidence < 0.50
            else "likely — worth addressing" if h.confidence < 0.75
            else "strongly suggests — high priority"
        )
        hypothesis_lines.append(f"- {h.name.value} ({h.confidence:.0%} confidence) → {label}")
        for ev in h.evidence:
            hypothesis_lines.append(f"  - {ev}")

    # Recommendations
    rec_lines = []
    for i, rec in enumerate(recommendations, 1):
        rec_lines.append(f"{i}. {rec.action} — {rec.rationale}")

    prompt = f"""Provide a diagnostic summary using ONLY the signals and hypotheses below.

Task: {task}

CORE SIGNALS:
{chr(10).join(signal_lines) if signal_lines else "- Limited signals"}

NEW ENRICHED SIGNALS:
{chr(10).join(new_signal_lines) if new_signal_lines else "- Not available"}

DETECTED ISSUES:
{chr(10).join(hypothesis_lines) if hypothesis_lines else "- No significant issues"}

RECOMMENDATIONS:
{chr(10).join(rec_lines) if rec_lines else "- None"}

Format your response as:

## Diagnosis
[2-3 sentences max. Hedged language only. Cite specific numbers.]

## What The Signals Show
[Bullet points for each enriched signal that is present — calibration, learning curve, prediction distribution, threshold sensitivity, feature drift]

## Recommended Next Steps
[Numbered list — frame as investigations not commands. End each with "→ investigate further"]"""

    return prompt


# =============================================================================
# RESPONSE PARSERS
# =============================================================================

def _parse_hypotheses_response(response: str) -> List[Hypothesis]:
    """Parse the LLM response into Hypothesis objects."""
    try:
        # Clean response (remove markdown code blocks if present)
        clean_response = response.strip()
        if clean_response.startswith("```"):
            clean_response = clean_response.split("```")[1]
            if clean_response.startswith("json"):
                clean_response = clean_response[4:]
        clean_response = clean_response.strip()
        
        data = json.loads(clean_response)
        
        hypotheses = []
        for h in data.get("hypotheses", []):
            failure_mode_str = h.get("failure_mode", "").lower().replace(" ", "_")
            
            # Map to FailureMode enum
            try:
                failure_mode = FailureMode(failure_mode_str)
            except ValueError:
                continue  # Skip unknown failure modes
            
            confidence = min(0.85, max(0.0, float(h.get("confidence", 0.5))))
            
            severity = h.get("severity", "medium").lower()
            if severity not in ("low", "medium", "high"):
                severity = "medium"
            
            evidence = h.get("evidence", [])
            if isinstance(evidence, str):
                evidence = [evidence]
            
            hypotheses.append(Hypothesis(
                name=failure_mode,
                confidence=confidence,
                severity=severity,
                evidence=evidence
            ))
        
        return hypotheses
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: Failed to parse hypotheses response: {e}")
        return []


def _parse_recommendations_response(
    response: str,
    max_recommendations: int
) -> List[Recommendation]:
    """Parse the LLM response into Recommendation objects."""
    try:
        # Clean response (remove markdown code blocks if present)
        clean_response = response.strip()
        if clean_response.startswith("```"):
            clean_response = clean_response.split("```")[1]
            if clean_response.startswith("json"):
                clean_response = clean_response[4:]
        clean_response = clean_response.strip()
        
        data = json.loads(clean_response)
        
        recommendations = []
        for rec in data.get("recommendations", [])[:max_recommendations]:
            action = rec.get("action", "")
            rationale = rec.get("rationale", "")
            related_str = rec.get("related_failure_mode") or rec.get("related_hypothesis")
            
            # Map to FailureMode enum
            related_hypothesis = None
            if related_str:
                try:
                    related_hypothesis = FailureMode(related_str.lower().replace(" ", "_"))
                except ValueError:
                    pass
            
            if action:
                recommendations.append(Recommendation(
                    action=action,
                    rationale=rationale,
                    related_hypothesis=related_hypothesis
                ))
        
        return recommendations
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: Failed to parse recommendations response: {e}")
        return []


def _generate_fallback_summary(
    hypotheses: List[Hypothesis],
    recommendations: List[Recommendation]
) -> str:
    """Generate a basic summary when LLM fails."""
    lines = ["## Diagnosis\n"]
    
    if not hypotheses:
        lines.append("No significant issues detected in your model.\n")
    else:
        lines.append("Based on the analysis, here are the key findings:\n")
        for h in sorted(hypotheses, key=lambda x: x.confidence, reverse=True)[:3]:
            lines.append(f"- **{h.name.value.replace('_', ' ').title()}** ({h.confidence:.0%} confidence, {h.severity} severity)")
            if h.evidence:
                lines.append(f"  - {h.evidence[0]}")
        lines.append("")
    
    if recommendations:
        lines.append("## Recommendations\n")
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"**{i}. {rec.action}**")
            lines.append(f"   {rec.rationale}")
            if rec.related_hypothesis:
                lines.append(f"   *(Addresses: {rec.related_hypothesis.value})*")
            lines.append("")
    
    return "\n".join(lines)