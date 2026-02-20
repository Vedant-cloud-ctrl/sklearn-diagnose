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

HYPOTHESIS_SYSTEM_PROMPT = """You are an expert ML diagnostician agent. Your task is to analyze model performance signals and identify potential failure modes.

You will be given:
1. Performance metrics (train score, validation score, CV scores, etc.)
2. A list of possible failure modes to consider

For each failure mode you detect, you must provide:
- failure_mode: The name of the failure mode (must be one of the provided options)
- confidence: A score between 0.0 and 1.0 indicating how confident you are (0.95 max)
- severity: "low", "medium", or "high"
- evidence: A list of specific observations that support this hypothesis

Guidelines:
- Only report failure modes with confidence >= 0.25
- Be conservative - don't over-diagnose
- Base your assessment solely on the provided signals
- Provide specific, quantitative evidence when possible
- A model can have multiple failure modes simultaneously

Output format (STRICT JSON - no markdown, no code blocks):
{
  "hypotheses": [
    {
      "failure_mode": "overfitting",
      "confidence": 0.85,
      "severity": "high",
      "evidence": ["Train-val gap of 25% indicates memorization", "Perfect training score (100%) with 75% validation"]
    }
  ]
}

If no significant issues are detected, return:
{"hypotheses": []}"""


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

Your role is to provide a comprehensive diagnostic summary that includes:
1. A summary of the detected issues
2. The recommended actions to fix them

Guidelines:
- Be concise and direct
- Focus on the most important issues first
- Present recommendations in order of importance
- Use markdown formatting for clarity
- Include specific numbers and metrics from the evidence
- For feature_redundancy, include the specific correlated feature pairs
- For class_imbalance, include class distribution and recall disparities
- For data_leakage, include suspicious feature correlations and CV-holdout gaps

Structure your response as:
## Diagnosis
[Brief summary of detected issues with evidence]

## Recommendations
[Numbered list of recommendations with rationale]"""


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
            provider: One of "openai", "anthropic", "openrouter"
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
                # base_url=self.base_url or "https://api.openai.com/v1",
                **self.kwargs
            )
        
        elif self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.model_name,
                api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"),
                # base_url=self.base_url or "https://api.anthropic.com",
                **self.kwargs
            )
        
        elif self.provider == "openrouter":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key or os.environ.get("OPENROUTER_API_KEY"),
                # base_url=self.base_url or "https://openrouter.ai/api/v1",
                **self.kwargs
            )
        
        elif self.provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=self.model_name,
                api_key=self.api_key or os.environ.get("GROQ_API_KEY"),
                # base_url=self.base_url or "",
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


class GroqChatClient(LangChainClient):
    """
    Groq client using LangChain.
    
    Groq provides ultra-fast inference for open-source models like
    Llama, Mixtral, Gemma, and more through their LPU technology.
    
    Example:
        >>> client = GroqChatClient(model="llama-3.3-70b-versatile", api_key="gsk_...")
        >>> # Or using environment variable
        >>> os.environ["GROQ_API_KEY"] = "gsk_..."
        >>> client = GroqChatClient(model="llama-3.3-70b-versatile")
    """
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
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
            - "groq": Use Groq for ultra-fast inference (Llama, Mixtral, etc.)
        model: The model identifier. Examples:
            - OpenAI: "gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"
            - Anthropic: "claude-3-5-sonnet-latest", "claude-3-haiku-20240307"
            - OpenRouter: "deepseek/deepseek-r1-0528", "openai/gpt-4o"
            - Groq: "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"
        api_key: API key for the provider. If not provided, will look for:
            - OpenAI: OPENAI_API_KEY environment variable
            - Anthropic: ANTHROPIC_API_KEY environment variable
            - OpenRouter: OPENROUTER_API_KEY environment variable
            - Groq: GROQ_API_KEY environment variable
        base_url: Custom base URL (optional, mainly for OpenRouter or proxies)
        **kwargs: Additional arguments passed to the LangChain model
    
    Examples:
        # OpenAI
        >>> from skdiagnose import setup_llm
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
        
        # Groq (ultra-fast inference)
        >>> setup_llm(provider="groq", model="llama-3.3-70b-versatile", api_key="gsk_...")
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
        _global_client = GroqChatClient(model=model, api_key=api_key, **kwargs)
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
    """Build the prompt for LLM hypothesis generation."""
    
    # Format signals
    signal_lines = []
    
    # Key metrics
    if signals.get("train_score") is not None:
        signal_lines.append(f"- Training score: {signals['train_score']:.1%}")
    if signals.get("val_score") is not None:
        signal_lines.append(f"- Validation score: {signals['val_score']:.1%}")
    if signals.get("train_val_gap") is not None:
        signal_lines.append(f"- Train-validation gap: {signals['train_val_gap']:.1%}")
    
    # CV metrics
    if signals.get("cv_mean") is not None:
        signal_lines.append(f"- CV mean score: {signals['cv_mean']:.1%}")
    if signals.get("cv_std") is not None:
        signal_lines.append(f"- CV standard deviation: {signals['cv_std']:.1%}")
    if signals.get("cv_range") is not None:
        signal_lines.append(f"- CV range (max-min): {signals['cv_range']:.1%}")
    if signals.get("cv_train_val_gap") is not None:
        signal_lines.append(f"- CV train-test gap: {signals['cv_train_val_gap']:.1%}")
    
    # Data characteristics
    if signals.get("n_samples_train") is not None:
        signal_lines.append(f"- Training samples: {signals['n_samples_train']}")
    if signals.get("n_features") is not None:
        signal_lines.append(f"- Number of features: {signals['n_features']}")
    if signals.get("feature_to_sample_ratio") is not None:
        signal_lines.append(f"- Feature to sample ratio: {signals['feature_to_sample_ratio']:.3f}")
    
    # Classification-specific
    if task == "classification":
        if signals.get("minority_class_ratio") is not None:
            signal_lines.append(f"- Minority class ratio: {signals['minority_class_ratio']:.1%}")
        if signals.get("n_classes") is not None:
            signal_lines.append(f"- Number of classes: {signals['n_classes']}")
        
        # Class distribution details
        if signals.get("class_distribution"):
            signal_lines.append("- Class distribution:")
            for class_label, ratio in signals["class_distribution"].items():
                signal_lines.append(f"    - Class {class_label}: {ratio:.1%}")
        
        # Per-class recall (if available)
        if signals.get("per_class_recall"):
            signal_lines.append("- Per-class recall:")
            for class_label, recall in signals["per_class_recall"].items():
                signal_lines.append(f"    - Class {class_label}: {recall:.1%}")
        
        # Per-class precision (if available)
        if signals.get("per_class_precision"):
            signal_lines.append("- Per-class precision:")
            for class_label, precision in signals["per_class_precision"].items():
                signal_lines.append(f"    - Class {class_label}: {precision:.1%}")
    
    # Regression-specific
    if task == "regression":
        if signals.get("residual_skew") is not None:
            signal_lines.append(f"- Residual skewness: {signals['residual_skew']:.2f}")
        if signals.get("residual_kurtosis") is not None:
            signal_lines.append(f"- Residual kurtosis: {signals['residual_kurtosis']:.2f}")
    
    # Feature correlations - include detailed pair information
    if signals.get("high_correlation_pairs") and len(signals["high_correlation_pairs"]) > 0:
        pairs = signals["high_correlation_pairs"]
        n_pairs = len(pairs)
        max_corr = pairs[0][2] if pairs else 0
        signal_lines.append(f"- Highly correlated feature pairs: {n_pairs} (max correlation: {max_corr:.1%})")
        
        # Add detailed pair information (limit to top 10 to avoid prompt bloat)
        signal_lines.append("  Correlated pairs (feature_i, feature_j, correlation):")
        for i, (feat_i, feat_j, corr) in enumerate(pairs[:10]):
            signal_lines.append(f"    - Feature {feat_i} ↔ Feature {feat_j}: {corr:.1%}")
        if n_pairs > 10:
            signal_lines.append(f"    - ... and {n_pairs - 10} more pairs")
    
    # Data leakage signals
    if signals.get("cv_holdout_gap") is not None:
        cv_holdout_gap = signals["cv_holdout_gap"]
        signal_lines.append(f"- CV-to-holdout gap: {cv_holdout_gap:.1%}")
        if cv_holdout_gap > 0.10:
            signal_lines.append("  WARNING: CV performance significantly exceeds holdout - possible data leakage")
    
    # Suspicious feature-target correlations (potential target leakage)
    if signals.get("suspicious_feature_correlations") and len(signals["suspicious_feature_correlations"]) > 0:
        suspicious = signals["suspicious_feature_correlations"]
        n_suspicious = len(suspicious)
        max_corr = suspicious[0][1] if suspicious else 0
        signal_lines.append(f"- Suspicious feature-target correlations: {n_suspicious} features (max: {abs(max_corr):.1%})")
        signal_lines.append("  Features with unusually high correlation to target (potential leakage):")
        for feat_idx, corr in suspicious[:5]:
            signal_lines.append(f"    - Feature {feat_idx}: {abs(corr):.1%} correlation with target")
        if n_suspicious > 5:
            signal_lines.append(f"    - ... and {n_suspicious - 5} more suspicious features")
    
    # Define failure modes
    failure_modes = """
Available failure modes to consider:
1. overfitting - Model memorizes training data, performs well on train but poorly on validation
2. underfitting - Model is too simple, performs poorly on both train and validation
3. high_variance - Model is unstable, performance varies significantly across data splits
4. class_imbalance - Skewed class distribution affecting model performance (classification only)
5. feature_redundancy - Highly correlated or duplicate features
6. label_noise - Incorrect or noisy target labels
7. data_leakage - Information from validation leaking into training
8. distribution_shift - Training and validation data come from different distributions
9. multicollinearity - Features are highly linearly dependent, causing unstable coefficients
10. target_leakage - Feature directly encodes target information (perfect or near-perfect correlation)
11. calibration_error - Predicted probabilities don't match actual frequencies (classification only)
"""
    
    prompt = f"""Analyze these model diagnostic signals and identify potential failure modes.

Task type: {task}

Observed Signals:
{chr(10).join(signal_lines) if signal_lines else "- Limited signals available"}

{failure_modes}

Based on these signals, identify which failure modes are present. For each detected issue, provide:
- The failure mode name (from the list above)
- Confidence score (0.0 to 0.95)
- Severity (low, medium, high)
- Specific evidence from the signals

IMPORTANT:
- For feature_redundancy, include the specific correlated feature pairs and their correlation values in the evidence.
- For class_imbalance, include the class distribution and any per-class recall/precision disparities in the evidence.
- For data_leakage, include the CV-to-holdout gap and any suspicious feature-target correlations in the evidence.
- For distribution_shift, include train/val performance difference and any detected distribution divergence.
- For multicollinearity, include VIF values or highly correlated feature pairs.
- For target_leakage, include features with suspiciously high target correlation (>0.9).
- For calibration_error, include calibration curve deviations or Brier score if available.

Return your analysis as JSON."""
    
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
    
    # Format signals
    signal_lines = []
    if signals.get("train_score") is not None:
        signal_lines.append(f"- Training score: {signals['train_score']:.1%}")
    if signals.get("val_score") is not None:
        signal_lines.append(f"- Validation score: {signals['val_score']:.1%}")
    if signals.get("train_val_gap") is not None:
        signal_lines.append(f"- Train-val gap: {signals['train_val_gap']:.1%}")
    if signals.get("cv_mean") is not None:
        signal_lines.append(f"- CV mean: {signals['cv_mean']:.1%}")
    if signals.get("cv_std") is not None:
        signal_lines.append(f"- CV std: {signals['cv_std']:.1%}")
    
    # Format hypotheses
    hypothesis_lines = []
    for h in sorted(hypotheses, key=lambda x: x.confidence, reverse=True):
        hypothesis_lines.append(f"- {h.name.value} (confidence: {h.confidence:.0%}, severity: {h.severity})")
        for ev in h.evidence:
            hypothesis_lines.append(f"  - Evidence: {ev}")
    
    # Format recommendations
    rec_lines = []
    for i, rec in enumerate(recommendations, 1):
        rec_lines.append(f"{i}. {rec.action}")
        rec_lines.append(f"   - Rationale: {rec.rationale}")
        if rec.related_hypothesis:
            rec_lines.append(f"   - Addresses: {rec.related_hypothesis.value}")
    
    prompt = f"""Provide a comprehensive diagnostic summary for this model.

Task type: {task}

Observed Signals:
{chr(10).join(signal_lines) if signal_lines else "- Limited signals available"}

Detected Issues (Hypotheses):
{chr(10).join(hypothesis_lines) if hypothesis_lines else "- No significant issues detected"}

Recommendations to Address Issues:
{chr(10).join(rec_lines) if rec_lines else "- No recommendations"}

Please provide a clear diagnostic report with this EXACT structure:

## Detected Issues
[List each detected issue with its confidence score and key evidence. If no issues, state the model appears healthy.]

## Diagnosis
[Brief explanation of what is happening with this model and why - under 100 words]

## Recommendations
[Present the recommendations above in a numbered list with rationale]

IMPORTANT: Always include the "Detected Issues" section showing what was found."""
    
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
            
            confidence = min(0.95, max(0.0, float(h.get("confidence", 0.5))))
            
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
