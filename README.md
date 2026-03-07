# sklearn-diagnose 

An intelligent diagnostic and recommendation engine for `scikit-learn` models. This fork optimizes the original library by integrating the **Groq API** for near-instant inference and cleaning up redundant console noise for a better developer experience.

## Key Features
• Model Failure Diagnosis: Detect overfitting, underfitting, high variance, label noise, feature redundancy, class imbalance, and data leakage symptoms

• Interactive Chatbot: Launch a web-based chatbot to have conversations about your diagnosis results

• Cross-Validation Interpretation: CV interpretation is a core signal extractor within sklearn-diagnose, used to detect instability, overfitting, and potential data leakage

• Evidence-Based Hypotheses: All diagnoses include confidence scores and supporting evidence

• Actionable Recommendations: Get specific suggestions to fix identified issues

• Read-Only Behavior: Never modifies your estimator, parameters, or data

• Universal Compatibility: Works with any fitted scikit-learn estimator or Pipeline

## Why this Fork?

When I discovered sklearn-diagnose, a genuinely innovative approach to model analysis that the ML community desperately needs. Here's why this project caught my attention:
The Problem It Solves
Every data scientist has been there: you train a model, check the accuracy, maybe look at a confusion matrix, and then... what? You know something's wrong, but pinpointing exactly what and how to fix it requires hours of manual investigation. You're checking metrics, plotting distributions, running statistical tests, comparing train/test splits - all while trying to remember which patterns indicate which problems.
sklearn-diagnose changes this entirely. Instead of you being the detective, the LLM becomes your expert consultant, analyzing all the signals simultaneously and giving you human-readable insights with actionable recommendations.


## Quick Start

### 1. Installation
Clone this fork and install it in editable mode:
```bash
git clone [https://github.com/YOUR_USERNAME/sklearn-diagnose.git](https://github.com/YOUR_USERNAME/sklearn-diagnose.git)
cd sklearn-diagnose
pip install -e .
```
### 2. Configure Groq
Ensure your API key is set in your environment variables:
```bash
export GROQ_API_KEY='your_lp_api_key_here'
```

### Usage 
```bash
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn_diagnose import setup_llm, diagnose

# Set up LLM (REQUIRED - must specify provider, model, and api_key)
# Using OpenAI:
setup_llm(provider="openai", model="gpt-4o", api_key="your-openai-key")
# setup_llm(provider="openai", model="gpt-4o-mini", api_key="your-openai-key")

# Or using Anthropic:
# setup_llm(provider="anthropic", model="claude-3-5-sonnet-latest", api_key="your-anthropic-key")

# Or using OpenRouter (access to many models):
# setup_llm(provider="openrouter", model="deepseek/deepseek-r1-0528", api_key="your-openrouter-key")

# Your existing sklearn workflow
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# Diagnose your model
report = diagnose(
    estimator=model,
    datasets={
        "train": (X_train, y_train),
        "val": (X_val, y_val)
    },
    task="classification"
)

# View results
print(report.summary())          # LLM-generated summary
print(report.hypotheses)         # Detected issues with confidence
print(report.recommendations)    # LLM-ranked actionable suggestions
```

## Why This Package Deserves More Attention
It's Solving a Universal Problem
Every data scientist spends hours debugging models. If sklearn-diagnose saves even 30 minutes per model, across thousands of data scientists globally, that's millions of hours saved.
The Timing Is Perfect

LLMs are now cheap/fast enough for this use case
ML is becoming more democratized (more people need guidance)
MLOps is maturing (automated quality checks are expected)
The industry is realizing that "more models" doesn't mean "better models"

## Credits & Philosophy
This fork builds upon the excellent foundation laid by sklearn-diagnose by leockl.

I believe in:

• Standing on the shoulders of giants - Great tools deserve to be enhanced

• Open source collaboration - The best tools are built by communities

• Accessible ML - Everyone should have access to expert-level model diagnosis

• Practical innovation - Features should solve real problems, not just be technically impressive

##  License
MIT License - Same as original sklearn-diagnose
