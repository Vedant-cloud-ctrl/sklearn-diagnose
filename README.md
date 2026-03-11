## Why This Fork?

When I discovered **sklearn-diagnose**, I saw a genuinely innovative approach to model analysis that the ML community desperately needs. However, I noticed a few areas where the developer experience could be tightened for better real-world utility. Here is why this project caught my attention and what I've improved:

### The Problem: The "Insight Gap"
Every data scientist has been there: you train a model, check the accuracy, glance at a confusion matrix, and then—**what?** You know something is wrong, but pinpointing the "why" and "how" requires hours of manual investigation. You’re often stuck in a loop of checking metrics and running statistical tests, trying to decode which specific patterns indicate model failure.

### The Solution: From Detective to Consultant
**sklearn-diagnose** fundamentally changes this paradigm. Instead of forcing you to play detective, it positions an **LLM as an expert consultant**. It analyzes all diagnostic signals simultaneously to provide human-readable insights and actionable recommendations—turning a "blank stare" into a clear roadmap for model improvement.

### My Enhancements
To make the tool even more accessible and production-ready, I’ve implemented two key changes:
* **Groq API Integration:** Added support for Groq to leverage lightning-fast inference, making the diagnostic feedback near-instant.
* **Streamlined Output:** Removed redundant and distracting CV (Cross-Validation) error messages to ensure the final report stays focused on actionable insights rather than console noise.

---

## Installation

To get started with this fork, follow the steps below to set up your environment and configure the **Groq API** for high-speed diagnostics.

### 1. Install the Package
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/](https://github.com/)[your-username]/sklearn-diagnose.git
cd sklearn-diagnose
pip install -r requirements.txt


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
export GROQ_API_KEY='your_api_key_here'
```

### Usage 
```bash
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn_diagnose import setup_llm, diagnose

# Set up LLM (REQUIRED - must specify provider, model, and api_key)
# Using OpenAI:
# setup_llm(provider="openai", model="gpt-4o-mini", api_key="your-openai-key")

# Or using Anthropic:
# setup_llm(provider="anthropic", model="claude-3-5-sonnet-latest", api_key="your-anthropic-key")

# Or using OpenRouter (access to many models):
# setup_llm(provider="openrouter", model="deepseek/deepseek-r1-0528", api_key="your-openrouter-key")

# Or using Groq (access to many models):
# setup_llm(provider="groq", model="llama-3.3-70b-versatile", api_key="your-groq-key")

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

## Credits & Philosophy
This project is a fork of the original [sklearn-diagnose](https://github.com/leockl/sklearn-diagnose) developed by **[leockl](https://github.com/leockl)**.
This fork builds upon the excellent foundation laid by sklearn-diagnose by leockl.

##  License
MIT License - Same as original sklearn-diagnose
