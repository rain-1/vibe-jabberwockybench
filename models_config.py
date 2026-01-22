"""
Model configurations for running Jabberwocky benchmark via OpenRouter.

This module defines all the model configurations for testing across
major AI models including Claude, GPT, Grok, and Gemini.
"""

# OpenRouter model identifiers (Updated for 2026)
# See: https://openrouter.ai/docs/models

MODELS = {
    # Claude models (Anthropic)
    "claude-4-opus": {
        "name": "anthropic/claude-4-opus",
        "provider": "openrouter",
        "description": "Claude 4 Opus - Most capable Claude model"
    },
    "claude-4-sonnet": {
        "name": "anthropic/claude-4-sonnet",
        "provider": "openrouter",
        "description": "Claude 4 Sonnet - Balanced performance"
    },

    # GPT models (OpenAI)
    "gpt-5": {
        "name": "openai/gpt-5",
        "provider": "openrouter",
        "description": "GPT-5 - Latest flagship OpenAI model"
    },
    "gpt-4o": {
        "name": "openai/gpt-4o",
        "provider": "openrouter",
        "description": "GPT-4o - Reliable workhorse"
    },

    # Grok models (xAI)
    "grok-4": {
        "name": "x-ai/grok-4",
        "provider": "openrouter",
        "description": "Grok 4 - xAI's latest flagship"
    },
    "grok-3": {
        "name": "x-ai/grok-3",
        "provider": "openrouter",
        "description": "Grok 3 - Previous flagship"
    },
    "grok-code-fast": {
        "name": "x-ai/grok-code-fast-1",
        "provider": "openrouter",
        "description": "Grok Code Fast - Specialized for code"
    },

    # Gemini models (Google)
    "gemini-2.0-flash": {
        "name": "google/gemini-2.0-flash-001",
        "provider": "openrouter",
        "description": "Gemini 2.0 Flash"
    },
    "gemini-2.0-pro": {
        "name": "google/gemini-2.0-pro-exp",
        "provider": "openrouter",
        "description": "Gemini 2.0 Pro Experimental"
    },
}


def get_model_id(model_key: str) -> str:
    """Get the OpenRouter model identifier for a given model key."""
    if model_key in MODELS:
        return f"openrouter/{MODELS[model_key]['name']}"
    else:
        # Fallback for dynamic model IDs
        if "/" in model_key:
             return f"openrouter/{model_key}"
        raise ValueError(f"Unknown model key: {model_key}")


def get_all_model_ids() -> list[str]:
    """Get all OpenRouter model identifiers."""
    return [f"openrouter/{config['name']}" for config in MODELS.values()]


def get_claude_models() -> list[str]:
    """Get only Claude model identifiers."""
    return [
        f"openrouter/{config['name']}"
        for key, config in MODELS.items()
        if 'claude' in key
    ]


def get_gpt_models() -> list[str]:
    """Get only GPT model identifiers."""
    return [
        f"openrouter/{config['name']}"
        for key, config in MODELS.items()
        if 'gpt' in key or 'openai' in key
    ]


def get_grok_models() -> list[str]:
    """Get only Grok model identifiers."""
    return [
        f"openrouter/{config['name']}"
        for key, config in MODELS.items()
        if 'grok' in key
    ]


def get_gemini_models() -> list[str]:
    """Get only Gemini model identifiers."""
    return [
        f"openrouter/{config['name']}"
        for key, config in MODELS.items()
        if 'gemini' in key
    ]
