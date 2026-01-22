"""
Model configurations for running Jabberwocky benchmark via OpenRouter.

This module defines all the model configurations for testing across
major AI models including Claude, GPT, Grok, and Gemini.
"""

# OpenRouter model identifiers
# See: https://openrouter.ai/docs/models

MODELS = {
    # Claude models (Anthropic)
    "claude-opus-4.5": {
        "name": "anthropic/claude-opus-4.5",
        "provider": "openrouter",
        "description": "Claude Opus 4.5 - Most capable Claude model"
    },
    "claude-sonnet-4.5": {
        "name": "anthropic/claude-sonnet-4.5",
        "provider": "openrouter",
        "description": "Claude Sonnet 4.5 - Balanced performance"
    },
    "claude-haiku-4": {
        "name": "anthropic/claude-haiku-4",
        "provider": "openrouter",
        "description": "Claude Haiku 4 - Fast and efficient"
    },

    # GPT models (OpenAI)
    "gpt-4.5-turbo": {
        "name": "openai/gpt-4.5-turbo",
        "provider": "openrouter",
        "description": "GPT 4.5 Turbo - Latest GPT model"
    },
    "chatgpt-5.2": {
        "name": "openai/chatgpt-5.2",
        "provider": "openrouter",
        "description": "ChatGPT 5.2 - Conversational model"
    },
    "gpt-4-turbo": {
        "name": "openai/gpt-4-turbo",
        "provider": "openrouter",
        "description": "GPT-4 Turbo - Previous generation"
    },

    # Grok models (xAI)
    "grok-2": {
        "name": "x-ai/grok-2",
        "provider": "openrouter",
        "description": "Grok 2 - xAI's model"
    },
    "grok-2-vision": {
        "name": "x-ai/grok-2-vision",
        "provider": "openrouter",
        "description": "Grok 2 with vision capabilities"
    },

    # Gemini models (Google)
    "gemini-2.0-flash": {
        "name": "google/gemini-2.0-flash",
        "provider": "openrouter",
        "description": "Gemini 2.0 Flash - Fast and efficient"
    },
    "gemini-2.0-pro": {
        "name": "google/gemini-2.0-pro",
        "provider": "openrouter",
        "description": "Gemini 2.0 Pro - Most capable Gemini"
    },
    "gemini-pro": {
        "name": "google/gemini-pro",
        "provider": "openrouter",
        "description": "Gemini Pro - Previous generation"
    },
}


def get_model_id(model_key: str) -> str:
    """Get the OpenRouter model identifier for a given model key."""
    if model_key in MODELS:
        return f"openrouter/{MODELS[model_key]['name']}"
    else:
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
        if 'gpt' in key or 'chatgpt' in key
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
