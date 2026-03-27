"""
ReAct Agent - Reasoning and Acting with self-extending tools.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import ReActAgent
    from .baseline import prompt_llm
    from .client import LLMClient
    from .secrets import get_secret, has_secret, list_available_secrets


def __getattr__(name: str):
    """Lazily expose top-level package attributes without importing submodules eagerly."""
    if name == "ReActAgent":
        from .agent import ReActAgent
        return ReActAgent
    if name == "prompt_llm":
        from .baseline import prompt_llm
        return prompt_llm
    if name == "LLMClient":
        from .client import LLMClient
        return LLMClient
    if name == "get_secret":
        from .secrets import get_secret
        return get_secret
    if name == "list_available_secrets":
        from .secrets import list_available_secrets
        return list_available_secrets
    if name == "has_secret":
        from .secrets import has_secret
        return has_secret
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ReActAgent",
    "prompt_llm",
    "LLMClient",
    "get_secret",
    "list_available_secrets",
    "has_secret",
]
