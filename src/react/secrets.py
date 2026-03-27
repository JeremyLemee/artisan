"""
Secure secrets manager for generated tools.

Secrets are loaded from environment variables or a secrets file.
Tools can retrieve secrets by name at runtime, but the agent
never sees the actual secret values - only the secret names.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load tool secrets into environment (values are only accessible via get_secret allowlist)
load_dotenv("config/.env.tools")


class SecretsManager:
    """
    Secure secrets manager that provides runtime access to API keys and secrets.

    The agent can reference secret names (e.g., "OPENWEATHER_TOOL_API_KEY") but
    never receives the actual secret values. Tools retrieve secrets at
    execution time using get_secret().
    """

    _instance: Optional["SecretsManager"] = None

    def __new__(cls):
        """Singleton pattern to ensure one secrets manager instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Registry of available secret names (not values!)
        # This is what the agent can see
        self._available_secrets: set[str] = set()
        self._load_available_secrets()

    def _load_available_secrets(self):
        """Load the list of available secret names from config/.env.tools file."""
        env_file = Path("config/.env.tools")
        if env_file.exists():
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key = line.split("=", 1)[0].strip()
                        # Only expose keys that look like tool API keys
                        if "TOOL_API_KEY" in key.upper():
                            self._available_secrets.add(key)

    def get_secret(self, name: str) -> Optional[str]:
        """
        Retrieve a secret value by name at runtime.

        This method is called by tools during execution, NOT by the agent.
        Only secrets in the allowlist (_available_secrets) can be retrieved.
        This prevents access to non-tool secrets like LLM provider API keys.

        Args:
            name: The secret name (e.g., "OPENWEATHER_TOOL_API_KEY")

        Returns:
            The secret value, or None if not in allowlist or not found
        """
        # Security: Only allow access to explicitly configured tool secrets
        if name not in self._available_secrets:
            return None
        return os.getenv(name)

    def list_available_secrets(self) -> list[str]:
        """
        List available secret names (NOT values).

        This is safe to show to the agent - it only reveals what
        secrets exist, not their values.

        Returns:
            List of secret names that can be used with get_secret()
        """
        return sorted(self._available_secrets)

    def has_secret(self, name: str) -> bool:
        """Check if a secret is available (without revealing its value)."""
        # Security: Only check the allowlist, not os.getenv()
        return name in self._available_secrets


# Global singleton instance
_secrets_manager = SecretsManager()


def get_secret(name: str) -> Optional[str]:
    """
    Retrieve a secret value by name.

    This is the function that generated tools should use to access
    API keys and other secrets at runtime.

    Example usage in a generated tool:
        from react.secrets import get_secret

        api_key = get_secret("OPENWEATHER_TOOL_API_KEY")
        if api_key is None:
            return "Error: OPENWEATHER_TOOL_API_KEY not configured"

    Args:
        name: The secret name (e.g., "OPENWEATHER_TOOL_API_KEY")

    Returns:
        The secret value, or None if not found
    """
    return _secrets_manager.get_secret(name)


def list_available_secrets() -> list[str]:
    """List available secret names (safe to show to agent)."""
    return _secrets_manager.list_available_secrets()


def has_secret(name: str) -> bool:
    """Check if a secret exists (safe to show to agent)."""
    return _secrets_manager.has_secret(name)
