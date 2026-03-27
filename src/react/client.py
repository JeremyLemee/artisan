"""
Multi-provider LLM client with conversation history support.
Supports Anthropic, OpenAI, Gemini, and Ollama APIs.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv("config/.env")


class LLMClient:
    """Multi-provider LLM client with conversation history."""

    def __init__(self, provider: str, config: dict):
        """
        Initialize LLM client.

        Args:
            provider: One of 'anthropic', 'openai', 'gemini', 'ollama'
            config: Full config dict containing provider settings
        """
        self.provider = provider
        self.config = config
        self.settings = config[provider]
        self.history: list[dict] = []
        self._client = None

    def _get_anthropic_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._client

    def _get_openai_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def _get_ollama_client(self):
        """Lazy initialization of Ollama client via the OpenAI-compatible API."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.settings.get("base_url", "http://localhost:11434/v1"),
                api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            )
        return self._client

    def _get_gemini_model(self):
        """Lazy initialization of Gemini model."""
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self._client = genai.GenerativeModel(self.settings["model"])
        return self._client

    def prompt(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        use_history: bool = True
    ) -> str:
        """
        Send a prompt to the LLM and return the response.

        Args:
            message: The user message to send
            system_prompt: Optional system prompt (used on first call or when history is empty)
            use_history: Whether to include conversation history

        Returns:
            The LLM's response text
        """
        if self.provider == "anthropic":
            response = self._prompt_anthropic(message, system_prompt, use_history)
        elif self.provider == "openai":
            response = self._prompt_openai(message, system_prompt, use_history)
        elif self.provider == "gemini":
            response = self._prompt_gemini(message, system_prompt, use_history)
        elif self.provider == "ollama":
            response = self._prompt_ollama(message, system_prompt, use_history)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        if use_history:
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": response})

        return response

    def _prompt_anthropic(
        self,
        message: str,
        system_prompt: Optional[str],
        use_history: bool
    ) -> str:
        """Send prompt to Anthropic API."""
        client = self._get_anthropic_client()

        messages = []
        if use_history:
            messages.extend(self.history)
        messages.append({"role": "user", "content": message})

        kwargs = {
            "model": self.settings["model"],
            "max_tokens": self.settings["max_tokens"],
            "temperature": self.settings["temperature"],
            "messages": messages
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)
        return response.content[0].text

    def _prompt_openai(
        self,
        message: str,
        system_prompt: Optional[str],
        use_history: bool
    ) -> str:
        """Send prompt to OpenAI API."""
        client = self._get_openai_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if use_history:
            messages.extend(self.history)
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model=self.settings["model"],
            max_tokens=self.settings["max_tokens"],
            temperature=self.settings["temperature"],
            messages=messages
        )
        return response.choices[0].message.content

    def _prompt_gemini(
        self,
        message: str,
        system_prompt: Optional[str],
        use_history: bool
    ) -> str:
        """Send prompt to Gemini API."""
        import google.generativeai as genai

        model = self._get_gemini_model()
        generation_config = genai.GenerationConfig(
            max_output_tokens=self.settings["max_tokens"],
            temperature=self.settings["temperature"]
        )

        # Build conversation content
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [system_prompt]})
            contents.append({"role": "model", "parts": ["Understood. I will follow these instructions."]})

        if use_history:
            for msg in self.history:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [msg["content"]]})

        contents.append({"role": "user", "parts": [message]})

        response = model.generate_content(contents, generation_config=generation_config)
        return response.text

    def _prompt_ollama(
        self,
        message: str,
        system_prompt: Optional[str],
        use_history: bool
    ) -> str:
        """Send prompt to a local Ollama instance using its OpenAI-compatible API."""
        client = self._get_ollama_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if use_history:
            messages.extend(self.history)
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model=self.settings["model"],
            max_tokens=self.settings["max_tokens"],
            temperature=self.settings["temperature"],
            messages=messages
        )
        return response.choices[0].message.content

    def clear_history(self):
        """Clear conversation history."""
        self.history = []

    @property
    def model(self) -> str:
        """Return the model name."""
        return self.settings["model"]
