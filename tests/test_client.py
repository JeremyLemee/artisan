import unittest
from unittest import mock

import httpx
from openai import APITimeoutError

from react.client import LLMClient


class LLMClientTests(unittest.TestCase):
    def test_resolve_ollama_base_url_uses_first_reachable_candidate(self):
        config = {
            "provider": "ollama",
            "ollama": {
                "model": "glm-4.7-flash:latest",
                "base_url": "http://host.docker.internal:11434/v1",
                "base_url_candidates": [
                    "http://host.docker.internal:11434/v1",
                    "http://localhost:11434/v1",
                ],
                "probe_timeout": 1,
                "timeout": 42,
                "max_tokens": 128,
                "temperature": 0.1,
            },
        }
        client = LLMClient("ollama", config)

        with mock.patch.object(
            client,
            "_ollama_healthcheck",
            side_effect=[False, True],
        ) as healthcheck:
            resolved = client._resolve_ollama_base_url()

        self.assertEqual(resolved, "http://localhost:11434/v1")
        self.assertEqual(client.settings["base_url"], "http://localhost:11434/v1")
        self.assertEqual(healthcheck.call_count, 2)

    def test_resolve_ollama_base_url_raises_when_no_candidate_is_reachable(self):
        config = {
            "provider": "ollama",
            "ollama": {
                "model": "glm-4.7-flash:latest",
                "base_url": "http://localhost:11434/v1",
                "base_url_candidates": [
                    "http://localhost:11434/v1",
                    "http://host.docker.internal:11434/v1",
                ],
                "probe_timeout": 1,
                "timeout": 42,
                "max_tokens": 128,
                "temperature": 0.1,
            },
        }
        client = LLMClient("ollama", config)

        with mock.patch.object(client, "_ollama_healthcheck", return_value=False):
            with self.assertRaisesRegex(
                RuntimeError,
                r"Could not reach Ollama from configured base URLs:",
            ):
                client._resolve_ollama_base_url()

    def test_ollama_timeout_is_reported_as_ollama_error(self):
        config = {
            "provider": "ollama",
            "ollama": {
                "model": "glm-4.7-flash:latest",
                "base_url": "http://localhost:11434/v1",
                "timeout": 42,
                "max_tokens": 128,
                "temperature": 0.1,
            },
        }
        client = LLMClient("ollama", config)

        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = APITimeoutError(
            request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
        )

        with mock.patch.object(client, "_get_ollama_client", return_value=fake_client):
            with self.assertRaisesRegex(
                RuntimeError,
                r"Ollama request timed out\. base_url=http://localhost:11434/v1, timeout=42s",
            ):
                client.prompt("hello")


if __name__ == "__main__":
    unittest.main()
