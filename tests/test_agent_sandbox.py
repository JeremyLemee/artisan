import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

from react import agent


class AgentSandboxTests(unittest.TestCase):
    def test_load_config_resolves_default_path_from_repo_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            config_dir = repo_root / "config"
            config_dir.mkdir()
            (config_dir / "config.yaml").write_text("provider: openai\nopenai:\n  model: gpt-4o\n")

            with mock.patch.object(agent, "_repo_root", return_value=repo_root):
                with mock.patch("pathlib.Path.cwd", return_value=repo_root / "elsewhere"):
                    config = agent.load_config("config/config.yaml")

            self.assertEqual(config["provider"], "openai")

    def test_build_docker_command_mounts_repo_read_only_and_generated_tools_writable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            generated_dir = repo_root / "src/react/tools/generated"
            generated_dir.mkdir(parents=True)

            with mock.patch.object(agent, "_repo_root", return_value=repo_root):
                with mock.patch.object(agent, "_docker_uses_host_network", return_value=True):
                    args = agent._build_parser().parse_args(["--sandbox", "-t", "hello"])
                    config = {
                        "provider": "openai",
                        "openai": {"model": "gpt-4o"},
                        "react": {"generated_tools_dir": "src/react/tools/generated"},
                    }

                    command = agent._build_docker_command(args, config, detach=True)

            self.assertIn(f"{repo_root}:/workspace:ro", command)
            self.assertIn(
                f"{generated_dir.resolve()}:/workspace/src/react/tools/generated:rw",
                command,
            )
            self.assertIn("--detach", command)
            self.assertIn("REACT_AGENT_DONE_MARKER=1", command)
            self.assertIn("--network", command)
            self.assertIn("host", command)
            self.assertIn("--sandbox-internal", command)

    def test_build_docker_command_remaps_external_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir) / "repo"
            repo_root.mkdir()
            generated_dir = repo_root / "src/react/tools/generated"
            generated_dir.mkdir(parents=True)

            external_config = Path(tmp_dir) / "custom.yaml"
            external_config.write_text("provider: openai\nopenai:\n  model: gpt-4o\n")

            with mock.patch.object(agent, "_repo_root", return_value=repo_root):
                args = agent._build_parser().parse_args(
                    ["--sandbox", "--config", str(external_config), "-t", "hello"]
                )
                config = {
                    "provider": "openai",
                    "openai": {"model": "gpt-4o"},
                    "react": {"generated_tools_dir": "src/react/tools/generated"},
                }

                command = agent._build_docker_command(args, config)

            self.assertIn(
                f"{external_config.resolve()}:/tmp/react-agent-config/custom.yaml:ro",
                command,
            )
            config_index = command.index("--config")
            self.assertEqual(command[config_index + 1], "/tmp/react-agent-config/custom.yaml")

    def test_sandboxed_logs_always_use_tmp_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "provider": "openai",
                "openai": {"model": "gpt-4o"},
                "logging": {"output_dir": str(Path(tmp_dir) / "logs")},
                "react": {"generated_tools_dir": str(Path(tmp_dir) / "generated")},
            }

            sandboxed_agent = agent.ReActAgent(config, sandboxed=True)
            sandboxed_agent.trace = []
            sandboxed_agent.tools_created = []

            log_file = sandboxed_agent._log_result("test task", "done", 1)

            self.assertTrue(Path(log_file).is_file())
            self.assertTrue(str(log_file).startswith("/tmp/react-agent-logs/openai/"))

    def test_prepare_sandbox_config_keeps_ollama_localhost_on_linux(self):
        config = {
            "provider": "ollama",
            "ollama": {"model": "glm-4.7-flash:latest", "base_url": "http://localhost:11434/v1"},
        }

        with mock.patch.object(agent, "_docker_uses_host_network", return_value=True):
            updated = agent._prepare_sandbox_config(config)

        self.assertEqual(updated["ollama"]["base_url"], "http://localhost:11434/v1")
        self.assertEqual(
            updated["ollama"]["base_url_candidates"],
            ["http://localhost:11434/v1", "http://host.docker.internal:11434/v1"],
        )

    def test_prepare_sandbox_config_rewrites_ollama_localhost_without_host_network(self):
        config = {
            "provider": "ollama",
            "ollama": {"model": "glm-4.7-flash:latest", "base_url": "http://localhost:11434/v1"},
        }

        with mock.patch.object(agent, "_docker_uses_host_network", return_value=False):
            updated = agent._prepare_sandbox_config(config)

        self.assertEqual(updated["ollama"]["base_url"], "http://host.docker.internal:11434/v1")
        self.assertEqual(
            updated["ollama"]["base_url_candidates"],
            ["http://host.docker.internal:11434/v1", "http://localhost:11434/v1"],
        )

    def test_log_result_falls_back_to_tmp_when_local_write_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "provider": "openai",
                "openai": {"model": "gpt-4o"},
                "logging": {"output_dir": str(Path(tmp_dir) / "logs")},
                "react": {"generated_tools_dir": str(Path(tmp_dir) / "generated")},
            }

            local_agent = agent.ReActAgent(config, sandboxed=False)
            local_agent.trace = []
            local_agent.tools_created = []

            real_open = open
            failure_target = agent._resolve_log_output_dir(config, False)

            def selective_open(path, *args, **kwargs):
                if Path(path).parent == failure_target:
                    raise OSError("read-only")
                return real_open(path, *args, **kwargs)

            with mock.patch("builtins.open", side_effect=selective_open):
                log_file = local_agent._log_result("test task", "done", 1)

            self.assertTrue(Path(log_file).is_file())
            self.assertTrue(str(log_file).startswith("/tmp/react-agent-logs/openai/"))

    def test_print_result_emits_completion_marker_when_requested(self):
        result = {
            "final_answer": "done",
            "iterations": 1,
            "tools_created": [],
            "log_file": "/tmp/react-agent-logs/openai/react_test.json",
            "trace": [],
        }
        output = StringIO()

        with mock.patch.dict("os.environ", {"REACT_AGENT_DONE_MARKER": "1"}, clear=False):
            with mock.patch("sys.stdout", output):
                agent._print_result(result, verbose=False)

        self.assertIn(agent.DOCKER_DONE_MARKER, output.getvalue())


if __name__ == "__main__":
    unittest.main()
