"""
ReAct (Reasoning and Acting) agent with self-extending tools.
Implements Thought/Action/Observation loops with dynamic tool creation.
"""

import argparse
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .client import LLMClient
from .tools import Tool, ToolRegistry, ControlTool, CalculatorTool, FinishSignal
from .secrets import list_available_secrets


REACT_SYSTEM_PROMPT = """You are a ReAct agent. You MUST ALWAYS respond using ONLY this exact format - no other text:

Thought: <reasoning>
Action: <tool.operation>
Action Input: <JSON>

{operations}

{secrets_info}

RULES:
1. NEVER respond with plain text - ALWAYS use the Thought/Action/Action Input format
2. After receiving an Observation with your answer, you MUST call control.finish
3. You can ONLY interact through tool calls - there is no other way to communicate
4. When creating tools that need API keys, use get_secret() - NEVER ask for or include actual secret values

EXAMPLE 1 - Math calculation:
Task: What is 15 * 7?

Your response:
Thought: I need to multiply 15 by 7.
Action: calculator.evaluate
Action Input: {{"expression": "15 * 7"}}

Observation: 105

Your response:
Thought: The result is 105. I have the answer and must finish.
Action: control.finish
Action Input: {{"answer": "15 multiplied by 7 equals 105"}}

EXAMPLE 2 - Creating a new tool:
Task: Create a tool to reverse strings

Your response:
Thought: I need to create a string tool. I must use the Tool base class with the @operation decorator.
Action: control.create_tool
Action Input: {{"name": "StringTool", "description": "String manipulation operations", "code": "from react.tools.base import Tool, operation\\n\\nclass StringTool(Tool):\\n    name = 'string'\\n    description = 'String manipulation operations'\\n\\n    @operation\\n    def reverse(self, text: str) -> str:\\n        '''Reverse a string'''\\n        return text[::-1]"}}

Observation: Tool 'string' created with operations: string.reverse

Your response:
Thought: The tool was created. Now I can use string.reverse.
Action: string.reverse
Action Input: {{"text": "hello"}}

EXAMPLE 3 - Creating a tool that uses an API key:
Task: Create a weather tool

Your response:
Thought: I need to create a weather tool that uses the OpenWeather API. I'll use get_secret() to securely access the API key at runtime.
Action: control.create_tool
Action Input: {{"name": "WeatherTool", "description": "Weather information tool", "code": "from react.tools.base import Tool, operation\\nfrom react.secrets import get_secret\\nimport urllib.request\\nimport json\\n\\nclass WeatherTool(Tool):\\n    name = 'weather'\\n    description = 'Get weather information'\\n\\n    @operation\\n    def get_current(self, city: str) -> str:\\n        '''Get current weather for a city'''\\n        api_key = get_secret('OPENWEATHER_TOOL_API_KEY')\\n        if not api_key:\\n            return 'Error: OPENWEATHER_TOOL_API_KEY not configured'\\n        url = f'https://api.openweathermap.org/data/2.5/weather?q={{city}}&appid={{api_key}}&units=metric'\\n        try:\\n            with urllib.request.urlopen(url) as response:\\n                data = json.loads(response.read())\\n                return f\\\"{{data['name']}}: {{data['main']['temp']}}°C, {{data['weather'][0]['description']}}\\\"\\n        except Exception as e:\\n            return f'Error: {{e}}'"}}

CRITICAL: You MUST use control.finish to provide your final answer. Plain text responses are NOT allowed.
SECURITY: NEVER include actual API keys or secrets in your code. ALWAYS use get_secret('SECRET_NAME').
"""

DOCKER_DONE_MARKER = "__REACT_AGENT_DONE__"


class ReActAgent:
    """ReAct agent with Thought/Action/Observation loop."""

    def __init__(self, config: dict, sandboxed: bool = False):
        self.config = config
        self.react_config = config.get("react", {})
        self.max_iterations = self.react_config.get("max_iterations", 10)
        self.enable_tool_creation = self.react_config.get("enable_tool_creation", True)
        self.generated_tools_dir = self.react_config.get("generated_tools_dir", "src/react/tools/generated")
        self.sandboxed = sandboxed

        # Initialize tool registry
        self.registry = ToolRegistry()
        self._register_builtin_tools()

        # Initialize LLM client
        self.client = LLMClient(config["provider"], config)

        # Trace for logging
        self.trace: list[dict] = []
        self.tools_created: list[str] = []

    def _register_builtin_tools(self):
        """Register built-in tools."""
        control_tool = ControlTool(self.registry, self.generated_tools_dir)
        self.registry.register_tool(control_tool)
        self.registry.register_tool(CalculatorTool())

        # Load any previously generated tools
        self._load_generated_tools()

    def _load_generated_tools(self):
        """Load all previously generated tools from the generated tools directory."""
        generated_dir = Path(self.generated_tools_dir)
        if not generated_dir.exists():
            return

        for tool_file in generated_dir.glob("*.py"):
            if tool_file.name.startswith("__"):
                continue

            try:
                # Load the module
                module_name = tool_file.stem
                spec = importlib.util.spec_from_file_location(module_name, tool_file)
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Find and register Tool subclasses
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                        issubclass(attr, Tool) and
                        attr is not Tool):
                        tool_instance = attr()
                        self.registry.register_tool(tool_instance)

            except Exception as e:
                # Skip tools that fail to load
                print(f"Warning: Failed to load generated tool {tool_file.name}: {e}")

    def _build_system_prompt(self) -> str:
        """Build the system prompt with available operations and secrets."""
        operations = self.registry.format_operations_for_prompt()

        # Build secrets info (names only, never values)
        available_secrets = list_available_secrets()
        if available_secrets:
            secrets_info = "Available secrets (use get_secret('NAME') in tools):\n"
            for secret_name in available_secrets:
                secrets_info += f"  - {secret_name}\n"
        else:
            secrets_info = "No API secrets currently configured."

        return REACT_SYSTEM_PROMPT.format(
            operations=operations,
            secrets_info=secrets_info
        )

    def _parse_response(self, response: str) -> dict:
        """Parse LLM response to extract Thought, Action, and Action Input."""
        result = {
            "thought": None,
            "action": None,
            "action_input": None,
            "raw": response
        }

        # Extract Thought (may not be present in all responses)
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # Extract Action
        action_match = re.search(r"Action:\s*([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)", response, re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1).strip()

        # Extract Action Input - look for JSON object
        input_match = re.search(r"Action Input:\s*(\{.*\})", response, re.DOTALL | re.IGNORECASE)
        if input_match:
            input_str = input_match.group(1).strip()
            # Find the balanced JSON object
            try:
                # Try to parse incrementally to find valid JSON
                depth = 0
                json_end = 0
                for i, char in enumerate(input_str):
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            json_end = i + 1
                            break
                if json_end > 0:
                    result["action_input"] = json.loads(input_str[:json_end])
                else:
                    result["action_input"] = json.loads(input_str)
            except json.JSONDecodeError:
                # Try to find any JSON object in the response
                json_match = re.search(r'\{[^{}]*\}', response)
                if json_match:
                    try:
                        result["action_input"] = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        result["action_input"] = {}
                else:
                    result["action_input"] = {}

        return result

    def _execute_action(self, action: str, action_input: dict) -> str:
        """Execute an action and return the observation."""
        try:
            result = self.registry.execute(action, **action_input)
            return str(result)
        except FinishSignal as e:
            raise e
        except Exception as e:
            return f"Error: {e}"

    def run(self, task: str) -> dict:
        """
        Run the ReAct loop on a task.

        Returns:
            dict with task, trace, final_answer, iterations, tools_created
        """
        self.trace = []
        self.tools_created = []
        system_prompt = self._build_system_prompt()

        # Initial prompt
        current_message = f"Task: {task}"
        final_answer = None
        iterations = 0

        for iteration in range(self.max_iterations):
            iterations = iteration + 1

            # Get LLM response
            if iteration == 0:
                response = self.client.prompt(current_message, system_prompt=system_prompt)
            else:
                response = self.client.prompt(current_message)

            # Parse response
            parsed = self._parse_response(response)

            # Record thought
            if parsed["thought"]:
                self.trace.append({"type": "thought", "content": parsed["thought"]})

            # Check for valid action
            if not parsed["action"]:
                # No action found - the LLM may have responded with plain text
                # Check if the response looks like a final answer
                if "answer" in response.lower() or "result" in response.lower() or any(c.isdigit() for c in response):
                    # Try to extract answer and finish
                    self.trace.append({"type": "thought", "content": f"(Agent provided answer directly: {response[:200]})"})
                    final_answer = response.strip()
                    break
                # Ask for proper format
                current_message = "You MUST respond using the format: Thought: <reasoning>\\nAction: <tool.operation>\\nAction Input: <JSON>\\n\\nUse control.finish to provide your answer."
                continue

            # Record action
            self.trace.append({
                "type": "action",
                "operation": parsed["action"],
                "input": parsed["action_input"]
            })

            # Execute action
            try:
                observation = self._execute_action(
                    parsed["action"],
                    parsed["action_input"] or {}
                )

                # Check if a new tool was created
                if parsed["action"] == "control.create_tool":
                    if "created with operations" in observation:
                        tool_name = parsed["action_input"].get("name", "unknown")
                        self.tools_created.append(tool_name)
                        # Update system prompt with new operations
                        system_prompt = self._build_system_prompt()

                # Record observation
                self.trace.append({"type": "observation", "content": observation})

                # Prepare next message
                current_message = f"Observation: {observation}"

            except FinishSignal as e:
                final_answer = e.answer
                break

        # Log the result
        log_file = self._log_result(task, final_answer, iterations)

        return {
            "task": task,
            "trace": self.trace,
            "final_answer": final_answer,
            "iterations": iterations,
            "tools_created": self.tools_created,
            "log_file": log_file
        }

    def _log_result(self, task: str, final_answer: Optional[str], iterations: int) -> str:
        """Log the ReAct trace to a file."""
        output_dir = _resolve_log_output_dir(self.config, self.sandboxed)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

        log_entry = {
            "timestamp": timestamp.isoformat(),
            "provider": self.config["provider"],
            "model": self.client.model,
            "task": task,
            "trace": self.trace,
            "final_answer": final_answer,
            "iterations": iterations,
            "tools_created": self.tools_created
        }

        filename = f"react_{timestamp_str}.json"
        filepath = output_dir / filename

        try:
            with open(filepath, "w") as f:
                json.dump(log_entry, f, indent=2)
        except OSError:
            fallback_dir = _resolve_log_output_dir(self.config, True)
            fallback_dir.mkdir(parents=True, exist_ok=True)
            filepath = fallback_dir / filename
            with open(filepath, "w") as f:
                json.dump(log_entry, f, indent=2)

        return str(filepath)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    resolved_path = _resolve_config_path(config_path)
    with open(resolved_path, "r") as f:
        return yaml.safe_load(f)


def _repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="ReAct agent with self-extending tools")
    parser.add_argument("-c", "--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("-t", "--task", required=True, help="Task for the agent to perform")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Run the agent inside Docker with the repository mounted read-only except for generated tools",
    )
    parser.add_argument(
        "--sandbox-internal",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _resolve_image_name(config: dict) -> str:
    """Return the Docker image name for sandboxed runs."""
    return config.get("react", {}).get("docker_image", "react-agent-sandbox:latest")


def _resolve_config_path(config_path: str) -> Path:
    """Resolve a config path against the current directory and repository root."""
    candidate = Path(config_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = (_repo_root() / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    return cwd_candidate


def _resolve_log_output_dir(config: dict, sandboxed: bool) -> Path:
    """Return the output directory for logs."""
    provider_dir = Path(config.get("provider", "unknown"))
    if sandboxed:
        return Path("/tmp/react-agent-logs") / provider_dir
    return Path(config.get("logging", {}).get("output_dir", "data/responses")) / provider_dir


def _docker_uses_host_network() -> bool:
    """Return whether Docker should use host networking for sandboxed runs."""
    return platform.system().lower() == "linux"


def _resolve_config_for_docker(config_path: str) -> tuple[str, list[str]]:
    """Resolve a config path and any extra bind mounts needed for Docker."""
    repo_root = _repo_root().resolve()
    host_config_path = _resolve_config_path(config_path)

    try:
        relative_path = host_config_path.relative_to(repo_root)
    except ValueError:
        container_path = f"/tmp/react-agent-config/{host_config_path.name}"
        mounts = ["-v", f"{host_config_path}:{container_path}:ro"]
        return container_path, mounts

    return f"/workspace/{relative_path.as_posix()}", []


def _build_docker_command(args: argparse.Namespace, config: dict, detach: bool = False) -> list[str]:
    """Build the docker run command for sandboxed execution."""
    repo_root = _repo_root()
    generated_dir = repo_root / config.get("react", {}).get("generated_tools_dir", "src/react/tools/generated")
    generated_dir.mkdir(parents=True, exist_ok=True)
    docker_config_path, config_mounts = _resolve_config_for_docker(args.config)

    image_name = _resolve_image_name(config)
    command = [
        "docker",
        "run",
        "--rm",
        "--workdir",
        "/workspace",
        "-e",
        "PYTHONPATH=/workspace/src",
        "-v",
        f"{repo_root}:/workspace:ro",
        "-v",
        f"{generated_dir.resolve()}:/workspace/{generated_dir.relative_to(repo_root).as_posix()}:rw",
    ]

    if detach:
        command.append("--detach")

    if _docker_uses_host_network():
        command.extend(["--network", "host"])
    else:
        command.extend(["--add-host", "host.docker.internal:host-gateway"])

    command.extend(config_mounts)

    for env_file in ("config/.env", "config/.env.tools"):
        env_path = repo_root / env_file
        if env_path.exists():
            command.extend(["--env-file", str(env_path)])

    command.extend(["-e", "REACT_AGENT_DONE_MARKER=1"])

    command.extend([
        image_name,
        "--sandbox-internal",
        "--config",
        docker_config_path,
        "--task",
        args.task,
    ])

    if args.verbose:
        command.append("--verbose")

    return command


def _ensure_sandbox_image(image_name: str) -> None:
    """Build the sandbox image if it does not already exist."""
    inspect_result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect_result.returncode == 0:
        return

    dockerfile = _repo_root() / "Dockerfile"
    build_result = subprocess.run(
        [
            "docker",
            "build",
            "-f",
            str(dockerfile),
            "-t",
            image_name,
            str(_repo_root()),
        ],
        check=False,
    )
    if build_result.returncode != 0:
        raise RuntimeError(f"Failed to build Docker sandbox image '{image_name}'")


def _stop_docker_container(container_id: str) -> None:
    """Stop a Docker container if it is still running."""
    subprocess.run(
        ["docker", "stop", "--time", "1", container_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _remove_docker_container(container_id: str) -> None:
    """Remove a Docker container if it still exists."""
    subprocess.run(
        ["docker", "rm", "-f", container_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _docker_container_running(container_id: str) -> bool:
    """Return whether a Docker container is still running."""
    inspect_result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
        capture_output=True,
        text=True,
        check=False,
    )
    return inspect_result.returncode == 0 and inspect_result.stdout.strip() == "true"


def _stream_docker_logs(container_id: str) -> bool:
    """Stream Docker logs to stdout and return whether the agent completion marker was seen."""
    completed = False
    logs_process = subprocess.Popen(
        ["docker", "logs", "-f", container_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert logs_process.stdout is not None
        for line in logs_process.stdout:
            if DOCKER_DONE_MARKER in line:
                completed = True
                break
            print(line, end="")
    finally:
        if logs_process.stdout is not None:
            logs_process.stdout.close()
        logs_process.terminate()
        try:
            logs_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            logs_process.kill()
            logs_process.wait()

    return completed


def _run_in_docker(args: argparse.Namespace, config: dict) -> int:
    """Execute the agent inside Docker."""
    image_name = _resolve_image_name(config)
    _ensure_sandbox_image(image_name)
    start_result = subprocess.run(
        _build_docker_command(args, config, detach=True),
        capture_output=True,
        text=True,
        check=False,
    )
    if start_result.returncode != 0:
        if start_result.stdout:
            print(start_result.stdout, end="")
        if start_result.stderr:
            print(start_result.stderr, end="", file=sys.stderr)
        return start_result.returncode

    container_id = start_result.stdout.strip().splitlines()[-1]

    try:
        completed = _stream_docker_logs(container_id)
        if completed and _docker_container_running(container_id):
            _stop_docker_container(container_id)
        wait_result = subprocess.run(
            ["docker", "wait", container_id],
            capture_output=True,
            text=True,
            check=False,
        )
        if wait_result.stdout.strip():
            return int(wait_result.stdout.strip())
        return wait_result.returncode
    finally:
        _stop_docker_container(container_id)
        _remove_docker_container(container_id)


def _run_agent(args: argparse.Namespace, sandboxed: bool = False) -> dict:
    """Run the agent locally and return the result payload."""
    config = load_config(args.config)
    if sandboxed:
        config = _prepare_sandbox_config(config)
    print(f"Using provider: {config['provider']} ({config[config['provider']]['model']})")
    print("Mode: ReAct")
    if sandboxed:
        print("Execution: Docker sandbox")

    agent = ReActAgent(config, sandboxed=sandboxed)
    return agent.run(args.task)


def _print_result(result: dict, verbose: bool) -> None:
    """Print the agent result to stdout."""
    if verbose:
        print("\n--- ReAct Trace ---")
        for entry in result["trace"]:
            if entry["type"] == "thought":
                print(f"\nThought: {entry['content']}")
            elif entry["type"] == "action":
                print(f"Action: {entry['operation']}")
                print(f"Action Input: {json.dumps(entry['input'])}")
            elif entry["type"] == "observation":
                print(f"Observation: {entry['content']}")
        print("--- End Trace ---\n")

    print(f"\nFinal Answer: {result['final_answer']}")
    print(f"Iterations: {result['iterations']}")
    if result["tools_created"]:
        print(f"Tools Created: {', '.join(result['tools_created'])}")
    print(f"Logged to: {result['log_file']}")
    if os.getenv("REACT_AGENT_DONE_MARKER") == "1":
        print(DOCKER_DONE_MARKER)


def _prepare_sandbox_config(config: dict) -> dict:
    """Adjust provider settings for Docker sandbox execution."""
    if config.get("provider") != "ollama":
        return config

    ollama_config = dict(config.get("ollama", {}))
    base_url = ollama_config.get("base_url")
    if not base_url:
        return config

    parsed = urlparse(base_url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "host.docker.internal"}:
        return config

    candidates = [base_url]
    host = "host.docker.internal"
    if parsed.port is not None:
        netloc = f"{host}:{parsed.port}"
    else:
        netloc = host
    host_gateway_url = parsed._replace(netloc=netloc).geturl()

    if _docker_uses_host_network():
        candidates.append(host_gateway_url)
    else:
        candidates = [host_gateway_url, base_url]

    ollama_config["base_url"] = candidates[0]
    ollama_config["base_url_candidates"] = candidates
    ollama_config.setdefault("probe_timeout", 3)
    updated_config = dict(config)
    updated_config["ollama"] = ollama_config
    return updated_config


def main():
    """Main entry point for the ReAct agent."""
    parser = _build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.sandbox and not args.sandbox_internal:
        raise SystemExit(_run_in_docker(args, config))

    try:
        result = _run_agent(args, sandboxed=args.sandbox_internal)
    except Exception as exc:
        if args.sandbox_internal:
            raise
        print(
            f"Local execution failed ({exc}). Retrying inside Docker sandbox...",
            file=sys.stderr,
        )
        raise SystemExit(_run_in_docker(args, config))

    _print_result(result, args.verbose)


if __name__ == "__main__":
    main()
