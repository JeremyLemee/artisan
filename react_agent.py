#!/usr/bin/env python3
"""
ReAct (Reasoning and Acting) agent with self-extending tools.
Implements Thought/Action/Observation loops with dynamic tool creation.
"""

import argparse
import importlib.util
import json
import re
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional

from llm_client import LLMClient
from tools import Tool, ToolRegistry, ControlTool, CalculatorTool, FinishSignal
from secrets_manager import list_available_secrets


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
Action Input: {{"name": "StringTool", "description": "String manipulation operations", "code": "from tools.base import Tool, operation\\n\\nclass StringTool(Tool):\\n    name = 'string'\\n    description = 'String manipulation operations'\\n\\n    @operation\\n    def reverse(self, text: str) -> str:\\n        '''Reverse a string'''\\n        return text[::-1]"}}

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
Action Input: {{"name": "WeatherTool", "description": "Weather information tool", "code": "from tools.base import Tool, operation\\nfrom secrets_manager import get_secret\\nimport urllib.request\\nimport json\\n\\nclass WeatherTool(Tool):\\n    name = 'weather'\\n    description = 'Get weather information'\\n\\n    @operation\\n    def get_current(self, city: str) -> str:\\n        '''Get current weather for a city'''\\n        api_key = get_secret('OPENWEATHER_TOOL_API_KEY')\\n        if not api_key:\\n            return 'Error: OPENWEATHER_TOOL_API_KEY not configured'\\n        url = f'https://api.openweathermap.org/data/2.5/weather?q={{city}}&appid={{api_key}}&units=metric'\\n        try:\\n            with urllib.request.urlopen(url) as response:\\n                data = json.loads(response.read())\\n                return f\\\"{{data['name']}}: {{data['main']['temp']}}°C, {{data['weather'][0]['description']}}\\\"\\n        except Exception as e:\\n            return f'Error: {{e}}'"}}

CRITICAL: You MUST use control.finish to provide your final answer. Plain text responses are NOT allowed.
SECURITY: NEVER include actual API keys or secrets in your code. ALWAYS use get_secret('SECRET_NAME').
"""


class ReActAgent:
    """ReAct agent with Thought/Action/Observation loop."""

    def __init__(self, config: dict):
        self.config = config
        self.react_config = config.get("react", {})
        self.max_iterations = self.react_config.get("max_iterations", 10)
        self.enable_tool_creation = self.react_config.get("enable_tool_creation", True)
        self.generated_tools_dir = self.react_config.get("generated_tools_dir", "tools/generated")

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
        log_config = self.config.get("logging", {})
        output_dir = Path(log_config.get("output_dir", "responses")) / self.config["provider"]
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

        with open(filepath, "w") as f:
            json.dump(log_entry, f, indent=2)

        return str(filepath)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for the ReAct agent."""
    parser = argparse.ArgumentParser(description="ReAct agent with self-extending tools")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file path")
    parser.add_argument("-t", "--task", required=True, help="Task for the agent to perform")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Using provider: {config['provider']} ({config[config['provider']]['model']})")
    print("Mode: ReAct")

    agent = ReActAgent(config)
    result = agent.run(args.task)

    # Print trace if verbose
    if args.verbose:
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

    # Print result
    print(f"\nFinal Answer: {result['final_answer']}")
    print(f"Iterations: {result['iterations']}")
    if result["tools_created"]:
        print(f"Tools Created: {', '.join(result['tools_created'])}")
    print(f"Logged to: {result['log_file']}")


if __name__ == "__main__":
    main()
