# ReAct Agent with Self-Extending Tools

A ReAct (Reasoning and Acting) agent that can reason about tasks using Thought/Action/Observation loops, use built-in tools, and dynamically generate new tools when needed.

## Features

- **ReAct Framework**: Step-by-step reasoning with tool use
- **Multi-Provider LLM Support**: Anthropic, OpenAI, and Gemini
- **Built-in Tools**: Calculator, control operations
- **Dynamic Tool Creation**: Agent can create new tools at runtime
- **Secure Secrets Management**: Tools can access API keys without exposing them to the agent
- **Baseline Mode**: Direct prompting without ReAct framework for comparison

## Setup

1. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

2. Configure LLM API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your LLM provider API keys
   ```

3. (Optional) Configure tool secrets:
   ```bash
   cp .env.tools.example .env.tools
   # Edit .env.tools with API keys for generated tools
   ```

4. (Optional) Edit `config.yaml` to select provider and model settings.

## Usage

### ReAct Agent

Run the agent with a task:
```bash
uv run react_agent.py -t "What is the square root of 144?"
```

With verbose trace output:
```bash
uv run react_agent.py -t "Calculate 2^10" -v
```

### Baseline Prompting

Direct prompting without ReAct framework (for comparison):
```bash
uv run llm_prompt.py -p "What is 2+2?"
```

Interactive mode:
```bash
uv run llm_prompt.py
```

From file:
```bash
uv run llm_prompt.py -f prompt.txt
```

## Built-in Tools

| Tool | Operation | Description |
|------|-----------|-------------|
| `control` | `control.finish(answer)` | Complete task with final answer |
| `control` | `control.create_tool(name, description, code)` | Create a new tool dynamically |
| `calculator` | `calculator.evaluate(expression)` | Evaluate math expressions |
| `calculator` | `calculator.sqrt(number)` | Square root |
| `calculator` | `calculator.power(base, exponent)` | Exponentiation |

## Creating Tools Dynamically

The agent can create new tools at runtime. Example task:

```bash
uv run react_agent.py -t "Create a tool to reverse strings, then reverse 'hello world'" -v
```

The agent will:
1. Create a `StringTool` with a `reverse` operation
2. Save it to `tools/generated/stringtool.py`
3. Use `string.reverse` to reverse the string
4. Return the result

Generated tools are automatically loaded on subsequent runs.

## Secure Secrets Management

Tools can access API keys without the agent ever seeing the values.

1. Add secrets to `.env.tools` (must contain `TOOL_API_KEY` in name):
   ```
   OPENWEATHER_TOOL_API_KEY=your_key_here
   ```

2. The agent sees only the secret name in its system prompt:
   ```
   Available secrets (use get_secret('NAME') in tools):
     - OPENWEATHER_TOOL_API_KEY
   ```

3. Generated tools use `get_secret()`:
   ```python
   from secrets_manager import get_secret

   api_key = get_secret('OPENWEATHER_TOOL_API_KEY')
   ```

4. The actual value is only retrieved at tool execution time.

**Security**: The secrets manager blocks access to non-tool secrets (like LLM provider keys) even if the agent tries to access them.

## Configuration

Edit `config.yaml`:

```yaml
# LLM provider: anthropic, openai, or gemini
provider: anthropic

# Provider-specific settings
anthropic:
  model: claude-sonnet-4-20250514
  max_tokens: 1024
  temperature: 0.7

# ReAct agent settings
react:
  max_iterations: 10
  enable_tool_creation: true
  generated_tools_dir: tools/generated

# Logging
logging:
  output_dir: responses
  format: json
```

## Response Logging

Responses are saved to `responses/<provider>/`:

| Mode | File Pattern | Contents |
|------|--------------|----------|
| ReAct | `react_<timestamp>.json` | Full trace with thoughts, actions, observations |
| Baseline | `base_<timestamp>.json` | Direct response |

Example ReAct log:
```json
{
  "timestamp": "2026-03-27T10:00:00",
  "provider": "anthropic",
  "model": "claude-sonnet-4-20250514",
  "task": "What is 2+2?",
  "trace": [
    {"type": "thought", "content": "I need to calculate 2+2"},
    {"type": "action", "operation": "calculator.evaluate", "input": {"expression": "2+2"}},
    {"type": "observation", "content": "4"}
  ],
  "final_answer": "4",
  "iterations": 2,
  "tools_created": []
}
```

## Project Structure

```
prompting/
├── config.yaml           # Configuration
├── .env                  # LLM provider API keys
├── .env.tools            # Tool secrets (secure)
├── react_agent.py        # ReAct agent
├── llm_client.py         # Multi-provider LLM client
├── llm_prompt.py         # Simple prompting script
├── secrets_manager.py    # Secure secrets access
├── tools/
│   ├── __init__.py
│   ├── base.py           # Tool base class and registry
│   ├── builtin.py        # Built-in tools
│   └── generated/        # Dynamically generated tools
└── responses/            # Logged responses
```

## Programmatic Usage

ReAct agent:
```python
from react_agent import ReActAgent, load_config

config = load_config()
agent = ReActAgent(config)
result = agent.run("Calculate 15 * 7")

print(result["final_answer"])
print(result["iterations"])
print(result["tools_created"])
```

Baseline prompting:
```python
from llm_prompt import prompt_llm

result = prompt_llm("Calculate 15 * 7")
print(result["response"])
```
