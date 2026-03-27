# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/react/`. Use `agent.py` for the ReAct loop, `baseline.py` for direct prompting, `client.py` for provider integrations, and `secrets.py` for protected secret access. Tooling code is under `src/react/tools/`: shared interfaces in `base.py`, built-ins in `builtin.py`, and runtime-generated tools in `generated/`. Configuration lives in `config/config.yaml`. Dependency metadata is in `pyproject.toml`, and resolved versions are pinned in `uv.lock`.

## Build, Test, and Development Commands
Set up the environment with `uv venv` and install the package with `uv pip install -e .`.

- `uv run react -t "Calculate 2^10" -v`: run the ReAct agent with verbose tracing.
- `uv run baseline -p "What is 2+2?"`: run the baseline prompt flow.
- `uv run ruff check .`: lint the repository.
- `uv run pyright`: run static type checks.

If you add tests, prefer `pytest` and run them with `uv run pytest`.

## Coding Style & Naming Conventions
Target Python 3.11+ and use 4-space indentation. Follow standard Python naming: `snake_case` for functions and modules, `PascalCase` for classes, and clear verb-based names for tool operations. Keep modules focused; provider logic belongs in `client.py`, and tool registration logic belongs in `src/react/tools/`. Use `ruff` to keep imports and formatting clean, and keep types explicit enough for `pyright` to pass.

## Testing Guidelines
There is no committed `tests/` directory yet. New features should add focused `pytest` coverage under `tests/`, named `test_<feature>.py`. Test behavior that is easy to regress: tool execution, config loading, provider selection, and generated-tool loading. Prefer fast unit tests over networked integration tests.

## Commit & Pull Request Guidelines
Current history uses short, imperative commit messages such as `Add MIT License to the project` and `Refactoring`. Keep that pattern: one clear action per commit. Pull requests should explain the behavioral change, list verification steps run locally, and note any config or secret setup needed. Include sample CLI output when changing agent behavior or tool execution.

## Security & Configuration Tips
Do not commit `config/.env` or `config/.env.tools`. Tool secrets must use names containing `TOOL_API_KEY`. Keep generated tools free of hardcoded credentials and load secrets through `react.secrets.get_secret()`.
