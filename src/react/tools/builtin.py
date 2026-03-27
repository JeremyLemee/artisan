"""
Built-in tools for the ReAct agent.
Includes ControlTool for agent control and CalculatorTool for math operations.
"""

import ast
import importlib.util
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Tool, operation

if TYPE_CHECKING:
    from .base import ToolRegistry


class FinishSignal(Exception):
    """Signal that the agent should finish with an answer."""
    def __init__(self, answer: str):
        self.answer = answer
        super().__init__(answer)


class ControlTool(Tool):
    """Tool for agent control operations."""

    name = "control"
    description = "Agent control operations including finishing tasks and creating new tools"

    def __init__(self, registry: "ToolRegistry", generated_tools_dir: str = "src/react/tools/generated"):
        self._registry = registry
        self._generated_tools_dir = Path(generated_tools_dir)
        self._generated_tools_dir.mkdir(parents=True, exist_ok=True)

    @operation
    def finish(self, answer: str) -> str:
        """Complete the task with the final answer."""
        raise FinishSignal(answer)

    @operation
    def create_tool(self, name: str, description: str, code: str) -> str:
        """Create a new tool class dynamically from Python code."""
        # Basic safety checks
        if not self._validate_tool_code(code):
            return "Error: Tool code failed safety validation"

        # Write the tool to a file
        tool_file = self._generated_tools_dir / f"{name.lower()}.py"
        try:
            with open(tool_file, "w") as f:
                f.write(code)
        except IOError as e:
            return f"Error writing tool file: {e}"

        # Load the module dynamically
        try:
            spec = importlib.util.spec_from_file_location(name, tool_file)
            if spec is None or spec.loader is None:
                return "Error: Could not create module spec"

            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        except Exception as e:
            return f"Error loading tool module: {e}"

        # Find and instantiate the Tool class
        tool_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                issubclass(attr, Tool) and
                attr is not Tool):
                tool_class = attr
                break

        if tool_class is None:
            return "Error: No Tool subclass found in the provided code"

        try:
            tool_instance = tool_class()
            self._registry.register_tool(tool_instance)
        except Exception as e:
            return f"Error instantiating tool: {e}"

        # Return list of new operations
        operations = tool_instance.get_operations()
        op_names = [op.name for op in operations]
        return f"Tool '{tool_instance.name}' created with operations: {', '.join(op_names)}"

    def _validate_tool_code(self, code: str) -> bool:
        """Basic safety validation for tool code."""
        # Check for obviously dangerous patterns
        dangerous_patterns = [
            "os.system",
            "subprocess",
            "eval(",
            "exec(",
            "__import__",
            "open(",  # Allow with caution in create_tool, but warn about it
        ]

        # Parse the code to check for syntax errors
        try:
            ast.parse(code)
        except SyntaxError:
            return False

        # Check for dangerous patterns (basic check, not comprehensive)
        code_lower = code.lower()
        for pattern in dangerous_patterns[:4]:  # Only block the most dangerous ones
            if pattern.lower() in code_lower:
                return False

        return True


class CalculatorTool(Tool):
    """Tool for mathematical calculations."""

    name = "calculator"
    description = "Mathematical calculation operations"

    @operation
    def evaluate(self, expression: str) -> str:
        """Evaluate a mathematical expression safely."""
        # Define allowed names for safe evaluation
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "int": int,
            "float": float,
            # Math module functions
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }

        try:
            # Parse the expression to an AST
            tree = ast.parse(expression, mode="eval")

            # Validate the AST only contains allowed operations
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if node.id not in allowed_names:
                        return f"Error: Unknown name '{node.id}'"
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id not in allowed_names:
                            return f"Error: Unknown function '{node.func.id}'"

            # Compile and evaluate
            code = compile(tree, "<expression>", "eval")
            result = eval(code, {"__builtins__": {}}, allowed_names)
            return str(result)
        except SyntaxError as e:
            return f"Error: Invalid expression syntax - {e}"
        except Exception as e:
            return f"Error: {e}"

    @operation
    def sqrt(self, number: float) -> str:
        """Calculate the square root of a number."""
        try:
            if number < 0:
                return "Error: Cannot calculate square root of negative number"
            result = math.sqrt(number)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    @operation
    def power(self, base: float, exponent: float) -> str:
        """Calculate base raised to the power of exponent."""
        try:
            result = math.pow(base, exponent)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
