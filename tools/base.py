#!/usr/bin/env python3
"""
Tool infrastructure for the ReAct agent.
Provides base classes, decorators, and registry for tools.
"""

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_type_hints


def operation(func: Callable) -> Callable:
    """Decorator to mark a method as a tool operation."""
    func._is_operation = True
    return func


@dataclass
class Operation:
    """Represents a callable operation within a tool."""
    name: str                    # e.g., "calculator.add"
    description: str             # From docstring
    parameters: dict             # JSON schema for parameters
    tool: "Tool"                 # Parent tool instance
    method: Callable             # The actual method to call

    def execute(self, **kwargs) -> Any:
        """Execute the operation with given parameters."""
        return self.method(**kwargs)

    def __str__(self) -> str:
        params_str = ", ".join(
            f"{name}: {info.get('type', 'any')}"
            for name, info in self.parameters.get("properties", {}).items()
        )
        return f"{self.name}({params_str}) - {self.description}"


class Tool:
    """Base class for all tools."""
    name: str = "tool"
    description: str = "A tool"

    def get_operations(self) -> list[Operation]:
        """Return all operations this tool provides."""
        operations = []

        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr = getattr(self, attr_name)
            if callable(attr) and getattr(attr, "_is_operation", False):
                op = self._create_operation(attr_name, attr)
                operations.append(op)

        return operations

    def _create_operation(self, method_name: str, method: Callable) -> Operation:
        """Create an Operation from a method."""
        # Get description from docstring
        description = (method.__doc__ or "").strip().split("\n")[0]

        # Build JSON schema from type hints
        parameters = self._build_parameter_schema(method)

        return Operation(
            name=f"{self.name}.{method_name}",
            description=description,
            parameters=parameters,
            tool=self,
            method=method
        )

    def _build_parameter_schema(self, method: Callable) -> dict:
        """Build JSON schema for method parameters."""
        sig = inspect.signature(method)
        hints = get_type_hints(method) if hasattr(method, "__annotations__") else {}

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = hints.get(param_name, Any)
            json_type = self._python_type_to_json(param_type)

            properties[param_name] = {"type": json_type}

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def _python_type_to_json(self, python_type) -> str:
        """Convert Python type to JSON schema type."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        # Handle Optional and other complex types
        origin = getattr(python_type, "__origin__", None)
        if origin is not None:
            # e.g., Optional[str] -> str
            args = getattr(python_type, "__args__", ())
            if args:
                return self._python_type_to_json(args[0])
        return type_map.get(python_type, "string")


class ToolRegistry:
    """Registry for managing tools and their operations."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._operations: dict[str, Operation] = {}

    def register_tool(self, tool: Tool) -> None:
        """Register a tool and all its operations."""
        self._tools[tool.name] = tool

        for op in tool.get_operations():
            self._operations[op.name] = op

    def unregister_tool(self, tool_name: str) -> None:
        """Unregister a tool and all its operations."""
        if tool_name in self._tools:
            tool = self._tools[tool_name]
            for op in tool.get_operations():
                if op.name in self._operations:
                    del self._operations[op.name]
            del self._tools[tool_name]

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)

    def get_operation(self, operation_name: str) -> Optional[Operation]:
        """Get an operation by its full name (e.g., 'calculator.add')."""
        return self._operations.get(operation_name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def list_operations(self) -> list[Operation]:
        """List all registered operations."""
        return list(self._operations.values())

    def execute(self, operation_name: str, **kwargs) -> Any:
        """Execute an operation by name."""
        op = self.get_operation(operation_name)
        if op is None:
            raise ValueError(f"Unknown operation: {operation_name}")
        return op.execute(**kwargs)

    def format_operations_for_prompt(self) -> str:
        """Format all operations as a string for the system prompt."""
        lines = ["Available operations:"]
        for op in self.list_operations():
            lines.append(f"  - {op}")
        return "\n".join(lines)
