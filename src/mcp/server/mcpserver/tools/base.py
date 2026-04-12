from __future__ import annotations

import copy
from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.context_injection import find_context_parameter
from mcp.server.mcpserver.utilities.func_metadata import FuncMetadata, func_metadata
from mcp.shared._callable_inspection import is_async_callable
from mcp.shared.exceptions import UrlElicitationRequiredError
from mcp.shared.tool_name_validation import validate_and_warn_tool_name
from mcp.types import Icon, ToolAnnotations

if TYPE_CHECKING:
    from mcp.server.context import LifespanContextT, RequestT
    from mcp.server.mcpserver.context import Context


def _dereference_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline local $ref definitions in a JSON schema.

    Pydantic's model_json_schema() may emit $ref/$defs for nested models.
    Some LLM clients cannot resolve $ref, so we inline them here for
    parity with the TypeScript SDK (modelcontextprotocol/typescript-sdk#1563).
    """
    defs = schema.get("$defs")
    if not defs:
        return schema

    def _resolve(node: Any, visiting: frozenset[str] = frozenset()) -> Any:
        if isinstance(node, list):
            node_list = cast(list[Any], node)
            return [_resolve(item, visiting) for item in node_list]
        if not isinstance(node, dict):
            return node
        node_dict = cast(dict[str, Any], node)
        ref_path = node_dict.get("$ref")
        if ref_path is not None:
            if not isinstance(ref_path, str) or not ref_path.startswith("#/$defs/"):
                return node_dict
            def_name: str = ref_path[len("#/$defs/") :]
            if def_name in visiting:
                return node_dict  # circular ref — leave as-is
            if def_name not in defs:
                return node_dict
            resolved: dict[str, Any] = _resolve(copy.deepcopy(defs[def_name]), visiting | {def_name})
            # Merge any sibling properties (e.g., description override)
            siblings = {k: v for k, v in node_dict.items() if k != "$ref"}
            return {**resolved, **siblings}
        return {k: _resolve(v, visiting) for k, v in node_dict.items()}

    result = _resolve(schema)
    result.pop("$defs", None)
    return result


class Tool(BaseModel):
    """Internal tool registration info."""

    fn: Callable[..., Any] = Field(exclude=True)
    name: str = Field(description="Name of the tool")
    title: str | None = Field(None, description="Human-readable title of the tool")
    description: str = Field(description="Description of what the tool does")
    parameters: dict[str, Any] = Field(description="JSON schema for tool parameters")
    fn_metadata: FuncMetadata = Field(
        description="Metadata about the function including a pydantic model for tool arguments"
    )
    is_async: bool = Field(description="Whether the tool is async")
    context_kwarg: str | None = Field(None, description="Name of the kwarg that should receive context")
    annotations: ToolAnnotations | None = Field(None, description="Optional annotations for the tool")
    icons: list[Icon] | None = Field(default=None, description="Optional list of icons for this tool")
    meta: dict[str, Any] | None = Field(default=None, description="Optional metadata for this tool")

    @cached_property
    def output_schema(self) -> dict[str, Any] | None:
        return self.fn_metadata.output_schema

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        context_kwarg: str | None = None,
        annotations: ToolAnnotations | None = None,
        icons: list[Icon] | None = None,
        meta: dict[str, Any] | None = None,
        structured_output: bool | None = None,
    ) -> Tool:
        """Create a Tool from a function."""
        func_name = name or fn.__name__

        validate_and_warn_tool_name(func_name)

        if func_name == "<lambda>":
            raise ValueError("You must provide a name for lambda functions")

        func_doc = description or fn.__doc__ or ""
        is_async = is_async_callable(fn)

        if context_kwarg is None:  # pragma: no branch
            context_kwarg = find_context_parameter(fn)

        func_arg_metadata = func_metadata(
            fn,
            skip_names=[context_kwarg] if context_kwarg is not None else [],
            structured_output=structured_output,
        )
        parameters = _dereference_schema(func_arg_metadata.arg_model.model_json_schema(by_alias=True))

        return cls(
            fn=fn,
            name=func_name,
            title=title,
            description=func_doc,
            parameters=parameters,
            fn_metadata=func_arg_metadata,
            is_async=is_async,
            context_kwarg=context_kwarg,
            annotations=annotations,
            icons=icons,
            meta=meta,
        )

    async def run(
        self,
        arguments: dict[str, Any],
        context: Context[LifespanContextT, RequestT],
        convert_result: bool = False,
    ) -> Any:
        """Run the tool with arguments.

        Raises:
            ToolError: If the tool function raises during execution.
        """
        try:
            result = await self.fn_metadata.call_fn_with_arg_validation(
                self.fn,
                self.is_async,
                arguments,
                {self.context_kwarg: context} if self.context_kwarg is not None else None,
            )

            if convert_result:
                result = self.fn_metadata.convert_result(result)

            return result
        except UrlElicitationRequiredError:
            # Re-raise UrlElicitationRequiredError so it can be properly handled
            # as an MCP error response with code -32042
            raise
        except Exception as e:
            raise ToolError(f"Error executing tool {self.name}: {e}") from e
