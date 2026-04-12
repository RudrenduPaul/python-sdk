import pytest
from pydantic import BaseModel

from mcp.client import Client
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.tools.base import Tool, _dereference_schema

pytestmark = pytest.mark.anyio


def test_context_detected_in_union_annotation():
    def my_tool(x: int, ctx: Context | None) -> str:
        raise NotImplementedError

    tool = Tool.from_function(my_tool)
    assert tool.context_kwarg == "ctx"


def test_dereference_schema_no_refs():
    """Schema without $defs passes through unchanged."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    assert _dereference_schema(schema) == schema


def test_dereference_schema_simple_ref():
    """Simple $ref gets inlined and $defs removed."""
    schema = {
        "type": "object",
        "properties": {"address": {"$ref": "#/$defs/Address"}},
        "$defs": {"Address": {"type": "object", "properties": {"street": {"type": "string"}}}},
    }
    result = _dereference_schema(schema)
    assert "$ref" not in str(result)
    assert "$defs" not in result
    assert result["properties"]["address"]["properties"]["street"]["type"] == "string"


def test_dereference_schema_circular_ref():
    """Circular $ref is left as-is (graceful fallback)."""
    schema = {
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/Node"}},
        "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}},
    }
    result = _dereference_schema(schema)
    # The outer ref should be resolved, but the inner circular one stays
    assert result["properties"]["node"]["type"] == "object"


def test_dereference_schema_unknown_ref():
    """A $ref pointing to a non-existent definition is left as-is."""
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/Missing"}},
        # $defs has an unrelated entry so defs is truthy, but "Missing" is absent
        "$defs": {"Other": {"type": "string"}},
    }
    result = _dereference_schema(schema)
    assert result["properties"]["x"]["$ref"] == "#/$defs/Missing"


def test_dereference_schema_external_ref():
    """A $ref pointing outside $defs (e.g., external URI) is left as-is."""
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "https://example.com/schema"}},
        "$defs": {"Foo": {"type": "string"}},
    }
    result = _dereference_schema(schema)
    assert result["properties"]["x"]["$ref"] == "https://example.com/schema"


def test_dereference_schema_sibling_properties_merged():
    """Sibling properties on a $ref node (e.g., description) are merged into resolved schema."""
    schema = {
        "type": "object",
        "properties": {"addr": {"$ref": "#/$defs/Addr", "description": "User address"}},
        "$defs": {"Addr": {"type": "object", "properties": {"city": {"type": "string"}}}},
    }
    result = _dereference_schema(schema)
    assert result["properties"]["addr"]["description"] == "User address"
    assert result["properties"]["addr"]["type"] == "object"


def test_dereference_schema_nested_refs():
    """Nested $refs (a definition that itself contains a $ref) are fully resolved."""
    schema = {
        "type": "object",
        "properties": {"order": {"$ref": "#/$defs/Order"}},
        "$defs": {
            "Order": {"type": "object", "properties": {"item": {"$ref": "#/$defs/Item"}}},
            "Item": {"type": "object", "properties": {"sku": {"type": "string"}}},
        },
    }
    result = _dereference_schema(schema)
    assert "$ref" not in str(result)
    assert "$defs" not in result
    assert result["properties"]["order"]["properties"]["item"]["properties"]["sku"]["type"] == "string"


def test_dereference_schema_list_items():
    """$refs inside list items are resolved."""
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"$ref": "#/$defs/Tag"}}},
        "$defs": {"Tag": {"type": "object", "properties": {"label": {"type": "string"}}}},
    }
    result = _dereference_schema(schema)
    assert "$ref" not in str(result)
    assert result["properties"]["tags"]["items"]["properties"]["label"]["type"] == "string"


async def test_tool_schema_refs_are_dereferenced():
    """Tools with nested Pydantic models should have inlined schemas with no $ref."""

    class Address(BaseModel):
        street: str
        city: str

    class UserInput(BaseModel):
        name: str
        address: Address

    mcp = MCPServer()

    @mcp.tool()
    def get_user(user: UserInput) -> str:  # pragma: no cover
        return user.name

    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert len(tools.tools) == 1
        tool = tools.tools[0]
        schema_str = str(tool.input_schema)
        assert "$ref" not in schema_str
        assert "$defs" not in schema_str
        # The nested model should be inlined
        user_prop = tool.input_schema.get("properties", {}).get("user", {})
        assert user_prop.get("type") == "object"
        address_prop = user_prop.get("properties", {}).get("address", {})
        assert address_prop.get("type") == "object"
        assert "street" in address_prop.get("properties", {})
