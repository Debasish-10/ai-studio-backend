import pytest

from app.schemas.tool import ToolCall
from app.tools import ToolExecutionError, execute_demo_tool, get_demo_tools


def test_demo_tools_execute_allowlisted_calls() -> None:
    assert [tool.name for tool in get_demo_tools()] == ["add_numbers", "get_project_name"]
    assert execute_demo_tool(
        ToolCall(id="call-1", name="add_numbers", arguments={"left": 2, "right": 3})
    ) == "5"
    assert execute_demo_tool(
        ToolCall(id="call-2", name="get_project_name", arguments={})
    ) == "AI Studio"


@pytest.mark.parametrize(
    "tool_call",
    [
        ToolCall(id="call-1", name="unknown_tool", arguments={}),
        ToolCall(id="call-2", name="add_numbers", arguments={"left": "two"}),
    ],
    ids=["unknown-tool", "invalid-arguments"],
)
def test_demo_tools_reject_unknown_or_invalid_calls(tool_call: ToolCall) -> None:
    with pytest.raises(ToolExecutionError):
        execute_demo_tool(tool_call)
