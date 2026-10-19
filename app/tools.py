"""Small deterministic tools used by the Sprint 5 function-calling demo."""

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.schemas.tool import ToolCall, ToolDefinition


class ToolExecutionError(RuntimeError):
    """Raised when a model requests an unknown or invalid backend tool call."""


class AddNumbersArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: int = Field(ge=-1_000_000, le=1_000_000)
    right: int = Field(ge=-1_000_000, le=1_000_000)


class ProjectNameArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


_TOOL_DEFINITIONS = (
    ToolDefinition(
        name="add_numbers",
        description="Add two whole numbers together.",
        parameters=AddNumbersArguments.model_json_schema(),
    ),
    ToolDefinition(
        name="get_project_name",
        description="Return the name of this application.",
        parameters=ProjectNameArguments.model_json_schema(),
    ),
)


def get_demo_tools() -> tuple[ToolDefinition, ...]:
    """Return the allowlisted tool definitions sent to an LLM provider."""

    return _TOOL_DEFINITIONS


def execute_demo_tool(tool_call: ToolCall) -> str:
    """Validate and execute one allowlisted deterministic demo tool."""

    try:
        if tool_call.name == "add_numbers":
            arguments = AddNumbersArguments.model_validate(tool_call.arguments)
            return str(arguments.left + arguments.right)
        if tool_call.name == "get_project_name":
            ProjectNameArguments.model_validate(tool_call.arguments)
            return "AI Studio"
    except ValidationError as error:
        raise ToolExecutionError("Invalid tool call.") from error

    raise ToolExecutionError("Unknown tool call.")
