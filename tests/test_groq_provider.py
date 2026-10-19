from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from app.llm.groq import GroqProvider
from app.llm.provider import LLMProviderError
from app.schemas.chat import ChatRole, ConversationMessage, StructuredChatResponse
from app.schemas.image import ImageGenerationResponse
from app.schemas.audio import AudioGenerationResponse
from app.schemas.tool import ToolCall
from app.tools import get_demo_tools


def test_groq_provider_generates_text_and_maps_conversation_messages() -> None:
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="Generated text"))
    ]
    provider = GroqProvider(
        api_key="test-api-key",
        model="test-model",
        client=client,
    )

    result = provider.generate_text(
        [
            ConversationMessage(role=ChatRole.USER, content="Hello"),
            ConversationMessage(role=ChatRole.ASSISTANT, content="Hi there"),
        ],
        temperature=0.35,
    )

    assert result == "Generated text"
    client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ],
        temperature=0.35,
    )


def test_groq_provider_streams_text_and_maps_conversation_messages() -> None:
    client = Mock()
    client.chat.completions.create.return_value = [
        Mock(choices=[Mock(delta=Mock(content="Generated "))]),
        Mock(choices=[Mock(delta=Mock(content=None))]),
        Mock(choices=[Mock(delta=Mock(content="text"))]),
    ]
    provider = GroqProvider(
        api_key="test-api-key",
        model="test-model",
        client=client,
    )

    chunks = list(
        provider.stream_text(
            [
                ConversationMessage(role=ChatRole.USER, content="Hello"),
                ConversationMessage(role=ChatRole.ASSISTANT, content="Hi there"),
            ],
            temperature=0.35,
        )
    )

    assert chunks == ["Generated ", "text"]
    client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ],
        temperature=0.35,
        stream=True,
    )


def test_groq_provider_generates_and_validates_structured_output() -> None:
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='{"answer": "Generated text"}'))
    ]
    provider = GroqProvider(
        api_key="test-api-key",
        model="test-model",
        client=client,
    )

    result = provider.generate_structured_output(
        [ConversationMessage(role=ChatRole.USER, content="Hello")],
        temperature=0.35,
    )

    assert result == StructuredChatResponse(answer="Generated text")
    call_arguments = client.chat.completions.create.call_args.kwargs
    assert call_arguments["temperature"] == 0.35
    assert call_arguments["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_chat_response",
            "strict": True,
            "schema": StructuredChatResponse.model_json_schema(),
        },
    }


@pytest.mark.parametrize(
    "raw_output",
    [
        '{"answer":',
        "{}",
        '{"answer": 42}',
        '{"answer": "Generated text", "unexpected": "value"}',
    ],
    ids=["malformed-json", "missing-required-field", "wrong-field-type", "extra-field"],
)
def test_groq_provider_wraps_invalid_structured_output(raw_output: str) -> None:
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content=raw_output))
    ]
    provider = GroqProvider(
        api_key="test-api-key",
        model="test-model",
        client=client,
    )

    with pytest.raises(LLMProviderError, match="Structured text generation failed"):
        provider.generate_structured_output(
            [ConversationMessage(role=ChatRole.USER, content="Hello")]
        )


def test_groq_provider_rejects_missing_api_key() -> None:
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqProvider(api_key="", model="test-model")


def test_groq_provider_rejects_empty_response() -> None:
    client = Mock()
    client.chat.completions.create.return_value.choices = [Mock(message=Mock(content=None))]
    provider = GroqProvider(
        api_key="test-api-key",
        model="test-model",
        client=client,
    )

    with pytest.raises(LLMProviderError, match="Text generation failed"):
        provider.generate_text(
            [ConversationMessage(role=ChatRole.USER, content="Write a greeting")]
        )


def test_groq_provider_wraps_sdk_errors() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("SDK failure: sensitive detail")
    provider = GroqProvider(
        api_key="test-api-key",
        model="test-model",
        client=client,
    )

    with pytest.raises(LLMProviderError, match="Text generation failed") as error:
        provider.generate_text(
            [ConversationMessage(role=ChatRole.USER, content="Write a greeting")]
        )

    assert isinstance(error.value.__cause__, RuntimeError)


def test_groq_provider_wraps_streaming_sdk_errors() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("SDK failure: sensitive detail")
    provider = GroqProvider(
        api_key="test-api-key",
        model="test-model",
        client=client,
    )

    with pytest.raises(LLMProviderError, match="Text generation failed") as error:
        list(
            provider.stream_text(
                [ConversationMessage(role=ChatRole.USER, content="Write a greeting")]
            )
        )

    assert isinstance(error.value.__cause__, RuntimeError)


def test_groq_provider_translates_tool_calls_and_returns_final_tool_text() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        Mock(
            choices=[
                Mock(
                    message=Mock(
                        tool_calls=[
                                SimpleNamespace(
                                    id="call-1",
                                    function=SimpleNamespace(
                                        name="add_numbers",
                                        arguments='{"left": 2, "right": 3}',
                                    ),
                            )
                        ]
                    )
                )
            ]
        ),
        Mock(choices=[Mock(message=Mock(content="Two plus three is five."))]),
    ]
    provider = GroqProvider(
        api_key="test-api-key",
        model="test-model",
        client=client,
    )
    messages = [ConversationMessage(role=ChatRole.USER, content="What is 2 plus 3?")]
    tools = get_demo_tools()

    first_response = provider.request_tool_response(messages, tools, temperature=0.35)
    final_text = provider.generate_text_from_tool_result(
        messages,
        tools,
        ToolCall(id="call-1", name="add_numbers", arguments={"left": 2, "right": 3}),
        "5",
        temperature=0.35,
    )

    assert first_response.tool_call == ToolCall(
        id="call-1",
        name="add_numbers",
        arguments={"left": 2, "right": 3},
    )
    assert final_text == "Two plus three is five."
    first_call = client.chat.completions.create.call_args_list[0].kwargs
    assert first_call["tool_choice"] == "auto"
    assert [tool["function"]["name"] for tool in first_call["tools"]] == [
        "add_numbers",
        "get_project_name",
    ]
    final_call = client.chat.completions.create.call_args_list[1].kwargs
    assert final_call["tool_choice"] == "none"
    assert final_call["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "add_numbers",
        "content": "5",
    }


def test_groq_provider_wraps_malformed_tool_call_arguments() -> None:
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        Mock(
            message=Mock(
                tool_calls=[
                    SimpleNamespace(
                        id="call-1",
                        function=SimpleNamespace(name="add_numbers", arguments="not-json"),
                    )
                ]
            )
        )
    ]
    provider = GroqProvider(api_key="test-api-key", model="test-model", client=client)

    with pytest.raises(LLMProviderError, match="Tool generation failed"):
        provider.request_tool_response(
            [ConversationMessage(role=ChatRole.USER, content="What is 2 plus 3?")],
            get_demo_tools(),
        )


def test_groq_provider_reports_image_generation_as_unsupported() -> None:
    provider = GroqProvider(api_key="test-api-key", model="test-model", client=Mock())

    with pytest.raises(LLMProviderError, match="Image generation is not supported"):
        provider.generate_image("A red kite")


def test_groq_provider_generates_a_provider_neutral_audio_result() -> None:
    client = Mock()
    client.audio.speech.create.return_value.read.return_value = b"audio-bytes"
    provider = GroqProvider(
        api_key="test-api-key",
        model="test-model",
        audio_model="test-audio-model",
        audio_voice="test-voice",
        client=client,
    )

    result = provider.generate_audio("Hello, world.")

    assert result == AudioGenerationResponse(
        audio_base64="YXVkaW8tYnl0ZXM=",
        mime_type="audio/wav",
    )
    client.audio.speech.create.assert_called_once_with(
        model="test-audio-model",
        voice="test-voice",
        input="Hello, world.",
        response_format="wav",
    )


def test_groq_provider_wraps_audio_generation_errors() -> None:
    client = Mock()
    client.audio.speech.create.side_effect = RuntimeError("SDK failure: sensitive detail")
    provider = GroqProvider(
        api_key="test-api-key",
        model="test-model",
        audio_model="test-audio-model",
        audio_voice="test-voice",
        client=client,
    )

    with pytest.raises(LLMProviderError, match="Audio generation failed") as error:
        provider.generate_audio("Hello, world.")

    assert isinstance(error.value.__cause__, RuntimeError)
