from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from app.llm.gemini import GeminiProvider
from app.llm.provider import LLMProviderError
from app.schemas.chat import ChatRole, ConversationMessage, StructuredChatResponse
from app.schemas.image import ImageGenerationResponse
from app.schemas.audio import AudioGenerationResponse
from app.schemas.tool import ToolCall
from app.tools import get_demo_tools


def test_gemini_provider_generates_text() -> None:
    client = Mock()
    client.models.generate_content.return_value.text = "Generated text"
    provider = GeminiProvider(
        api_key="test-api-key",
        model="test-model",
        client=client,
    )

    result = provider.generate_text(
        [ConversationMessage(role=ChatRole.USER, content="Write a greeting")],
        temperature=0.35,
    )

    assert result == "Generated text"
    call_arguments = client.models.generate_content.call_args.kwargs
    assert call_arguments["model"] == "test-model"
    assert call_arguments["config"].automatic_function_calling.disable is True
    assert call_arguments["config"].temperature == 0.35
    assert [(content.role, content.parts[0].text) for content in call_arguments["contents"]] == [
        ("user", "Write a greeting")
    ]


def test_gemini_provider_maps_assistant_messages_to_the_model_role() -> None:
    client = Mock()
    client.models.generate_content.return_value.text = "Generated text"
    provider = GeminiProvider(
        api_key="test-api-key",
        model="test-model",
        client=client,
    )

    provider.generate_text(
        [
            ConversationMessage(role=ChatRole.USER, content="Hello"),
            ConversationMessage(role=ChatRole.ASSISTANT, content="Hi there"),
            ConversationMessage(role=ChatRole.USER, content="How are you?"),
        ]
    )

    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert [(content.role, content.parts[0].text) for content in contents] == [
        ("user", "Hello"),
        ("model", "Hi there"),
        ("user", "How are you?"),
    ]


def test_gemini_provider_streams_its_generated_text_as_one_chunk() -> None:
    client = Mock()
    client.models.generate_content.return_value.text = "Generated text"
    provider = GeminiProvider(
        api_key="test-api-key",
        model="test-model",
        client=client,
    )

    chunks = list(
        provider.stream_text(
            [ConversationMessage(role=ChatRole.USER, content="Write a greeting")],
            temperature=0.35,
        )
    )

    assert chunks == ["Generated text"]
    assert client.models.generate_content.call_args.kwargs["config"].temperature == 0.35


def test_gemini_provider_generates_and_validates_structured_output() -> None:
    client = Mock()
    client.models.generate_content.return_value.text = '{"answer": "Generated text"}'
    provider = GeminiProvider(
        api_key="test-api-key",
        model="test-model",
        client=client,
    )

    result = provider.generate_structured_output(
        [ConversationMessage(role=ChatRole.USER, content="Write a greeting")],
        temperature=0.35,
    )

    assert result == StructuredChatResponse(answer="Generated text")
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.temperature == 0.35
    assert config.response_mime_type == "application/json"
    assert config.response_schema is StructuredChatResponse


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
def test_gemini_provider_wraps_invalid_structured_output(raw_output: str) -> None:
    client = Mock()
    client.models.generate_content.return_value.text = raw_output
    provider = GeminiProvider(
        api_key="test-api-key",
        model="test-model",
        client=client,
    )

    with pytest.raises(LLMProviderError, match="Structured text generation failed"):
        provider.generate_structured_output(
            [ConversationMessage(role=ChatRole.USER, content="Write a greeting")]
        )


def test_gemini_provider_rejects_missing_api_key() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiProvider(api_key="", model="test-model")


def test_gemini_provider_rejects_empty_response() -> None:
    client = Mock()
    client.models.generate_content.return_value.text = None
    provider = GeminiProvider(
        api_key="test-api-key",
        model="test-model",
        client=client,
    )

    with pytest.raises(LLMProviderError, match="Text generation failed"):
        provider.generate_text(
            [ConversationMessage(role=ChatRole.USER, content="Write a greeting")]
        )


def test_gemini_provider_wraps_sdk_errors() -> None:
    client = Mock()
    client.models.generate_content.side_effect = RuntimeError("SDK failure: sensitive detail")
    provider = GeminiProvider(
        api_key="test-api-key",
        model="test-model",
        client=client,
    )

    with pytest.raises(LLMProviderError, match="Text generation failed") as error:
        provider.generate_text(
            [ConversationMessage(role=ChatRole.USER, content="Write a greeting")]
        )

    assert isinstance(error.value.__cause__, RuntimeError)


def test_gemini_provider_translates_tool_calls_and_returns_final_tool_text() -> None:
    client = Mock()
    client.models.generate_content.side_effect = [
        Mock(
            function_calls=[
                SimpleNamespace(
                    id="call-1",
                    name="add_numbers",
                    args={"left": 2, "right": 3},
                )
            ]
        ),
        Mock(text="Two plus three is five."),
    ]
    provider = GeminiProvider(
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
    first_config = client.models.generate_content.call_args_list[0].kwargs["config"]
    assert [
        declaration.name for declaration in first_config.tools[0].function_declarations
    ] == ["add_numbers", "get_project_name"]
    final_contents = client.models.generate_content.call_args_list[1].kwargs["contents"]
    assert final_contents[-2].parts[0].function_call.name == "add_numbers"
    assert final_contents[-1].parts[0].function_response.response == {"output": "5"}


def test_gemini_provider_generates_a_provider_neutral_image_result() -> None:
    client = Mock()
    client.models.generate_images.return_value.generated_images = [
        Mock(image=Mock(image_bytes=b"image-bytes", mime_type="image/png"))
    ]
    provider = GeminiProvider(
        api_key="test-api-key",
        model="test-model",
        image_model="test-image-model",
        client=client,
    )

    result = provider.generate_image("A red kite")

    assert result == ImageGenerationResponse(
        image_base64="aW1hZ2UtYnl0ZXM=",
        mime_type="image/png",
    )
    call_arguments = client.models.generate_images.call_args.kwargs
    assert call_arguments["model"] == "test-image-model"
    assert call_arguments["prompt"] == "A red kite"
    assert call_arguments["config"].number_of_images == 1


def test_gemini_provider_wraps_image_generation_errors() -> None:
    client = Mock()
    client.models.generate_images.side_effect = RuntimeError("SDK failure: sensitive detail")
    provider = GeminiProvider(
        api_key="test-api-key",
        model="test-model",
        image_model="test-image-model",
        client=client,
    )

    with pytest.raises(LLMProviderError, match="Image generation failed") as error:
        provider.generate_image("A red kite")

    assert isinstance(error.value.__cause__, RuntimeError)


def test_gemini_provider_wraps_missing_image_output() -> None:
    client = Mock()
    client.models.generate_images.return_value.generated_images = []
    provider = GeminiProvider(
        api_key="test-api-key",
        model="test-model",
        image_model="test-image-model",
        client=client,
    )

    with pytest.raises(LLMProviderError, match="Image generation failed"):
        provider.generate_image("A red kite")


def test_gemini_provider_generates_a_provider_neutral_audio_result() -> None:
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            inline_data=SimpleNamespace(
                                data=b"audio-bytes",
                                mime_type="audio/wav",
                            )
                        )
                    ]
                )
            )
        ]
    )
    provider = GeminiProvider(
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
    call_arguments = client.models.generate_content.call_args.kwargs
    assert call_arguments["model"] == "test-audio-model"
    assert call_arguments["contents"] == "Hello, world."
    assert call_arguments["config"].response_modalities == ["AUDIO"]
    assert call_arguments["config"].speech_config.voice_config.prebuilt_voice_config.voice_name == (
        "test-voice"
    )


def test_gemini_provider_wraps_audio_generation_errors() -> None:
    client = Mock()
    client.models.generate_content.side_effect = RuntimeError("SDK failure: sensitive detail")
    provider = GeminiProvider(
        api_key="test-api-key",
        model="test-model",
        audio_model="test-audio-model",
        audio_voice="test-voice",
        client=client,
    )

    with pytest.raises(LLMProviderError, match="Audio generation failed") as error:
        provider.generate_audio("Hello, world.")

    assert isinstance(error.value.__cause__, RuntimeError)


def test_gemini_provider_wraps_missing_audio_output() -> None:
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(candidates=[])
    provider = GeminiProvider(
        api_key="test-api-key",
        model="test-model",
        audio_model="test-audio-model",
        audio_voice="test-voice",
        client=client,
    )

    with pytest.raises(LLMProviderError, match="Audio generation failed"):
        provider.generate_audio("Hello, world.")
