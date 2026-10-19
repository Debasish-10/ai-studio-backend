from pydantic import BaseModel, Field


class AudioGenerationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4_000)


class AudioGenerationResponse(BaseModel):
    """Provider-neutral generated audio returned as Base64-encoded bytes."""

    audio_base64: str = Field(min_length=1)
    mime_type: str = Field(min_length=1, max_length=100)
