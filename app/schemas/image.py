from pydantic import BaseModel, Field


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4_000)


class ImageGenerationResponse(BaseModel):
    """Provider-neutral generated image returned as Base64-encoded bytes."""

    image_base64: str = Field(min_length=1)
    mime_type: str = Field(min_length=1, max_length=100)
