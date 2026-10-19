from fastapi import APIRouter, Depends, HTTPException, status

from app.llm.provider import LLMProviderError
from app.schemas.audio import AudioGenerationRequest, AudioGenerationResponse
from app.services.audio_service import AudioService, get_audio_service

router = APIRouter(tags=["audio"])


@router.post("/audio/generate", response_model=AudioGenerationResponse)
def generate_audio(
    request: AudioGenerationRequest,
    audio_service: AudioService = Depends(get_audio_service),
) -> AudioGenerationResponse:
    try:
        return audio_service.generate_audio(request.text)
    except LLMProviderError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model is temporarily unavailable. Please try again later.",
        ) from None
