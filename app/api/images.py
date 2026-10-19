from fastapi import APIRouter, Depends, HTTPException, status

from app.llm.provider import LLMProviderError
from app.schemas.image import ImageGenerationRequest, ImageGenerationResponse
from app.services.image_service import ImageService, get_image_service

router = APIRouter(tags=["images"])


@router.post("/images/generate", response_model=ImageGenerationResponse)
def generate_image(
    request: ImageGenerationRequest,
    image_service: ImageService = Depends(get_image_service),
) -> ImageGenerationResponse:
    try:
        return image_service.generate_image(request.prompt)
    except LLMProviderError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model is temporarily unavailable. Please try again later.",
        ) from None
