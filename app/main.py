from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.audio import router as audio_router
from app.api.chat import router as chat_router
from app.api.evaluations import router as evaluations_router
from app.api.images import router as images_router
from app.api.prompts import router as prompts_router
from app.core.config import settings

app = FastAPI(
    title="AI Studio API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(audio_router)
app.include_router(evaluations_router)
app.include_router(images_router)
app.include_router(prompts_router)


@app.get("/health")
def health_check():
    return {"status": "ok boss dear"}
