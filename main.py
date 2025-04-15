from fastapi import FastAPI
from fish_speech_api.endpoints.speech import router as tts_router

app = FastAPI(
    title="Fish Speech TTS API",
    version="1.0.0",
    description="OpenAI-compatible API for Fish Speech 1.5"
)

app.include_router(tts_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
