import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import FileResponse
from fastapi.exceptions import HTTPException
from fish_speech_api.services.tts_service import generate_tts

# Configure logger for this module
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("fish_speech_api.routes.speech")

router = APIRouter()


@router.post("/audio/speech")
async def speech_endpoint(
    model: str = Form(...),
    input: str = Form(...),
    voice: str = Form(default=None),
    top_p: float = Form(default=0.7),
    repetition_penalty: float = Form(default=1.5),
    temperature: float = Form(default=0.7),
    chunk_length: int = Form(default=150),
    max_new_tokens: int = Form(default=2048),
    seed: int = Form(default=None),
    reference_audio: UploadFile = File(default=None),
):
    logger.info(f"TTS request | Model: {model} | Chars: {len(input)}")

    try:
        # Валидация входных параметров
        if len(input) > 4096:
            raise HTTPException(status_code=400, detail="Input too long (max 4096 chars)")

        if reference_audio and reference_audio.size > 25 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Audio file too large (max 25MB)")

        # Обработка референсного аудио
        audio_bytes = None
        if reference_audio:
            if not reference_audio.filename.lower().endswith('.wav'):
                raise HTTPException(status_code=400, detail="Invalid audio format")

            audio_bytes = await reference_audio.read()
            logger.info(f"Received reference audio: {len(audio_bytes)} bytes")

        # Генерация речи
        output_path = generate_tts(
            text=input,
            model_name=model,
            voice_sample=audio_bytes,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            chunk_length=chunk_length,
            max_new_tokens=max_new_tokens,
            seed=seed
        )

        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=Path(output_path).name
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
