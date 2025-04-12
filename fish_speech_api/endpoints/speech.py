import logging
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from fish_speech_api.services.tts_service import generate_tts

# Configure logger for this module
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("fish_speech_api.routes.speech")

router = APIRouter()


@router.post("/speech")
async def speech_endpoint(
    model: str = Form(...),
    input: str = Form(...),
    voice: str = Form(None),
    instructions: str = Form(None),
    top_p: float = Form(0.7),
    repetition_penalty: float = Form(1.2),
    temperature: float = Form(0.7),
    chunk_length: int = Form(200),
    max_new_tokens: int = Form(1024),
    seed: int = Form(None),
    reference_audio: UploadFile = File(None)
):
    logger.info(f"Received TTS request - model: {model}, voice: {voice}")

    try:
        # Read uploaded reference audio if provided
        audio_bytes = await reference_audio.read() if reference_audio else None
        if reference_audio:
            logger.info(f"Reference audio file received: {reference_audio.filename} ({len(audio_bytes)} bytes)")

        # Call the TTS service
        logger.info("Invoking TTS generation service...")
        output_path = generate_tts(
            text=input,
            model_name=model,
            voice_sample=audio_bytes,
            voice_name=voice,
            instructions=instructions,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            chunk_length=chunk_length,
            max_new_tokens=max_new_tokens,
            seed=seed
        )
        logger.info(f"TTS generation completed. Output file: {output_path}")

        # Return audio file as response
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename="speech.wav"
        )

    except Exception as e:
        logger.exception(f"Error while handling TTS request: {e}")
        return {"error": str(e)}
