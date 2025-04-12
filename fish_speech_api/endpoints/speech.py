from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from fish_speech_api.services.tts_service import generate_tts

router = APIRouter()


@router.post("/speech")
async def speech_endpoint(
    model: str = Form(...),
    input: str = Form(...),
    voice: str = Form(None),
    instructions: str = Form(None),
    reference_audio: UploadFile = File(None)
):
    output_path = generate_tts(
        text=input,
        model_name=model,
        voice_sample=await reference_audio.read() if reference_audio else None,
        voice_name=voice,
        instructions=instructions
    )

    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename="speech.wav"
    )
