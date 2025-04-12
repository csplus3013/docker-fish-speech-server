import logging
import tempfile
import shutil

from fish_speech_api.utils import get_model_paths, save_temp_audio, get_temp_file
from fish_speech_infer import text_to_speech

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("fish_speech_api.services.tts_service")


def generate_tts(
    text: str,
    model_name: str,
    voice_sample: bytes = None,
    voice_name: str = None,
    instructions: str = None,
    top_p: float = 0.7,
    temperature: float = 0.7,
    repetition_penalty: float = 1.2,
    chunk_length: int = 200,
    max_new_tokens: int = 1024,
    seed: int = None,
    compile_model: bool = False,
):
    logger.info(f"Starting TTS generation for model: {model_name}")

    model_paths = get_model_paths(model_name)

    try:
        # Optional: process voice sample to temp file
        reference_audio_path = None
        if voice_sample:
            logger.info("Processing reference voice sample...")
            reference_audio_path = save_temp_audio(voice_sample)

        # Create a safe temp output file
        output_file = get_temp_file()
        output_file_path = output_file.name

        logger.info("Calling text_to_speech...")
        text_to_speech(
            text=text,
            output_path=output_file_path,
            reference_audio_path=reference_audio_path,
            checkpoint_dir=model_paths["path"],
            device="cuda",
            compile_model=compile_model,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            chunk_length=chunk_length,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

        # Move output to final location (if needed)
        final_output_path = f"{tempfile.gettempdir()}/fish_tts_output.wav"
        shutil.copy(output_file_path, final_output_path)
        logger.info(f"TTS output saved to: {final_output_path}")

        return final_output_path

    except Exception as e:
        logger.exception(f"TTS generation failed: {e}")
        raise
