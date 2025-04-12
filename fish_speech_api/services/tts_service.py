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


def generate_tts(text, model_name, voice_sample=None, voice_name=None, instructions=None):
    logger.info(f"Starting TTS generation for model: {model_name}")

    model_paths = get_model_paths(model_name)
    logger.debug(f"Loaded model paths: {model_paths}")

    try:
        # Reference voice
        reference_audio_path = None
        if voice_sample:
            logger.info("Processing reference voice sample...")
            reference_audio_path = save_temp_audio(voice_sample)
            logger.debug(f"Saved reference audio to: {reference_audio_path}")

        # Output file path inside temp
        output_file = get_temp_file()
        output_file_path = output_file.name
        logger.debug(f"Temporary output file created at: {output_file_path}")

        logger.info("Calling text_to_speech...")
        text_to_speech(
            text=text,
            output_path=output_file_path,
            reference_audio_path=reference_audio_path,
            checkpoint_dir=model_paths["path"],
            device="cuda"
        )

        # Copy output to a stable location if needed (e.g., returned from an endpoint)
        final_output_path = f"{tempfile.gettempdir()}/fish_tts_output.wav"
        shutil.copy(output_file_path, final_output_path)
        logger.info(f"TTS output saved to: {final_output_path}")

        return final_output_path

    except Exception as e:
        logger.exception(f"TTS generation failed: {e}")
        raise
