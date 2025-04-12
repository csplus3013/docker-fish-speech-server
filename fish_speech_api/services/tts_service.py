import logging
import tempfile
import shutil

from fish_speech_api.utils import get_model_paths, save_temp_audio
from fish_speech_infer import text_to_speech

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_tts(text, model_name, voice_sample=None, voice_name=None, instructions=None):
    model_paths = get_model_paths(model_name)

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Reference voice
            reference_audio_path = None
            if voice_sample:
                reference_audio_path = save_temp_audio(voice_sample, temp_dir=temp_dir)

            # Output file path inside temp
            output_path = f"{temp_dir}/output.wav"

            text_to_speech(
                text=text,
                output_path=output_path,
                reference_audio_path=reference_audio_path,
                checkpoint_dir=model_paths["path"],
                device="cuda"
            )

            # Copy output to a stable location if needed (e.g., returned from an endpoint)
            final_output_path = f"{tempfile.gettempdir()}/fish_tts_output.wav"
            shutil.copy(output_path, final_output_path)

            return final_output_path

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise
