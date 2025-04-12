import logging
import tempfile

from fish_speech_api.utils import get_model_paths, save_temp_audio
from fish_speech_infer import text_to_speech

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_tts(text, model_name, voice_sample=None, voice_name=None, instructions=None):
    model_paths = get_model_paths(model_name)

    # Reference voice
    reference_path = None
    if voice_sample:
        reference_path = save_temp_audio(voice_sample)

    # Output file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="temp") as out_file:
        output_path = out_file.name
        logger.debug("Output file will be saved to: %s", output_path)

    text_to_speech(
        text=text,
        output_path=output_path,
        reference_audio=reference_path,
        checkpoint_dir=model_paths["path"],
        device="cuda"
    )

    return output_path
