import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_temp_file(suffix=".wav", dir="temp"):
    return tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=dir)


def save_temp_audio(content: bytes) -> str:
    try:
        temp_file = get_temp_file()
        temp_file.write(content)
        temp_file_path = temp_file.name
        temp_file.close()

        logger.info(f"Temporary audio file saved at: {temp_file_path}")
        return temp_file_path

    except Exception as e:
        logger.error(f"Failed to save temporary audio file: {e}")
        raise
