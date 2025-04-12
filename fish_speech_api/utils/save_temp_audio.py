import os
import uuid


def save_temp_audio(content: bytes) -> str:
    os.makedirs("temp", exist_ok=True)
    path = os.path.join("temp", f"{uuid.uuid4().hex}.wav")
    with open(path, "wb") as f:
        f.write(content)
    return path
