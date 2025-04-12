from pydantic import BaseModel


class TTSRequest(BaseModel):
    model: str
    input: str
    voice: str = None
    instructions: str = None
