from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str

class PromptResponse(BaseModel):
    result: str

class StatusResponse(BaseModel):
    message: str
    status: str
    ollama_installed: bool
    system: str 