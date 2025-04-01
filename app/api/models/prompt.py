from pydantic import BaseModel
from typing import List

class PromptRequest(BaseModel):
    prompt: str

class PromptResponse(BaseModel):
    result: str

class BatchPromptRequest(BaseModel):
    prompt: str
    models: List[str]

class ModelPromptResponse(BaseModel):
    model: str
    response: str

class BatchPromptResponse(BaseModel):
    results: List[ModelPromptResponse]

class StatusResponse(BaseModel):
    message: str
    status: str
    ollama_installed: bool
    system: str 