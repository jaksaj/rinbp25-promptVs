from pydantic import BaseModel
from typing import List, Optional

class ModelRequest(BaseModel):
    model_name: str

class ModelResponse(BaseModel):
    message: str
    model_name: str

class ModelsListResponse(BaseModel):
    models: List[str]
    current_model: Optional[str] = None 