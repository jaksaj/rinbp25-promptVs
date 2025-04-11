from pydantic import BaseModel
from typing import List, Optional, Dict

class ModelRequest(BaseModel):
    model_name: str
    
    model_config = {
        'protected_namespaces': ()
    }

class BatchModelRequest(BaseModel):
    models: List[str]

class ModelResponse(BaseModel):
    message: str
    model_name: Optional[str] = None
    
    model_config = {
        'protected_namespaces': ()
    }

class ModelsListResponse(BaseModel):
    models: List[str]

class RunningModelStatus(BaseModel):
    name: str
    is_running: bool

class RunningModelsResponse(BaseModel):
    running_models: List[RunningModelStatus]

class ModelStatusResponse(BaseModel):
    name: str
    is_pulled: bool
    is_running: bool
    status_message: str