from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

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
    
    model_config = {
        'protected_namespaces': ()
    }

class BatchPromptResponse(BaseModel):
    results: List[ModelPromptResponse]

class StatusResponse(BaseModel):
    message: str
    status: str
    ollama_installed: bool
    system: str

# New models for Neo4j integration with PromptGroup hierarchy

class PromptGroupCreate(BaseModel):
    name: str
    description: str
    tags: Optional[List[str]] = []

class PromptGroupResponse(BaseModel):
    id: str
    name: str
    description: str
    tags: List[str]
    created_at: datetime
    prompt_count: int = 0

class PromptCreate(BaseModel):
    prompt_group_id: str
    content: str
    name: str
    description: str
    tags: Optional[List[str]] = []

class PromptResponse(BaseModel):
    id: str
    prompt_group_id: str
    content: str
    name: str
    description: str
    tags: List[str]
    created_at: datetime
    version_count: int = 0

class PromptVersionCreate(BaseModel):
    prompt_id: str
    content: str
    version: str
    derived_from: Optional[str] = None
    notes: Optional[str] = None

class PromptVersionResponse(BaseModel):
    id: str
    prompt_id: str
    content: str
    version: str
    created_at: datetime
    test_runs_count: int = 0
    derived_from: Optional[str] = None
    notes: Optional[str] = None

class TestRunMetrics(BaseModel):
    latency_ms: int
    token_count: int
    token_per_second: float = Field(None)
    custom_scores: Dict[str, float] = {}

class TestRunCreate(BaseModel):
    prompt_version_id: str
    model_used: str
    output: str
    metrics: TestRunMetrics
    input_params: Dict[str, Any] = {}
    
    model_config = {
        'protected_namespaces': ()
    }

class TestRunResponse(BaseModel):
    id: str
    prompt_version_id: str
    model_used: str
    output: str
    metrics: TestRunMetrics
    input_params: Dict[str, Any]
    created_at: datetime
    
    model_config = {
        'protected_namespaces': ()
    }

class PromptComparisonRequest(BaseModel):
    test_run_ids: List[str]

class PromptVersionDetails(BaseModel):
    id: str
    content: str
    version: str

class TestRunComparisonResponse(BaseModel):
    id: str
    model_used: str
    output: str
    metrics: TestRunMetrics
    created_at: datetime
    prompt_version: PromptVersionDetails
    input_params: Dict[str, Any]
    score: Optional[float] = None
    
    model_config = {
        'protected_namespaces': ()
    }