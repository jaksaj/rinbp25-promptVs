from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class PromptRequest(BaseModel):
    prompt: str

class SimplePromptResponse(BaseModel):
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
    expected_solution: Optional[str] = None
    tags: Optional[List[str]] = []

class PromptResponse(BaseModel):
    id: str
    prompt_group_id: str
    content: str
    name: str
    description: str
    expected_solution: Optional[str] = None
    tags: List[str]
    created_at: datetime
    version_count: int = 0

class PromptVersionCreate(BaseModel):
    prompt_id: str
    content: str
    version: str
    expected_solution: Optional[str] = None
    derived_from: Optional[str] = None
    notes: Optional[str] = None

class PromptVersionResponse(BaseModel):
    id: str
    prompt_id: str
    content: str
    version: str
    expected_solution: Optional[str] = None
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

# New models for batch operations

class QuestionAnswer(BaseModel):
    question: str
    answer: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class BatchQuestionImport(BaseModel):
    questions: List[QuestionAnswer]
    prompt_group_name: str
    prompt_group_description: Optional[str] = "Imported questions batch"
    tags: Optional[List[str]] = []

class PromptVersionTemplate(BaseModel):
    template: str
    version_name: str
    notes: Optional[str] = None
    
class BatchVersionCreate(BaseModel):
    prompt_ids: List[str]
    templates: List[PromptVersionTemplate]
    
class BatchTestRequest(BaseModel):
    prompt_version_ids: List[str]
    models: List[str]
    
class BatchTestResponse(BaseModel):
    total_tests: int
    completed: int
    failed: int
    test_run_ids: List[str]
    batch_id: Optional[str] = None
    
class TestResultsSummary(BaseModel):
    model: str
    prompt_version: str
    avg_latency_ms: float
    avg_token_count: float
    avg_tokens_per_second: float
    total_tests: int
    custom_metrics: Dict[str, float] = {}
    
class TestResultsAggregation(BaseModel):
    summaries: List[TestResultsSummary]
    best_performing_model: str
    best_performing_version: str
    comparison_metrics: Dict[str, Dict[str, float]] = {}

# New models for prompt generation

class PromptVariationRequest(BaseModel):
    base_prompt: str
    num_variations: int = 3
    variation_type: str = "rephrase"
    template_params: Optional[Dict[str, Any]] = None
    generator_model: Optional[str] = None

class PromptVariationResponse(BaseModel):
    variations: List[str]
    original_prompt: str
    variation_type: str
    generation_time: float

class BatchGenerateVariationsRequest(BaseModel):
    base_prompts: List[str]
    num_variations_each: int = 3
    variation_types: List[str] = ["rephrase"]
    template_params: Optional[Dict[str, Any]] = None
    generator_model: Optional[str] = None

class BatchGenerateVariationsResponse(BaseModel):
    results: List[PromptVariationResponse]
    total_variations: int
    failed_prompts: List[str] = []

class PromptAnalysisRequest(BaseModel):
    prompt: str
    generator_model: Optional[str] = None

class PromptAnalysisResponse(BaseModel):
    clarity: int
    specificity: int
    potential_issues: List[str]
    improvement_suggestions: List[str]
    template_parameters: List[str]
    raw_response: Optional[str] = None

# New models for simplified prompt generation

class VariationTypeRequest(BaseModel):
    variation_type: str
    instruction: str

class VariationTypeResponse(BaseModel):
    variation_types: Dict[str, str]

class GenerateVariationsRequest(BaseModel):
    prompt_uuid: str
    variation_types: List[str]
    template_params: Optional[Dict[str, Any]] = None
    save: bool = False
    generator_model: Optional[str] = None

class PromptVariation(BaseModel):
    variation_type: str
    content: str
    uuid: Optional[str] = None

class GenerateVariationsResponse(BaseModel):
    prompt_uuid: str
    variations: List[PromptVariation]
    original_prompt: str

class BatchGenerateVariationsRequest(BaseModel):
    prompt_uuids: List[str]
    variation_types: List[str]
    template_params: Optional[Dict[str, Any]] = None
    save: bool = False
    generator_model: Optional[str] = None

class BatchVariationResult(BaseModel):
    prompt_uuid: str
    variations: List[PromptVariation]
    original_prompt: str

class BatchGenerateVariationsResponse(BaseModel):
    results: List[BatchVariationResult]
    total_variations: int

# Revised models for evaluation results

class TestRunEvaluationRequest(BaseModel):
    test_run_id: str
    model: Optional[str] = None

class EvaluationResponse(BaseModel):
    id: str
    question: str
    expected_answer: str
    actual_answer: str
    accuracy: float
    relevance: Optional[float] = None
    completeness: Optional[float] = None
    conciseness: Optional[float] = None
    overall_score: float
    explanation: str
    created_at: datetime
    model: Optional[str] = None
    prompt_version_id: Optional[str] = None
    test_run_id: str

class BatchEvaluationRequest(BaseModel):
    test_run_ids: List[str]
    evaluation_model: Optional[str] = None
    detailed_metrics: bool = False

class BatchEvaluationResponse(BaseModel):
    results: List[EvaluationResponse]
    total_evaluations: int
    average_score: float