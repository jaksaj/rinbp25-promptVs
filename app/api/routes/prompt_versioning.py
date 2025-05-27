from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any
from app.api.models.prompt import (
    PromptGroupCreate, 
    PromptGroupResponse,
    PromptCreate, 
    PromptResponse,
    PromptVersionCreate,
    PromptVersionResponse,
    TestRunCreate,
    TestRunResponse,
    TestRunMetrics,
    PromptComparisonRequest,
    TestRunComparisonResponse,
    BulkPromptVersionMetadataRequest,
    BulkPromptVersionMetadataResponse,
    BatchPromptCreate,
    BatchPromptCreateResponse,
    BatchApplyTechniquesRequest,
    BatchApplyTechniquesResponse,
    BatchApplyTechniquesStatusResponse
)
from app.core.dependencies import get_neo4j_service, get_ollama_service
from app.services.neo4j import Neo4jService
from app.services.ollama import OllamaService
import logging
import time
from datetime import datetime
from uuid import uuid4
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory job store for demo (replace with persistent store for production)
batch_jobs = {}

# PromptGroup endpoints
@router.post("/prompt-groups", response_model=Dict[str, str])
async def create_prompt_group(
    request: PromptGroupCreate,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Create a new prompt group."""
    try:
        group_id = neo4j_service.create_prompt_group(
            name=request.name,
            description=request.description,
            tags=request.tags
        )
        return {"id": group_id}
    except Exception as e:
        logger.error(f"Error creating prompt group: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prompt-groups", response_model=List[Dict[str, Any]])
async def get_prompt_groups(
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Get all prompt groups."""
    try:
        groups = neo4j_service.get_prompt_groups()
        return groups
    except Exception as e:
        logger.error(f"Error getting prompt groups: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prompt-groups/{group_id}", response_model=Dict[str, Any])
async def get_prompt_group(
    group_id: str,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Get a specific prompt group."""
    try:
        group = neo4j_service.get_prompt_group(group_id)
        if not group:
            raise HTTPException(status_code=404, detail=f"Prompt group with ID {group_id} not found")
        return group
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt group: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Prompt endpoints
@router.post("/prompts", response_model=Dict[str, str])
async def create_prompt(
    request: PromptCreate,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Create a new prompt within a prompt group."""
    try:
        prompt_id = neo4j_service.create_prompt(request)
        return {"id": prompt_id}
    except Exception as e:
        logger.error(f"Error creating prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/prompts/batch", response_model=BatchPromptCreateResponse)
async def create_prompts_batch(
    request: BatchPromptCreate,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Create multiple prompts in a batch."""
    try:
        ids = neo4j_service.create_prompts_batch(request.prompts)
        return BatchPromptCreateResponse(ids=ids)
    except Exception as e:
        logger.error(f"Error creating prompts batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prompt-groups/{group_id}/prompts", response_model=List[Dict[str, Any]])
async def get_prompts_by_group(
    group_id: str,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Get all prompts in a specific prompt group."""
    try:
        prompts = neo4j_service.get_prompts_by_group(group_id)
        return prompts
    except Exception as e:
        logger.error(f"Error getting prompts by group: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prompts/{prompt_id}", response_model=Dict[str, Any])
async def get_prompt(
    prompt_id: str,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Get a specific prompt."""
    try:
        prompt = neo4j_service.get_prompt(prompt_id)
        if not prompt:
            raise HTTPException(status_code=404, detail=f"Prompt with ID {prompt_id} not found")
        return prompt
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# PromptVersion endpoints
@router.post("/prompt-versions", response_model=Dict[str, str])
async def create_prompt_version(
    request: PromptVersionCreate,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Create a new prompt version."""
    try:
        version_id = neo4j_service.create_prompt_version(request)
        return {"id": version_id}
    except Exception as e:
        logger.error(f"Error creating prompt version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prompts/{prompt_id}/versions", response_model=List[Dict[str, Any]])
async def get_prompt_versions(
    prompt_id: str,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Get all versions of a specific prompt."""
    try:
        versions = neo4j_service.get_prompt_versions(prompt_id)
        return versions
    except Exception as e:
        logger.error(f"Error getting prompt versions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prompt-versions/{version_id}", response_model=Dict[str, Any])
async def get_prompt_version(
    version_id: str,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Get a specific prompt version."""
    try:
        version = neo4j_service.get_prompt_version(version_id)
        if not version:
            raise HTTPException(status_code=404, detail=f"Prompt version with ID {version_id} not found")
        return version
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/prompt-versions/bulk", response_model=BulkPromptVersionMetadataResponse)
async def get_bulk_prompt_version_metadata(
    request: BulkPromptVersionMetadataRequest,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Get metadata for multiple prompt versions in bulk."""
    try:
        results = []
        for version_id in request.version_ids:
            version = neo4j_service.get_prompt_version(version_id)
            if version:
                results.append(version)
        return BulkPromptVersionMetadataResponse(results=results, total=len(results))
    except Exception as e:
        logger.error(f"Error in get_bulk_prompt_version_metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# TestRun endpoints
@router.post("/test-runs", response_model=Dict[str, str])
async def create_test_run(
    request: TestRunCreate,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Create a new test run for a prompt version."""
    try:
        test_run_id = neo4j_service.create_test_run(request)
        return {"id": test_run_id}
    except Exception as e:
        logger.error(f"Error creating test run: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test-runs/{test_run_id}", response_model=Dict[str, Any])
async def get_test_run(
    test_run_id: str,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Get a specific test run by ID, including prompt_version_id."""
    try:
        test_run = neo4j_service.get_test_run(test_run_id)
        if not test_run:
            raise HTTPException(status_code=404, detail=f"Test run with ID {test_run_id} not found")
        # Add prompt_version_id if not already present
        if "prompt_version_id" not in test_run or not test_run["prompt_version_id"]:
            prompt_version_id = neo4j_service.get_prompt_version_id_for_test_run(test_run_id)
            test_run["prompt_version_id"] = prompt_version_id
        return test_run
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting test run: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prompt-versions/{version_id}/test-runs", response_model=List[Dict[str, Any]])
async def get_test_runs(
    version_id: str,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Get all test runs for a specific prompt version."""
    try:
        test_runs = neo4j_service.get_test_runs(version_id)
        return test_runs
    except Exception as e:
        logger.error(f"Error getting test runs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prompt-versions/{version_id}/lineage", response_model=List[Dict[str, Any]])
async def get_prompt_lineage(
    version_id: str,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Get the lineage of a prompt version (its ancestry)."""
    try:
        lineage = neo4j_service.get_prompt_lineage(version_id)
        return lineage
    except Exception as e:
        logger.error(f"Error getting prompt lineage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-test-runs", response_model=List[Dict[str, Any]])
async def compare_test_runs(
    request: PromptComparisonRequest,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Compare multiple test runs side-by-side."""
    try:
        comparison = neo4j_service.compare_test_runs(request.test_run_ids)
        return comparison
    except Exception as e:
        logger.error(f"Error comparing test runs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search-test-runs", response_model=List[Dict[str, Any]])
async def search_test_runs(
    query: str,
    model: str = None,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Search for test runs containing specific text in their output."""
    try:
        results = neo4j_service.search_test_runs(query, model)
        return results
    except Exception as e:
        logger.error(f"Error searching test runs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-prompt-and-save", response_model=Dict[str, str])
async def test_prompt_and_save(
    version_id: str,
    model_name: str,
    background_tasks: BackgroundTasks,
    neo4j_service: Neo4jService = Depends(get_neo4j_service),
    ollama_service: OllamaService = Depends(get_ollama_service),
    custom_prompt: str = None
):
    """Test a prompt version with a model and save the results.
    Uses the prompt content from the specified version_id.
    Optionally, a custom_prompt can be provided to override the stored prompt."""
    try:
        # First, check if the model is running
        running_models = ollama_service.list_running_models()
        running_model_names = [model.name for model in running_models]
        if model_name not in running_model_names:
            raise HTTPException(status_code=400, detail=f"Model {model_name} is not running")

        # Get the prompt version content
        prompt_version = neo4j_service.get_prompt_version(version_id)
        if not prompt_version:
            raise HTTPException(status_code=404, detail=f"Prompt version with ID {version_id} not found")

        # Use the provided custom_prompt if available, otherwise use the stored prompt content
        prompt_content = custom_prompt if custom_prompt is not None else prompt_version["content"]
        
        # Process the prompt
        start_time = time.time()
        result = ollama_service.process_prompt(prompt_content, model_name)
        end_time = time.time()
        
        # Create metrics
        latency_ms = int((end_time - start_time) * 1000)
        token_count = len(result.split())  # Simple approximation
        token_per_second = token_count / ((end_time - start_time) or 1)
        
        metrics = TestRunMetrics(
            latency_ms=latency_ms,
            token_count=token_count,
            token_per_second=token_per_second
        )
        
        # Create input params object to track if custom prompt was used
        input_params = {}
        if custom_prompt is not None:
            input_params["custom_prompt"] = custom_prompt
            input_params["used_custom_prompt"] = True
        else:
            input_params["used_stored_prompt"] = True
        
        # Create test run
        test_run = TestRunCreate(
            prompt_version_id=version_id,
            model_used=model_name,
            output=result,
            metrics=metrics,
            input_params=input_params
        )
        
        # Save the test run and get the ID immediately instead of using a background task
        test_run_id = neo4j_service.create_test_run(test_run)
        
        return {"result": result, "run_id": test_run_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing and saving prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch apply techniques endpoints
@router.post("/techniques/batch-apply", response_model=BatchApplyTechniquesResponse)
async def batch_apply_techniques(
    request: BatchApplyTechniquesRequest,
    background_tasks: BackgroundTasks,
    neo4j_service: Neo4jService = Depends(get_neo4j_service),
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Apply techniques to multiple prompts in the background."""
    job_id = str(uuid4())
    submitted_at = datetime.utcnow()
    batch_jobs[job_id] = {
        "status": "pending",
        "submitted_at": submitted_at,
        "results": None,
        "error": None,
        "completed_at": None
    }
    background_tasks.add_task(_run_batch_apply_techniques, job_id, request, neo4j_service, ollama_service)
    return BatchApplyTechniquesResponse(job_id=job_id, status="pending", submitted_at=submitted_at)

def _run_batch_apply_techniques(job_id, request, neo4j_service, ollama_service):
    try:
        from app.services.prompt_generator import PromptGenerator
        from app.api.models.prompt import PromptVersionCreate
        results = {}
        generator_model = request.generator_model or "llama3.1:8b"
        prompt_generator = PromptGenerator(ollama_service, generator_model)
        for prompt_id in request.prompt_ids:
            results[prompt_id] = {}
            prompt = neo4j_service.get_prompt(prompt_id)
            if not prompt:
                for technique in request.techniques:
                    results[prompt_id][technique] = None
                continue
            for technique in request.techniques:
                try:
                    # Properly call async apply_technique
                    transformed_prompt = asyncio.run(prompt_generator.apply_technique(
                        prompt=prompt["content"],
                        technique=technique
                    ))
                    version = PromptVersionCreate(
                        prompt_id=prompt_id,
                        content=transformed_prompt,
                        version=f"{technique}_technique",
                        notes=f"Auto-generated using {technique}"
                    )
                    version_id = neo4j_service.create_prompt_version(version)
                    results[prompt_id][technique] = version_id
                except Exception as e:
                    results[prompt_id][technique] = None
                    logger.error(f"Batch version creation failed for prompt {prompt_id}, technique {technique}: {e}")
        batch_jobs[job_id]["status"] = "completed"
        batch_jobs[job_id]["results"] = results
        batch_jobs[job_id]["completed_at"] = datetime.utcnow()
    except Exception as e:
        batch_jobs[job_id]["status"] = "failed"
        batch_jobs[job_id]["error"] = str(e)
        batch_jobs[job_id]["completed_at"] = datetime.utcnow()

@router.get("/techniques/batch-status/{job_id}", response_model=BatchApplyTechniquesStatusResponse)
async def get_batch_apply_techniques_status(job_id: str):
    job = batch_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return BatchApplyTechniquesStatusResponse(
        job_id=job_id,
        status=job["status"],
        submitted_at=job["submitted_at"],
        completed_at=job["completed_at"],
        results=job["results"],
        error=job["error"]
    )

# Batch test-prompt-and-save endpoint
@router.post("/test-prompt-and-save/batch", response_model=Dict[str, Any])
async def test_prompt_and_save_batch(
    requests: List[Dict[str, str]],  # Each dict: {"version_id": ..., "model_name": ...}
    background_tasks: BackgroundTasks,
    neo4j_service: Neo4jService = Depends(get_neo4j_service),
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Batch test-prompt-and-save for multiple version/model pairs."""
    job_id = str(uuid4())
    submitted_at = datetime.utcnow()
    batch_jobs[job_id] = {
        "status": "pending",
        "submitted_at": submitted_at,
        "results": None,
        "error": None,
        "completed_at": None
    }
    background_tasks.add_task(_run_batch_test_prompt_and_save, job_id, requests, neo4j_service, ollama_service)
    return {"job_id": job_id, "status": "pending", "submitted_at": submitted_at}


def _run_batch_test_prompt_and_save(job_id, requests, neo4j_service, ollama_service):
    import time
    results = []
    try:
        for req in requests:
            version_id = req.get("version_id")
            model_name = req.get("model_name")
            try:
                # Check if model is running
                running_models = ollama_service.list_running_models()
                running_model_names = [model.name for model in running_models]
                if model_name not in running_model_names:
                    results.append({
                        "version_id": version_id,
                        "model_name": model_name,
                        "error": f"Model {model_name} is not running"
                    })
                    continue
                # Get prompt version
                prompt_version = neo4j_service.get_prompt_version(version_id)
                if not prompt_version:
                    results.append({
                        "version_id": version_id,
                        "model_name": model_name,
                        "error": f"Prompt version with ID {version_id} not found"
                    })
                    continue
                prompt_content = prompt_version["content"]
                start_time = time.time()
                result = ollama_service.process_prompt(prompt_content, model_name)
                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)
                token_count = len(result.split())
                token_per_second = token_count / ((end_time - start_time) or 1)
                metrics = TestRunMetrics(
                    latency_ms=latency_ms,
                    token_count=token_count,
                    token_per_second=token_per_second
                )
                test_run = TestRunCreate(
                    prompt_version_id=version_id,
                    model_used=model_name,
                    output=result,
                    metrics=metrics,
                    input_params={"used_stored_prompt": True}
                )
                test_run_id = neo4j_service.create_test_run(test_run)
                results.append({
                    "version_id": version_id,
                    "model_name": model_name,
                    "run_id": test_run_id,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "version_id": version_id,
                    "model_name": model_name,
                    "error": str(e)
                })
        batch_jobs[job_id]["status"] = "completed"
        batch_jobs[job_id]["results"] = results
        batch_jobs[job_id]["completed_at"] = datetime.utcnow()
    except Exception as e:
        batch_jobs[job_id]["status"] = "failed"
        batch_jobs[job_id]["error"] = str(e)
        batch_jobs[job_id]["completed_at"] = datetime.utcnow()