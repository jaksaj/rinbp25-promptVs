from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from app.api.models.prompt import (
    BatchQuestionImport, BatchVersionCreate, BatchTestRequest, BatchTestResponse,
    TestResultsAggregation, TestResultsSummary, QuestionAnswer, PromptVersionTemplate,
    PromptCreate, PromptVersionCreate, TestRunCreate, TestRunMetrics
)
from app.core.dependencies import get_neo4j_service, get_ollama_service
from app.services.neo4j import Neo4jService
from app.services.ollama import OllamaService
from app.core.config import NEO4J_DATABASE
import asyncio
import time
import logging
import uuid
from typing import Dict, List, Any, Optional
import json
import statistics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/batch", tags=["batch"])

@router.post("/import-questions", response_model=Dict[str, Any])
async def import_questions(
    request: BatchQuestionImport,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Import a batch of questions and answers, creating a new prompt group and prompts.
    """
    try:
        # Create a prompt group for this batch
        group_id = neo4j_service.create_prompt_group(
            name=request.prompt_group_name,
            description=request.prompt_group_description,
            tags=request.tags
        )
        
        # Create individual prompts for each question
        prompt_ids = []
        for idx, qa in enumerate(request.questions):
            question_number = idx + 1
            prompt = PromptCreate(
                prompt_group_id=group_id,
                content=qa.question,
                name=f"Q{question_number}: {qa.question[:50]}{'...' if len(qa.question) > 50 else ''}",
                description=f"Question {question_number}",
                expected_solution=qa.answer,
                tags=request.tags
            )
            prompt_id = neo4j_service.create_prompt(prompt)
            prompt_ids.append(prompt_id)
            
        return {
            "group_id": group_id,
            "prompt_ids": prompt_ids,
            "total_imported": len(prompt_ids)
        }
    except Exception as e:
        logger.error(f"Error importing questions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-versions", response_model=Dict[str, Any])
async def create_versions(
    request: BatchVersionCreate,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Create multiple versions for multiple prompts based on templates.
    Templates can include {question} placeholder which will be replaced with the prompt content.
    """
    try:
        results = {
            "created_versions": [],
            "failed_versions": [],
            "total_created": 0,
            "total_failed": 0
        }
        
        # For each prompt
        for prompt_id in request.prompt_ids:
            try:
                # Get the prompt content to use in templates
                prompt = neo4j_service.get_prompt(prompt_id)
                if not prompt:
                    results["failed_versions"].append({
                        "prompt_id": prompt_id,
                        "error": "Prompt not found"
                    })
                    continue
                
                question = prompt["content"]
                expected_solution = prompt.get("expected_solution")
                
                # Create each version using the templates
                for template in request.templates:
                    try:
                        # Replace placeholder with actual question
                        content = template.template.replace("{question}", question)
                        
                        # Create the version
                        version = PromptVersionCreate(
                            prompt_id=prompt_id,
                            content=content,
                            version=template.version_name,
                            expected_solution=expected_solution,
                            notes=template.notes
                        )
                        
                        version_id = neo4j_service.create_prompt_version(version)
                        results["created_versions"].append({
                            "prompt_id": prompt_id,
                            "version_id": version_id,
                            "version": template.version_name
                        })
                        results["total_created"] += 1
                    except Exception as ve:
                        results["failed_versions"].append({
                            "prompt_id": prompt_id,
                            "template": template.version_name,
                            "error": str(ve)
                        })
                        results["total_failed"] += 1
            except Exception as pe:
                results["failed_versions"].append({
                    "prompt_id": prompt_id,
                    "error": str(pe)
                })
                results["total_failed"] += 1
                
        return results
    except Exception as e:
        logger.error(f"Error creating versions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run-tests", response_model=BatchTestResponse)
async def run_tests(
    request: BatchTestRequest,
    background_tasks: BackgroundTasks,
    neo4j_service: Neo4jService = Depends(get_neo4j_service),
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """
    Run batch tests for multiple prompt versions with multiple models.
    This endpoint will start the process and return immediately.
    The tests will continue running in the background.
    """
    try:
        # Validate inputs
        if not request.prompt_version_ids:
            raise HTTPException(status_code=400, detail="No prompt versions specified")
            
        if not request.models:
            raise HTTPException(status_code=400, detail="No models specified")
            
        # Check if models are running
        running_models = ollama_service.list_running_models()
        running_model_names = [model.name for model in running_models]
        
        missing_models = [model for model in request.models if model not in running_model_names]
        if missing_models:
            raise HTTPException(
                status_code=400, 
                detail=f"The following models are not running: {', '.join(missing_models)}"
            )
        
        # Create batch ID
        batch_id = str(uuid.uuid4())
        
        # Start the background task
        background_tasks.add_task(
            run_batch_tests,
            batch_id,
            request.prompt_version_ids,
            request.models,
            neo4j_service,
            ollama_service
        )
        
        total_tests = len(request.prompt_version_ids) * len(request.models)
        
        # Return the batch ID in both the response and the headers
        response = BatchTestResponse(
            total_tests=total_tests,
            completed=0,
            failed=0,
            test_run_ids=[],
            batch_id=batch_id  # Include batch ID in response
        )
        
        # Add batch ID to headers so client can extract it
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content=response.dict(),
            headers={"X-Batch-ID": batch_id, "Location": f"/api/batch/test-status/{batch_id}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting batch tests: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_batch_tests(
    batch_id: str,
    prompt_version_ids: List[str],
    models: List[str],
    neo4j_service: Neo4jService,
    ollama_service: OllamaService
):
    """
    Background task to run batch tests.
    This will iterate through all prompt versions and models, run tests and save results.
    """
    logger.info(f"Starting batch test run {batch_id} with {len(prompt_version_ids)} prompts and {len(models)} models")
    
    test_run_ids = []
    completed = 0
    failed = 0
    
    # For each prompt version
    for version_id in prompt_version_ids:
        try:
            # Get the prompt version
            version = neo4j_service.get_prompt_version(version_id)
            if not version:
                logger.error(f"Prompt version {version_id} not found")
                failed += len(models)
                continue
                
            content = version["content"]
            
            # Run for each model
            for model_name in models:
                try:
                    # Process the prompt
                    start_time = time.time()
                    result = ollama_service.process_prompt(content, model_name)
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
                    
                    # Create input params
                    input_params = {
                        "batch_id": batch_id
                    }
                    
                    # Create test run
                    test_run = TestRunCreate(
                        prompt_version_id=version_id,
                        model_used=model_name,
                        output=result,
                        metrics=metrics,
                        input_params=input_params
                    )
                    
                    # Save the test run
                    test_run_id = neo4j_service.create_test_run(test_run)
                    test_run_ids.append(test_run_id)
                    completed += 1
                    
                    logger.info(f"Completed test for version {version_id} with model {model_name}")
                except Exception as me:
                    logger.error(f"Error running test for version {version_id} with model {model_name}: {str(me)}")
                    failed += 1
        except Exception as ve:
            logger.error(f"Error processing version {version_id}: {str(ve)}")
            failed += len(models)
    
    logger.info(f"Batch test run {batch_id} completed: {completed} successful, {failed} failed")
    
@router.get("/test-status/{batch_id}", response_model=BatchTestResponse)
async def get_test_status(
    batch_id: str,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Check the status of a batch test run.
    """
    try:
        # Query Neo4j for test runs with this batch ID
        with neo4j_service.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (tr:TestRun)
                WHERE tr.input_params_json CONTAINS $batch_id
                RETURN tr.id as test_run_id
                """,
                batch_id=batch_id
            )
            
            test_run_ids = [record["test_run_id"] for record in result]
            
            return BatchTestResponse(
                total_tests=0,  # We don't know the total from here
                completed=len(test_run_ids),
                failed=0,  # We don't track failures in Neo4j
                test_run_ids=test_run_ids
            )
    except Exception as e:
        logger.error(f"Error getting batch test status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{batch_id}", response_model=TestResultsAggregation)
async def get_test_results(
    batch_id: str,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Get aggregated results for a batch test run.
    """
    try:
        # Query Neo4j for test runs with this batch ID
        with neo4j_service.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (tr:TestRun)-[:TESTED_WITH]->(pv:PromptVersion)
                WHERE tr.input_params_json CONTAINS $batch_id
                RETURN 
                    tr.model_used as model,
                    pv.version as version,
                    tr.metrics_json as metrics_json
                """,
                batch_id=batch_id
            )
            
            # Group by model and version
            data = {}
            for record in result:
                model = record["model"]
                version = record["version"]
                metrics = json.loads(record["metrics_json"])
                
                if model not in data:
                    data[model] = {}
                    
                if version not in data[model]:
                    data[model][version] = {
                        "latencies": [],
                        "token_counts": [],
                        "tokens_per_second": [],
                        "count": 0
                    }
                
                # Add metrics
                data[model][version]["latencies"].append(metrics["latency_ms"])
                data[model][version]["token_counts"].append(metrics["token_count"])
                
                if metrics.get("token_per_second"):
                    data[model][version]["tokens_per_second"].append(metrics["token_per_second"])
                    
                data[model][version]["count"] += 1
            
            # Calculate aggregated metrics
            summaries = []
            comparison_metrics = {"latency_ms": {}, "token_count": {}, "tokens_per_second": {}}
            
            for model, versions in data.items():
                for version, metrics in versions.items():
                    avg_latency = statistics.mean(metrics["latencies"]) if metrics["latencies"] else 0
                    avg_tokens = statistics.mean(metrics["token_counts"]) if metrics["token_counts"] else 0
                    avg_tps = statistics.mean(metrics["tokens_per_second"]) if metrics["tokens_per_second"] else 0
                    
                    # Create summary
                    summary = TestResultsSummary(
                        model=model,
                        prompt_version=version,
                        avg_latency_ms=avg_latency,
                        avg_token_count=avg_tokens,
                        avg_tokens_per_second=avg_tps,
                        total_tests=metrics["count"]
                    )
                    summaries.append(summary)
                    
                    # Add to comparison metrics
                    key = f"{model}/{version}"
                    comparison_metrics["latency_ms"][key] = avg_latency
                    comparison_metrics["token_count"][key] = avg_tokens
                    comparison_metrics["tokens_per_second"][key] = avg_tps
            
            # Find the best performing model/version based on latency
            best_model_version = min(
                [(model, version) for model, versions in data.items() for version in versions],
                key=lambda x: statistics.mean(data[x[0]][x[1]]["latencies"]) if data[x[0]][x[1]]["latencies"] else float('inf')
            )
            
            return TestResultsAggregation(
                summaries=summaries,
                best_performing_model=best_model_version[0],
                best_performing_version=best_model_version[1],
                comparison_metrics=comparison_metrics
            )
    except Exception as e:
        logger.error(f"Error getting batch test results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export-results/{batch_id}")
async def export_test_results(
    batch_id: str,
    format: str = Query("json", regex="^(json|csv)$"),
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Export the results of a batch test run in JSON or CSV format.
    """
    try:
        # Query Neo4j for test runs with this batch ID
        with neo4j_service.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (tr:TestRun)-[:TESTED_WITH]->(pv:PromptVersion)-[:VERSION_OF]->(p:Prompt)
                WHERE tr.input_params_json CONTAINS $batch_id
                RETURN 
                    tr.id as run_id,
                    tr.model_used as model,
                    pv.id as version_id, 
                    pv.version as version,
                    p.id as prompt_id,
                    p.content as question,
                    p.expected_solution as expected_answer,
                    tr.output as response,
                    tr.metrics_json as metrics_json,
                    tr.created_at as timestamp
                """,
                batch_id=batch_id
            )
            
            data = []
            for record in result:
                metrics = json.loads(record["metrics_json"])
                entry = {
                    "run_id": record["run_id"],
                    "model": record["model"],
                    "version": record["version"],
                    "question": record["question"],
                    "expected_answer": record["expected_answer"],
                    "response": record["response"],
                    "latency_ms": metrics.get("latency_ms"),
                    "token_count": metrics.get("token_count"),
                    "tokens_per_second": metrics.get("token_per_second"),
                    "timestamp": record["timestamp"].iso_format()
                }
                data.append(entry)
            
            if format == "json":
                return data
            elif format == "csv":
                import csv
                from fastapi.responses import StreamingResponse
                import io
                
                # Create CSV in memory
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=data[0].keys() if data else [])
                writer.writeheader()
                for row in data:
                    writer.writerow(row)
                    
                # Return as downloadable file
                output.seek(0)
                headers = {
                    'Content-Disposition': f'attachment; filename=batch_results_{batch_id}.csv'
                }
                return StreamingResponse(output, media_type="text/csv", headers=headers)
    except Exception as e:
        logger.error(f"Error exporting batch test results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-runs/bulk")
async def get_bulk_test_run_metadata(
    request: Dict[str, List[str]],  # expects {"test_run_ids": [...]}
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Get metadata for multiple test runs in bulk."""
    try:
        results = []
        for test_run_id in request.get("test_run_ids", []):
            test_run = neo4j_service.get_test_run(test_run_id)
            if test_run:
                results.append(test_run)
        return {"results": results, "total": len(results)}
    except Exception as e:
        logger.error(f"Error in get_bulk_test_run_metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))