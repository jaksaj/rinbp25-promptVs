from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional
from app.services.evaluator import LLMEvaluator
from app.services.neo4j import Neo4jService
from app.core.dependencies import get_ollama_service, get_neo4j_service
from app.api.models.prompt import (
    TestRunEvaluationRequest, 
    EvaluationResponse, 
    BatchEvaluationRequest, 
    BatchEvaluationResponse
)
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation")  # Add the prefix here to match the client calls

@router.post("/test-run", response_model=EvaluationResponse)
async def evaluate_test_run(
    request: TestRunEvaluationRequest,
    evaluation_model: Optional[str] = None,
    ollama_service = Depends(get_ollama_service),
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Evaluate a test run using the linked prompt version and prompt data.
    """
    try:
        # Get the test run details to extract output (actual answer) and prompt version ID
        test_run = neo4j_service.get_test_run(request.test_run_id)
        if not test_run:
            raise HTTPException(status_code=404, detail=f"Test run with ID {request.test_run_id} not found")
        
        # Get the prompt version details using the test run's prompt version ID
        prompt_version_id = test_run.get("prompt_version_id")
        prompt_version = neo4j_service.get_prompt_version(prompt_version_id)
        if not prompt_version:
            raise HTTPException(status_code=404, detail=f"Prompt version with ID {prompt_version_id} not found")
        
        # Get the associated prompt to retrieve the expected solution if not in version
        expected_solution = prompt_version.get("expected_solution")
        if not expected_solution:
            prompt_id = prompt_version.get("prompt_id")
            if prompt_id:
                prompt = neo4j_service.get_prompt(prompt_id)
                expected_solution = prompt.get("expected_solution", "")
        
        # Create the evaluator with the Neo4j service for saving results
        evaluator = LLMEvaluator(
            ollama_service=ollama_service,
            evaluation_model=evaluation_model or "gemma3:4b",
            neo4j_service=neo4j_service
        )
        
        # Evaluate the accuracy of the test run
        result = await evaluator.evaluate_accuracy(
            question=prompt_version.get("content", ""),
            expected_answer=expected_solution or "",
            actual_answer=test_run.get("output", ""),
            prompt_version_id=prompt_version_id,
            test_run_id=request.test_run_id,
            model=request.model or test_run.get("model_used")
        )
        
        # Create a response with all fields
        response = EvaluationResponse(
            id=str(uuid.uuid4()),
            question=prompt_version.get("content", ""),
            expected_answer=expected_solution or "",
            actual_answer=test_run.get("output", ""),
            accuracy=result.get("score", 0.0),
            overall_score=result.get("score", 0.0),
            explanation=result.get("explanation", ""),
            created_at=datetime.now(),
            model=request.model or test_run.get("model_used"),
            prompt_version_id=prompt_version_id,
            test_run_id=request.test_run_id
        )
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in evaluate_test_run: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/metrics/test-run", response_model=EvaluationResponse)
async def evaluate_test_run_metrics(
    request: TestRunEvaluationRequest,
    evaluation_model: Optional[str] = None,
    ollama_service = Depends(get_ollama_service),
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Evaluate multiple metrics of a test run using the linked prompt version and prompt data.
    """
    try:
        # Get the test run details to extract output (actual answer) and prompt version ID
        test_run = neo4j_service.get_test_run(request.test_run_id)
        if not test_run:
            raise HTTPException(status_code=404, detail=f"Test run with ID {request.test_run_id} not found")
        
        # Get the prompt version details using the test run's prompt version ID
        prompt_version_id = test_run.get("prompt_version_id")
        prompt_version = neo4j_service.get_prompt_version(prompt_version_id)
        if not prompt_version:
            raise HTTPException(status_code=404, detail=f"Prompt version with ID {prompt_version_id} not found")
        
        # Get the associated prompt to retrieve the expected solution if not in version
        expected_solution = prompt_version.get("expected_solution")
        if not expected_solution:
            prompt_id = prompt_version.get("prompt_id")
            if prompt_id:
                prompt = neo4j_service.get_prompt(prompt_id)
                expected_solution = prompt.get("expected_solution", "")
        
        # Create the evaluator with the Neo4j service for saving results
        evaluator = LLMEvaluator(
            ollama_service=ollama_service,
            evaluation_model=evaluation_model or "gemma3:4b",
            neo4j_service=neo4j_service
        )
        
        # Evaluate detailed metrics
        result = await evaluator.evaluate_response_metrics(
            question=prompt_version.get("content", ""),
            expected_answer=expected_solution or "",
            actual_answer=test_run.get("output", ""),
            prompt_version_id=prompt_version_id,
            test_run_id=request.test_run_id,
            model=request.model or test_run.get("model_used")
        )
        
        # Create a response with all fields
        response = EvaluationResponse(
            id=str(uuid.uuid4()),
            question=prompt_version.get("content", ""),
            expected_answer=expected_solution or "",
            actual_answer=test_run.get("output", ""),
            accuracy=result.get("accuracy", 0.0),
            relevance=result.get("relevance"),
            completeness=result.get("completeness"),
            conciseness=result.get("conciseness"),
            overall_score=result.get("overall_score", 0.0),
            explanation=result.get("explanation", ""),
            created_at=datetime.now(),
            model=request.model or test_run.get("model_used"),
            prompt_version_id=prompt_version_id,
            test_run_id=request.test_run_id
        )
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in evaluate_test_run_metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=BatchEvaluationResponse)
async def evaluate_batch(
    request: BatchEvaluationRequest,
    ollama_service = Depends(get_ollama_service),
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Evaluate a batch of test runs using their linked prompt versions and prompt data.
    """
    try:
        # Create evaluator with the Neo4j service
        evaluator = LLMEvaluator(
            ollama_service=ollama_service,
            evaluation_model=request.evaluation_model or "gemma3:4b",
            neo4j_service=neo4j_service
        )
        
        # Process each test run ID
        eval_dicts = []
        for test_run_id in request.test_run_ids:
            try:
                # Get the test run details
                test_run = neo4j_service.get_test_run(test_run_id)
                if not test_run:
                    logger.warning(f"Test run with ID {test_run_id} not found. Skipping.")
                    continue
                
                # Get the prompt version details
                prompt_version_id = test_run.get("prompt_version_id")
                prompt_version = neo4j_service.get_prompt_version(prompt_version_id)
                if not prompt_version:
                    logger.warning(f"Prompt version with ID {prompt_version_id} not found. Skipping.")
                    continue
                
                # Get the associated prompt to retrieve the expected solution if not in version
                expected_solution = prompt_version.get("expected_solution")
                if not expected_solution:
                    prompt_id = prompt_version.get("prompt_id")
                    if prompt_id:
                        prompt = neo4j_service.get_prompt(prompt_id)
                        expected_solution = prompt.get("expected_solution", "")
                
                # Add the evaluation item with question and expected answer
                eval_dicts.append({
                    "question": prompt_version.get("content", ""),
                    "expected_answer": expected_solution or "",
                    "actual_answer": test_run.get("output", ""),
                    "prompt_version_id": prompt_version_id,
                    "test_run_id": test_run_id,
                    "model": test_run.get("model_used")
                })
            except Exception as e:
                logger.error(f"Error processing test run {test_run_id}: {str(e)}")
                continue
        
        # Evaluate the batch
        results = await evaluator.evaluate_batch(eval_dicts, request.detailed_metrics)
        
        # Convert results to response models
        responses = []
        total_score = 0.0
        
        for result in results:
            # Create evaluation response
            response = EvaluationResponse(
                id=str(uuid.uuid4()),
                question=result.get("question", ""),
                expected_answer=result.get("expected_answer", ""),
                actual_answer=result.get("actual_answer", ""),
                accuracy=result.get("accuracy", 0.0),
                relevance=result.get("relevance"),
                completeness=result.get("completeness"),
                conciseness=result.get("conciseness"),
                overall_score=result.get("overall_score", 0.0),
                explanation=result.get("explanation", ""),
                created_at=datetime.now(),
                model=result.get("model"),
                prompt_version_id=result.get("prompt_version_id"),
                test_run_id=result.get("test_run_id")
            )
            responses.append(response)
            total_score += response.overall_score
        
        # Calculate average score
        average_score = total_score / len(responses) if responses else 0.0
        
        return BatchEvaluationResponse(
            results=responses,
            total_evaluations=len(responses),
            average_score=average_score
        )
    except Exception as e:
        logger.error(f"Error in evaluate_batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
@router.get("/test-run/{test_run_id}", response_model=List[EvaluationResponse])
async def get_evaluations_for_test_run(
    test_run_id: str,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Get all evaluations for a specific test run.
    """
    try:
        # Check if the test run exists
        test_run = neo4j_service.get_test_run(test_run_id)
        if not test_run:
            raise HTTPException(status_code=404, detail=f"Test run with ID {test_run_id} not found")
            
        # Retrieve evaluations from Neo4j
        evaluations = neo4j_service.get_evaluations_for_test_run(test_run_id)
        
        # Convert to response models
        responses = []
        for eval_dict in evaluations:
            responses.append(EvaluationResponse(
                id=eval_dict.get("id", ""),
                question=eval_dict.get("question", ""),
                expected_answer=eval_dict.get("expected_answer", ""),
                actual_answer=eval_dict.get("actual_answer", ""),
                accuracy=eval_dict.get("accuracy", 0.0),
                relevance=eval_dict.get("relevance"),
                completeness=eval_dict.get("completeness"),
                conciseness=eval_dict.get("conciseness"),
                overall_score=eval_dict.get("overall_score", 0.0),
                explanation=eval_dict.get("explanation", ""),
                created_at=eval_dict.get("created_at", datetime.now()),
                model=eval_dict.get("model"),
                prompt_version_id=eval_dict.get("prompt_version_id"),
                test_run_id=test_run_id
            ))
        
        return responses
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving evaluations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))