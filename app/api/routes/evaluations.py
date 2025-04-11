from fastapi import APIRouter, HTTPException, Depends, Body
from app.core.dependencies import get_ollama_service, get_neo4j_service
from app.services.ollama import OllamaService
from app.services.neo4j import Neo4jService
from app.services.evaluator import LLMEvaluator
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
import json

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# Pydantic models for request/response validation
class EvaluationRequest(BaseModel):
    question: str
    expected_answer: str
    actual_answer: str
    
    model_config = {
        'protected_namespaces': ()
    }

class BatchEvaluationRequest(BaseModel):
    evaluations: List[EvaluationRequest]
    evaluation_model: Optional[str] = "gemma3:4b"
    detailed_metrics: Optional[bool] = False
    
    model_config = {
        'protected_namespaces': ()
    }

class TestRunEvaluationRequest(BaseModel):
    test_run_id: str
    evaluation_model: Optional[str] = "gemma3:4b"
    detailed_metrics: Optional[bool] = False
    
    model_config = {
        'protected_namespaces': ()
    }

class CompareTestRunsRequest(BaseModel):
    test_run_ids: List[str]
    evaluation_model: Optional[str] = "gemma3:4b"
    detailed_metrics: Optional[bool] = False
    
    model_config = {
        'protected_namespaces': ()
    }

@router.post("/accuracy", response_model=Dict[str, Any])
async def evaluate_accuracy(
    request: EvaluationRequest,
    evaluation_model: str = "gemma3:4b",
    detailed_metrics: bool = False,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """
    Evaluate the accuracy of an answer using a larger LLM model.
    
    This endpoint allows semantic evaluation rather than simple string matching.
    """
    try:
        # First, check if the evaluation model is running
        running_models = ollama_service.list_running_models()
        running_model_names = [model.name for model in running_models]
        if evaluation_model not in running_model_names:
            raise HTTPException(status_code=400, detail=f"Evaluation model {evaluation_model} is not running")
        
        # Create the evaluator with the specified model
        evaluator = LLMEvaluator(ollama_service, evaluation_model)
        
        # Run the evaluation
        if detailed_metrics:
            result = await evaluator.evaluate_response_metrics(
                request.question,
                request.expected_answer,
                request.actual_answer
            )
        else:
            result = await evaluator.evaluate_accuracy(
                request.question,
                request.expected_answer,
                request.actual_answer
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating accuracy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=List[Dict[str, Any]])
async def batch_evaluate(
    request: BatchEvaluationRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """
    Batch evaluate multiple question-answer pairs using a larger LLM model.
    """
    try:
        # First, check if the evaluation model is running
        running_models = ollama_service.list_running_models()
        running_model_names = [model.name for model in running_models]
        if request.evaluation_model not in running_model_names:
            raise HTTPException(status_code=400, detail=f"Evaluation model {request.evaluation_model} is not running")
        
        # Create the evaluator with the specified model
        evaluator = LLMEvaluator(ollama_service, request.evaluation_model)
        
        # Process each evaluation
        results = []
        for eval_req in request.evaluations:
            try:
                if request.detailed_metrics:
                    result = await evaluator.evaluate_response_metrics(
                        eval_req.question,
                        eval_req.expected_answer,
                        eval_req.actual_answer
                    )
                else:
                    result = await evaluator.evaluate_accuracy(
                        eval_req.question,
                        eval_req.expected_answer,
                        eval_req.actual_answer
                    )
                
                # Add the question and answer to the result for reference
                result["question"] = eval_req.question
                result["expected_answer"] = eval_req.expected_answer
                result["actual_answer"] = eval_req.actual_answer
                
                results.append(result)
            except Exception as e:
                # Include error in results but continue with other evaluations
                results.append({
                    "question": eval_req.question,
                    "expected_answer": eval_req.expected_answer,
                    "actual_answer": eval_req.actual_answer,
                    "error": str(e)
                })
        
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-run/{test_run_id}", response_model=Dict[str, Any])
async def evaluate_test_run(
    test_run_id: str,
    evaluation_model: str = "gemma3:4b",
    detailed_metrics: bool = False,
    ollama_service: OllamaService = Depends(get_ollama_service),
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Evaluate a specific test run using a larger LLM model.
    
    This retrieves the test run data from Neo4j, evaluates it, and returns the results.
    Optionally, it can update the test run with the evaluation results.
    """
    try:
        # First, check if the evaluation model is running
        running_models = ollama_service.list_running_models()
        running_model_names = [model.name for model in running_models]
        if evaluation_model not in running_model_names:
            raise HTTPException(status_code=400, detail=f"Evaluation model {evaluation_model} is not running")
        
        # Get the test run from Neo4j
        with neo4j_service.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (tr:TestRun {id: $test_run_id})-[:TESTED_WITH]->(pv:PromptVersion)
                MATCH (pv)-[:VERSION_OF]->(p:Prompt)
                RETURN tr.output as output, p.content as question, p.expected_solution as expected_answer
                """,
                test_run_id=test_run_id
            )
            record = result.single()
            if not record:
                raise HTTPException(status_code=404, detail=f"Test run with ID {test_run_id} not found")
            
            # Get data from the record
            output = record["output"]
            question = record["question"]
            expected_answer = record["expected_answer"]
            
            # Check if we have the expected answer
            if not expected_answer:
                raise HTTPException(status_code=400, detail="No expected answer available for this test run")
        
        # Create the evaluator with the specified model
        evaluator = LLMEvaluator(ollama_service, evaluation_model)
        
        # Run the evaluation
        if detailed_metrics:
            evaluation_result = await evaluator.evaluate_response_metrics(
                question,
                expected_answer,
                output
            )
        else:
            evaluation_result = await evaluator.evaluate_accuracy(
                question,
                expected_answer,
                output
            )
        
        # Add the context to the result
        evaluation_result["test_run_id"] = test_run_id
        evaluation_result["question"] = question
        evaluation_result["expected_answer"] = expected_answer
        evaluation_result["actual_answer"] = output
        
        # Update the test run with the evaluation results
        try:
            with neo4j_service.driver.session(database="neo4j") as session:
                # Store evaluation results as a JSON property
                session.run(
                    """
                    MATCH (tr:TestRun {id: $test_run_id})
                    SET tr.evaluation_json = $evaluation_json
                    """,
                    test_run_id=test_run_id,
                    evaluation_json=json.dumps(evaluation_result)
                )
        except Exception as e:
            logger.error(f"Failed to update test run with evaluation results: {str(e)}")
            evaluation_result["update_status"] = "failed"
        else:
            evaluation_result["update_status"] = "success"
        
        return evaluation_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating test run: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-test-runs", response_model=List[Dict[str, Any]])
async def compare_test_runs_with_evaluation(
    request: CompareTestRunsRequest,
    ollama_service: OllamaService = Depends(get_ollama_service),
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Compare multiple test runs with evaluation.
    
    This retrieves the test runs from Neo4j, evaluates them if they have expected answers,
    and returns a comparison with evaluation results.
    """
    try:
        # First, check if the evaluation model is running
        running_models = ollama_service.list_running_models()
        running_model_names = [model.name for model in running_models]
        if request.evaluation_model not in running_model_names:
            raise HTTPException(status_code=400, detail=f"Evaluation model {request.evaluation_model} is not running")
        
        # Create the evaluator
        evaluator = LLMEvaluator(ollama_service, request.evaluation_model)
        
        # Get the test runs from Neo4j
        with neo4j_service.driver.session(database="neo4j") as session:
            result = session.run(
                """
                MATCH (tr:TestRun)
                WHERE tr.id IN $test_run_ids
                MATCH (tr)-[:TESTED_WITH]->(pv:PromptVersion)
                MATCH (pv)-[:VERSION_OF]->(p:Prompt)
                RETURN tr.id as test_run_id, tr.output as output, tr.model_used as model,
                       p.content as question, p.expected_solution as expected_answer,
                       tr.metrics_json as metrics_json, tr.evaluation_json as evaluation_json
                """,
                test_run_ids=request.test_run_ids
            )
            
            test_runs = []
            for record in result:
                test_run = {
                    "test_run_id": record["test_run_id"],
                    "output": record["output"],
                    "model": record["model"],
                    "question": record["question"],
                    "expected_answer": record["expected_answer"],
                    "metrics": json.loads(record["metrics_json"]) if record["metrics_json"] else {},
                }
                
                # If we already have an evaluation, use it
                if record["evaluation_json"]:
                    test_run["evaluation"] = json.loads(record["evaluation_json"])
                    test_run["evaluation_source"] = "cached"
                
                test_runs.append(test_run)
        
        # Evaluate test runs that have expected answers but no evaluations
        for test_run in test_runs:
            if test_run.get("expected_answer") and not test_run.get("evaluation"):
                try:
                    if request.detailed_metrics:
                        evaluation = await evaluator.evaluate_response_metrics(
                            test_run["question"],
                            test_run["expected_answer"],
                            test_run["output"]
                        )
                    else:
                        evaluation = await evaluator.evaluate_accuracy(
                            test_run["question"],
                            test_run["expected_answer"],
                            test_run["output"]
                        )
                    
                    test_run["evaluation"] = evaluation
                    test_run["evaluation_source"] = "new"
                    
                    # Update the test run with the evaluation
                    try:
                        with neo4j_service.driver.session(database="neo4j") as session:
                            session.run(
                                """
                                MATCH (tr:TestRun {id: $test_run_id})
                                SET tr.evaluation_json = $evaluation_json
                                """,
                                test_run_id=test_run["test_run_id"],
                                evaluation_json=json.dumps(evaluation)
                            )
                    except Exception as e:
                        logger.error(f"Failed to update test run with evaluation: {str(e)}")
                except Exception as e:
                    logger.error(f"Error evaluating test run {test_run['test_run_id']}: {str(e)}")
                    test_run["evaluation"] = {"error": str(e)}
                    test_run["evaluation_source"] = "error"
            
            # If there's no expected answer, mark it
            elif not test_run.get("expected_answer"):
                test_run["evaluation"] = {"message": "No expected answer available"}
                test_run["evaluation_source"] = "missing_expected_answer"
        
        return test_runs
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing test runs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/csv", response_model=Dict[str, Any])
async def evaluate_csv_data(
    file_path: str = Body(...),
    evaluation_model: str = Body("gemma3:4b"),
    detailed_metrics: bool = Body(False),
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """
    Evaluate question-answer pairs from a CSV file.
    
    The CSV file should have 'question' and 'answer' columns.
    The function will evaluate actual answers from a model against these expected answers.
    """
    try:
        import csv
        import os
        
        # Check if the file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Check if the evaluation model is running
        running_models = ollama_service.list_running_models()
        running_model_names = [model.name for model in running_models]
        if evaluation_model not in running_model_names:
            raise HTTPException(status_code=400, detail=f"Evaluation model {evaluation_model} is not running")
        
        # Create the evaluator
        evaluator = LLMEvaluator(ollama_service, evaluation_model)
        
        # Read the CSV file
        qa_pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Read header row
            
            # Check if required columns exist
            question_idx = header.index('question') if 'question' in header else None
            answer_idx = header.index('answer') if 'answer' in header else None
            
            if question_idx is None or answer_idx is None:
                raise HTTPException(status_code=400, detail="CSV must have 'question' and 'answer' columns")
            
            # Read rows
            for row in reader:
                if len(row) > max(question_idx, answer_idx):
                    qa_pairs.append({
                        "question": row[question_idx],
                        "expected_answer": row[answer_idx]
                    })
        
        # Process each question with the model to get actual answers
        results = []
        for qa in qa_pairs:
            # Get the actual answer by processing the question with the model
            try:
                actual_answer = ollama_service.process_prompt(qa["question"], evaluation_model)
                
                # Evaluate the answer
                if detailed_metrics:
                    evaluation = await evaluator.evaluate_response_metrics(
                        qa["question"],
                        qa["expected_answer"],
                        actual_answer
                    )
                else:
                    evaluation = await evaluator.evaluate_accuracy(
                        qa["question"],
                        qa["expected_answer"],
                        actual_answer
                    )
                
                # Add context to the result
                result = {
                    "question": qa["question"],
                    "expected_answer": qa["expected_answer"],
                    "actual_answer": actual_answer,
                    **evaluation
                }
                
                results.append(result)
            except Exception as e:
                results.append({
                    "question": qa["question"],
                    "expected_answer": qa["expected_answer"],
                    "error": str(e)
                })
        
        # Calculate summary statistics
        summary = {
            "total_questions": len(qa_pairs),
            "completed_evaluations": len([r for r in results if "error" not in r]),
            "failed_evaluations": len([r for r in results if "error" in r]),
        }
        
        if detailed_metrics:
            # Calculate average scores across all dimensions
            metrics = ["accuracy", "relevance", "completeness", "conciseness", "overall_score"]
            for metric in metrics:
                values = [r.get(metric, 0) for r in results if metric in r]
                summary[f"avg_{metric}"] = sum(values) / len(values) if values else 0
        else:
            # Calculate average score and correct percentage
            scores = [r.get("score", 0) for r in results if "score" in r]
            correct = [r for r in results if r.get("is_correct", False)]
            
            summary["avg_score"] = sum(scores) / len(scores) if scores else 0
            summary["correct_percentage"] = len(correct) / len(results) * 100 if results else 0
        
        return {
            "results": results,
            "summary": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating CSV data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))