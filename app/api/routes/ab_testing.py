from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional
from app.services.ab_evaluator import ABTestingEvaluator
from app.services.neo4j import Neo4jService
from app.core.dependencies import get_ollama_service, get_neo4j_service, get_ab_evaluator
from app.api.models.prompt import (
    ComparisonRequest,
    ComparisonResult,
    BatchComparisonRequest,
    BatchComparisonResponse,
    EloRatingResult,
    BulkEloRatingsRequest,
    BulkEloRatingsResponse
)
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ab-testing")

@router.post("/compare", response_model=ComparisonResult)
async def compare_test_runs(
    request: ComparisonRequest,
    evaluation_model: Optional[str] = None,
    ab_evaluator: ABTestingEvaluator = Depends(get_ab_evaluator)
):
    """
    Compare two test runs to determine which one is better.
    """
    try:
        # Use the injected ab_evaluator
        result = await ab_evaluator.compare_test_runs(
            test_run_id1=request.test_run_id1,
            test_run_id2=request.test_run_id2,
            compare_within_version=request.compare_within_version
        )
        
        # Return response
        return ComparisonResult(
            id=result.get("id", str(uuid.uuid4())),
            test_run_id1=result.get("test_run_id1"),
            test_run_id2=result.get("test_run_id2"),
            winner_test_run_id=result.get("winner_test_run_id"),
            explanation=result.get("explanation", ""),
            created_at=datetime.now(),
            compare_within_version=result.get("compare_within_version", False)
        )
    except Exception as e:
        logger.error(f"Error in compare_test_runs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-compare", response_model=BatchComparisonResponse)
async def batch_compare_test_runs(
    request: BatchComparisonRequest,
    evaluation_model: Optional[str] = None,
    ab_evaluator: ABTestingEvaluator = Depends(get_ab_evaluator)
):
    """
    Compare multiple pairs of test runs in batch.
    """
    try:
        # Use the injected ab_evaluator
        results = await ab_evaluator.batch_compare_test_runs(
            test_run_pairs=request.test_run_pairs,
            compare_within_version=request.compare_within_version
        )
        
        # Format the results
        comparison_results = []
        for result in results:
            comparison_results.append(ComparisonResult(
                id=result.get("id", str(uuid.uuid4())),
                test_run_id1=result.get("test_run_id1"),
                test_run_id2=result.get("test_run_id2"),
                winner_test_run_id=result.get("winner_test_run_id"),
                explanation=result.get("explanation", ""),
                created_at=datetime.now(),
                compare_within_version=result.get("compare_within_version", False)
            ))
        
        return BatchComparisonResponse(
            results=comparison_results,
            total_comparisons=len(comparison_results)
        )
    except Exception as e:
        logger.error(f"Error in batch_compare_test_runs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/elo-ratings/bulk", response_model=BulkEloRatingsResponse)
async def get_bulk_elo_ratings(
    request: BulkEloRatingsRequest,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Get ELO ratings for multiple test run IDs in bulk.
    """
    try:
        ratings = neo4j_service.get_bulk_elo_ratings(request.test_run_ids)
        results = []
        for elo_rating in ratings:
            results.append(EloRatingResult(
                id=elo_rating.get("id"),
                test_run_id=elo_rating.get("test_run_id"),
                elo_score=elo_rating.get("elo_score", 1000),
                version_elo_score=elo_rating.get("version_elo_score", 1000),
                global_elo_score=elo_rating.get("global_elo_score", 1000),
                updated_at=elo_rating.get("updated_at", datetime.now())
            ))
        return BulkEloRatingsResponse(results=results, total=len(results))
    except Exception as e:
        logger.error(f"Error in get_bulk_elo_ratings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/elo-rating/{test_run_id}", response_model=EloRatingResult)
async def get_elo_rating(
    test_run_id: str,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Get the ELO rating for a test run.
    """
    try:
        # Get the ELO rating from the database
        elo_rating = neo4j_service.get_elo_rating(test_run_id)
        logger.debug(f"ELO rating raw result for test_run_id={test_run_id}: {elo_rating}")
        if elo_rating and "updated_at" in elo_rating:
            logger.debug(f"ELO rating updated_at type: {type(elo_rating['updated_at'])}, value: {elo_rating['updated_at']}")
        if not elo_rating:
            raise HTTPException(status_code=404, detail=f"ELO rating for test run {test_run_id} not found")
        
        return EloRatingResult(
            id=elo_rating.get("id"),
            test_run_id=test_run_id,
            elo_score=elo_rating.get("elo_score", 1000),
            version_elo_score=elo_rating.get("version_elo_score", 1000),
            global_elo_score=elo_rating.get("global_elo_score", 1000),
            updated_at=elo_rating.get("updated_at", datetime.now())
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ELO rating: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/best-test-run/{prompt_version_id}")
async def get_best_test_run(
    prompt_version_id: str,
    ab_evaluator: ABTestingEvaluator = Depends(get_ab_evaluator),
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Get the best test run for a prompt version based on ELO ratings.
    """
    try:
        # Find the best test run
        best_test_run_id = ab_evaluator.find_best_test_run_for_prompt_version(prompt_version_id)
        
        if not best_test_run_id:
            raise HTTPException(status_code=404, detail=f"No test runs found for prompt version {prompt_version_id}")
        
        # Get the test run details
        test_run = neo4j_service.get_test_run(best_test_run_id)
        
        if not test_run:
            raise HTTPException(status_code=404, detail=f"Test run with ID {best_test_run_id} not found")
            
        # Get ELO rating
        elo_rating = neo4j_service.get_elo_rating(best_test_run_id)
        
        if elo_rating:
            test_run["elo_rating"] = elo_rating
        
        return test_run
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting best test run: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/best-prompt-version/{prompt_id}")
async def get_best_prompt_version(
    prompt_id: str,
    ab_evaluator: ABTestingEvaluator = Depends(get_ab_evaluator),
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Get the best prompt version for a prompt based on ELO ratings.
    """
    try:
        # Find the best prompt version
        best_version_id = ab_evaluator.find_best_prompt_version(prompt_id)
        
        if not best_version_id:
            raise HTTPException(status_code=404, detail=f"No prompt versions found for prompt {prompt_id}")
        
        # Get the prompt version details
        prompt_version = neo4j_service.get_prompt_version(best_version_id)
        
        if not prompt_version:
            raise HTTPException(status_code=404, detail=f"Prompt version with ID {best_version_id} not found")
            
        # Get the best test run for this version
        best_test_run_id = ab_evaluator.find_best_test_run_for_prompt_version(best_version_id)
        
        if best_test_run_id:
            # Get the test run details
            test_run = neo4j_service.get_test_run(best_test_run_id)
            if test_run:
                prompt_version["best_test_run"] = test_run
                
            # Get ELO rating
            elo_rating = neo4j_service.get_elo_rating(best_test_run_id)
            if elo_rating:
                prompt_version["elo_rating"] = elo_rating
        
        return prompt_version
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting best prompt version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/comparisons/{test_run_id}")
async def get_comparisons_for_test_run(
    test_run_id: str,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """
    Get all comparisons involving a specific test run.
    """
    try:
        # Get the comparisons from the database
        comparisons = neo4j_service.get_comparison_results(test_run_id)
        
        return comparisons
    except Exception as e:
        logger.error(f"Error getting comparisons: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/comparisons/bulk")
async def get_bulk_comparison_results(
    request: dict,  # expects {"test_run_ids": [...]}
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    """Get all comparison results for a list of test run IDs in bulk."""
    try:
        all_comparisons = []
        for test_run_id in request.get("test_run_ids", []):
            comparisons = neo4j_service.get_comparison_results(test_run_id)
            if isinstance(comparisons, list):
                all_comparisons.extend(comparisons)
            elif comparisons:
                all_comparisons.append(comparisons)
        return {"results": all_comparisons, "total": len(all_comparisons)}
    except Exception as e:
        logger.error(f"Error in get_bulk_comparison_results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
