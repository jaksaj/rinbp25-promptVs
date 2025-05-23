import logging
from app.services.ollama import OllamaService
from app.services.neo4j import Neo4jService
from typing import Dict, Any, Optional, List, Tuple
import json
import math

logger = logging.getLogger(__name__)

class EloRatingSystem:
    """
    Implements an ELO rating system for evaluating and comparing test runs.
    """
    
    # Constants for ELO calculation
    DEFAULT_ELO = 1000
    K_FACTOR = 32  # Determines how much ratings change after each comparison
    
    @staticmethod
    def calculate_expected_score(rating_a: int, rating_b: int) -> float:
        """
        Calculate the expected score (probability of winning) for a player with rating_a
        against a player with rating_b using the ELO formula.
        
        Args:
            rating_a: The ELO rating of player A
            rating_b: The ELO rating of player B
            
        Returns:
            The expected score (winning probability) for player A
        """
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))
    
    @staticmethod
    def update_elo_ratings(rating_a: int, rating_b: int, score_a: float) -> Tuple[int, int]:
        """
        Update ELO ratings based on the actual score of player A.
        
        Args:
            rating_a: The current ELO rating of player A
            rating_b: The current ELO rating of player B
            score_a: The actual score of player A (1 for win, 0 for loss, 0.5 for draw)
            
        Returns:
            Tuple containing (new_rating_a, new_rating_b)
        """
        expected_a = EloRatingSystem.calculate_expected_score(rating_a, rating_b)
        
        # Update ratings
        new_rating_a = int(rating_a + EloRatingSystem.K_FACTOR * (score_a - expected_a))
        new_rating_b = int(rating_b + EloRatingSystem.K_FACTOR * ((1 - score_a) - (1 - expected_a)))
        
        return new_rating_a, new_rating_b


class ABTestingEvaluator:
    """
    Evaluates test runs using A/B testing and an ELO rating system.
    """
    
    def __init__(self, ollama_service: OllamaService, evaluation_model: str = "gemma3:4b", neo4j_service: Optional[Neo4jService] = None):
        """
        Initialize the A/B testing evaluator.
        
        Args:
            ollama_service: The OllamaService instance to use for processing prompts
            evaluation_model: The name of the model to use for evaluation
            neo4j_service: Optional Neo4jService for storing evaluation results
        """
        self.ollama_service = ollama_service
        self.evaluation_model = evaluation_model
        self.neo4j_service = neo4j_service
        self.elo_system = EloRatingSystem()
    
    async def compare_test_runs(self, test_run_id1: str, test_run_id2: str, compare_within_version: bool = False) -> Dict[str, Any]:
        """
        Compare two test runs using an LLM evaluator to determine which one is better.
        
        Args:
            test_run_id1: ID of the first test run
            test_run_id2: ID of the second test run
            compare_within_version: Whether this comparison is between test runs of the same prompt version
            
        Returns:
            Comparison result with winner and explanation
        """
        if not self.neo4j_service:
            raise ValueError("Neo4j service is required for test run comparison")
        
        # Get test run details
        test_run1 = self.neo4j_service.get_test_run(test_run_id1)
        test_run2 = self.neo4j_service.get_test_run(test_run_id2)
        
        if not test_run1 or not test_run2:
            missing_id = test_run_id1 if not test_run1 else test_run_id2
            raise ValueError(f"Test run with ID {missing_id} not found")
            
        # Get the prompt versions
        prompt_version_id1 = test_run1.get("prompt_version_id")
        prompt_version_id2 = test_run2.get("prompt_version_id")
        
        prompt_version1 = self.neo4j_service.get_prompt_version(prompt_version_id1)
        prompt_version2 = self.neo4j_service.get_prompt_version(prompt_version_id2)
        
        if not prompt_version1 or not prompt_version2:
            missing_id = prompt_version_id1 if not prompt_version1 else prompt_version_id2
            raise ValueError(f"Prompt version with ID {missing_id} not found")
          # Get the original prompt as context
        prompt_id1 = prompt_version1.get("prompt_id")
        prompt1 = self.neo4j_service.get_prompt(prompt_id1)
        
        # Check if they have the same prompt parent if comparing within version
        if compare_within_version and prompt_version1.get("prompt_id") != prompt_version2.get("prompt_id"):
            raise ValueError("For within-version comparison, test runs must be from the same prompt")
            
        # Compare the outputs
        output1 = test_run1.get("output", "")
        output2 = test_run2.get("output", "")
        
        # Construct the comparison prompt
        # If comparing within the same version, use prompt version content
        # Otherwise use original prompt content (with fallback to version content)
        if compare_within_version:
            # For within-version comparison, use the prompt version content
            prompt_text = prompt_version1.get("content", "")
        else:
            # For across-version comparison, use the original prompt content with fallback
            prompt_text = prompt1.get("content", prompt_version1.get("content", ""))
        expected_solution = (prompt1.get("expected_solution") or 
                           prompt_version1.get("expected_solution") or 
                           "")
        
        prompt = f"""
You are an expert evaluator tasked with comparing two different AI-generated responses to the same prompt.

ORIGINAL PROMPT:
{prompt_text}

EXPECTED SOLUTION (if available):
{expected_solution}

RESPONSE A:
{output1}

RESPONSE B:
{output2}

Compare the two responses and determine which one better answers the original prompt. Consider factors like:
1. Accuracy and correctness of information
2. Comprehensiveness - does it address all aspects of the prompt?
3. Clarity and organization
4. Relevance to the prompt
5. Quality and usefulness of the response

Return your evaluation as a JSON object with these keys:
- "winner": Either "A" or "B" (the better response)
- "explanation": A detailed explanation of why the chosen response is better

Only return the JSON object, with no other text.
"""

        try:
            # Process the comparison prompt
            result = self.ollama_service.process_prompt(prompt, self.evaluation_model)
            
            # Parse the result
            try:
                result = result.strip()
                
                # Handle case where JSON might be embedded in a code block
                if "```json" in result:
                    json_content = result.split("```json")[1].split("```")[0].strip()
                    comparison = json.loads(json_content)
                elif "```" in result:
                    json_content = result.split("```")[1].strip()
                    comparison = json.loads(json_content)
                else:
                    comparison = json.loads(result)
                
                # Map "A" or "B" to the actual test run IDs
                winner_key = comparison.get("winner", "").upper()
                if winner_key == "A":
                    winner_test_run_id = test_run_id1
                elif winner_key == "B":
                    winner_test_run_id = test_run_id2
                else:
                    raise ValueError(f"Invalid winner value: {winner_key}")
                
                # Create comparison result data
                comparison_data = {
                    "test_run_id1": test_run_id1,
                    "test_run_id2": test_run_id2,
                    "winner_test_run_id": winner_test_run_id,
                    "explanation": comparison.get("explanation", ""),
                    "compare_within_version": compare_within_version
                }
                
                # Save to database
                if self.neo4j_service:
                    comparison_result = self.neo4j_service.create_comparison_result(comparison_data)
                    
                    # Update ELO ratings
                    self._update_elo_ratings(test_run_id1, test_run_id2, winner_test_run_id, compare_within_version)
                    
                    return comparison_result
                
                return comparison_data
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing comparison result: {str(e)}")
                raise ValueError(f"Failed to parse comparison result: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error during test run comparison: {str(e)}")
            raise
    
    def _update_elo_ratings(self, test_run_id1: str, test_run_id2: str, winner_test_run_id: str, compare_within_version: bool) -> None:
        """
        Update ELO ratings for both test runs based on the comparison result.
        
        Args:
            test_run_id1: ID of the first test run
            test_run_id2: ID of the second test run
            winner_test_run_id: ID of the winning test run
            compare_within_version: Whether this is a within-version comparison
        """
        try:
            # Get current ELO ratings
            elo1 = self.neo4j_service.get_elo_rating(test_run_id1)
            elo2 = self.neo4j_service.get_elo_rating(test_run_id2)
            
            # Use default ELO if not found
            elo1_score = elo1.get("elo_score", EloRatingSystem.DEFAULT_ELO) if elo1 else EloRatingSystem.DEFAULT_ELO
            elo2_score = elo2.get("elo_score", EloRatingSystem.DEFAULT_ELO) if elo2 else EloRatingSystem.DEFAULT_ELO
            
            # For within-version comparisons, use version_elo_score
            elo1_version_score = elo1.get("version_elo_score", EloRatingSystem.DEFAULT_ELO) if elo1 else EloRatingSystem.DEFAULT_ELO
            elo2_version_score = elo2.get("version_elo_score", EloRatingSystem.DEFAULT_ELO) if elo2 else EloRatingSystem.DEFAULT_ELO
            
            # For across-version comparisons, use global_elo_score
            elo1_global_score = elo1.get("global_elo_score", EloRatingSystem.DEFAULT_ELO) if elo1 else EloRatingSystem.DEFAULT_ELO
            elo2_global_score = elo2.get("global_elo_score", EloRatingSystem.DEFAULT_ELO) if elo2 else EloRatingSystem.DEFAULT_ELO
            
            # Set scores based on winner
            score_a = 1.0 if winner_test_run_id == test_run_id1 else 0.0
            
            # Update overall ELO scores
            new_elo1, new_elo2 = EloRatingSystem.update_elo_ratings(elo1_score, elo2_score, score_a)
            
            # Update appropriate category ELO scores
            if compare_within_version:
                new_elo1_version, new_elo2_version = EloRatingSystem.update_elo_ratings(elo1_version_score, elo2_version_score, score_a)
                new_elo1_global, new_elo2_global = elo1_global_score, elo2_global_score  # Unchanged
            else:
                new_elo1_version, new_elo2_version = elo1_version_score, elo2_version_score  # Unchanged
                new_elo1_global, new_elo2_global = EloRatingSystem.update_elo_ratings(elo1_global_score, elo2_global_score, score_a)
            
            # Save updated ratings
            self.neo4j_service.create_elo_rating({
                "test_run_id": test_run_id1,
                "elo_score": new_elo1,
                "version_elo_score": new_elo1_version,
                "global_elo_score": new_elo1_global
            })
            
            self.neo4j_service.create_elo_rating({
                "test_run_id": test_run_id2,
                "elo_score": new_elo2,
                "version_elo_score": new_elo2_version,
                "global_elo_score": new_elo2_global
            })
            
        except Exception as e:
            logger.error(f"Error updating ELO ratings: {str(e)}")
            raise
    
    async def batch_compare_test_runs(self, test_run_pairs: List[Dict[str, str]], compare_within_version: bool = False) -> List[Dict[str, Any]]:
        """
        Compare multiple pairs of test runs in batch.
        
        Args:
            test_run_pairs: List of dictionaries with test_run_id1 and test_run_id2 keys
            compare_within_version: Whether these are within-version comparisons
            
        Returns:
            List of comparison results
        """
        results = []
        
        for pair in test_run_pairs:
            try:
                test_run_id1 = pair.get("test_run_id1")
                test_run_id2 = pair.get("test_run_id2")
                
                if not test_run_id1 or not test_run_id2:
                    logger.warning(f"Invalid test run pair: {pair}")
                    continue
                
                comparison = await self.compare_test_runs(
                    test_run_id1, 
                    test_run_id2,
                    compare_within_version
                )
                
                results.append(comparison)
                
            except Exception as e:
                logger.error(f"Error comparing test run pair {pair}: {str(e)}")
                continue
        
        return results
    
    def find_best_test_run_for_prompt_version(self, prompt_version_id: str) -> Optional[str]:
        """
        Find the test run with the highest version-ELO rating for a prompt version.
        
        Args:
            prompt_version_id: ID of the prompt version
            
        Returns:
            ID of the best test run or None if not found
        """
        if not self.neo4j_service:
            raise ValueError("Neo4j service is required")
            
        # Get all test runs for this prompt version
        test_runs = self.neo4j_service.get_test_runs_for_prompt_version(prompt_version_id)
        
        if not test_runs:
            return None
            
        best_test_run_id = None
        highest_elo = -1
        
        for test_run in test_runs:
            test_run_id = test_run.get("id")
            elo_rating = self.neo4j_service.get_elo_rating(test_run_id)
            
            if not elo_rating:
                # If no ELO rating exists, initialize one with default value
                elo_rating = self.neo4j_service.create_elo_rating({
                    "test_run_id": test_run_id,
                    "elo_score": EloRatingSystem.DEFAULT_ELO,
                    "version_elo_score": EloRatingSystem.DEFAULT_ELO,
                    "global_elo_score": EloRatingSystem.DEFAULT_ELO
                })
            
            version_elo = elo_rating.get("version_elo_score", EloRatingSystem.DEFAULT_ELO)
            
            if version_elo > highest_elo:
                highest_elo = version_elo
                best_test_run_id = test_run_id
        
        return best_test_run_id
    
    def find_best_prompt_version(self, prompt_id: str) -> Optional[str]:
        """
        Find the prompt version with the best test run based on global ELO ratings.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            ID of the best prompt version or None if not found
        """
        if not self.neo4j_service:
            raise ValueError("Neo4j service is required")
            
        # Get all versions for this prompt
        prompt_versions = self.neo4j_service.get_prompt_versions(prompt_id)
        
        if not prompt_versions:
            return None
            
        best_prompt_version_id = None
        highest_elo = -1
        
        for version in prompt_versions:
            version_id = version.get("id")
            
            # Find the best test run for this version
            best_test_run_id = self.find_best_test_run_for_prompt_version(version_id)
            
            if not best_test_run_id:
                continue
                
            # Get its global ELO rating
            elo_rating = self.neo4j_service.get_elo_rating(best_test_run_id)
            
            if not elo_rating:
                continue
                
            global_elo = elo_rating.get("global_elo_score", EloRatingSystem.DEFAULT_ELO)
            
            if global_elo > highest_elo:
                highest_elo = global_elo
                best_prompt_version_id = version_id
        
        return best_prompt_version_id
