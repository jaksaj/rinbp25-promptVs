import logging
from app.services.ollama import OllamaService
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

class LLMEvaluator:
    """
    Uses a larger LLM model to evaluate the accuracy of answers provided by other models.
    This allows for semantic evaluation rather than simple string matching.
    """
    
    def __init__(self, ollama_service: OllamaService, evaluation_model: str = "gemma3:4b"):
        """
        Initialize the LLM evaluator with a service and model to use for evaluations.
        
        Args:
            ollama_service: The OllamaService instance to use for processing prompts
            evaluation_model: The name of the model to use for evaluation (should be larger than test models)
        """
        self.ollama_service = ollama_service
        self.evaluation_model = evaluation_model
        
    async def evaluate_accuracy(self, question: str, expected_answer: str, actual_answer: str) -> Dict[str, Any]:
        """
        Evaluate whether the actual answer correctly addresses the question compared to the expected answer.
        
        Args:
            question: The original question
            expected_answer: The reference (correct) answer
            actual_answer: The generated answer to evaluate
            
        Returns:
            Dict containing evaluation results with score and explanation
        """
        prompt = f"""
You are an expert evaluator tasked with determining whether a given answer is correct.

Question: {question}
Expected Answer: {expected_answer}
Actual Answer: {actual_answer}

Evaluate the actual answer's accuracy compared to the expected answer. Consider semantic equivalence, not just exact wording.
Your evaluation should include:
1. Whether the actual answer is correct (Yes/No)
2. A score from 0.0 to 1.0 (0 = completely wrong, 1 = completely correct)
3. A brief explanation of your rating

Return your evaluation as a JSON object with these keys: "is_correct" (boolean), "score" (float), "explanation" (string).
Only return the JSON object, with no other text.
"""

        try:
            # Process the evaluation prompt with the larger model
            result = self.ollama_service.process_prompt(prompt, self.evaluation_model)
            
            # Try to parse the result as JSON
            try:
                # Extract JSON if it's embedded in text
                result = result.strip()
                
                # Handle case where JSON might be embedded in a code block or surrounded by text
                if "```json" in result:
                    json_content = result.split("```json")[1].split("```")[0].strip()
                    evaluation = json.loads(json_content)
                elif "```" in result:
                    json_content = result.split("```")[1].strip()
                    evaluation = json.loads(json_content)
                else:
                    evaluation = json.loads(result)
                    
                # Ensure all required fields are present
                if not all(key in evaluation for key in ["is_correct", "score", "explanation"]):
                    raise ValueError("Evaluation is missing required fields")
                
                return evaluation
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing evaluation result: {str(e)}")
                # Fall back to a simple extraction method if JSON parsing fails
                is_correct = "yes" in result.lower() and "no" not in result.lower()[:20]  # Check beginning for "no"
                
                # Try to extract a score if present (simple heuristic)
                import re
                score_match = re.search(r'score[:\s]+([0-9.]+)', result, re.IGNORECASE)
                score = float(score_match.group(1)) if score_match else (1.0 if is_correct else 0.0)
                
                return {
                    "is_correct": is_correct,
                    "score": score,
                    "explanation": "Failed to parse structured evaluation. This is a heuristic result.",
                    "raw_response": result
                }
                
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return {
                "is_correct": False,
                "score": 0.0,
                "explanation": f"Evaluation failed: {str(e)}",
                "error": str(e)
            }
    
    async def evaluate_response_metrics(
        self, 
        question: str, 
        expected_answer: str, 
        actual_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate multiple dimensions of a response, including:
        - Accuracy (correctness of information)
        - Relevance (how relevant the answer is to the question)
        - Completeness (whether all parts of the question are addressed)
        - Conciseness (appropriate level of detail without unnecessary information)
        
        Args:
            question: The original question
            expected_answer: The reference (correct) answer
            actual_answer: The generated answer to evaluate
            
        Returns:
            Dict containing evaluation metrics across multiple dimensions
        """
        prompt = f"""
You are an expert evaluator tasked with providing a detailed assessment of an answer to a question.

Question: {question}
Expected Answer: {expected_answer}
Actual Answer: {actual_answer}

Evaluate the actual answer on the following dimensions:
1. Accuracy (0-1): Is the information correct compared to the expected answer?
2. Relevance (0-1): How relevant is the answer to the question asked?
3. Completeness (0-1): Does the answer address all aspects of the question?
4. Conciseness (0-1): Is the answer appropriately detailed without unnecessary information?

Return your evaluation as a JSON object with these keys:
- "accuracy": float score from 0 to 1
- "relevance": float score from 0 to 1
- "completeness": float score from 0 to 1
- "conciseness": float score from 0 to 1
- "overall_score": float score from 0 to 1 (average of the above)
- "explanation": string explaining your evaluation

Only return the JSON object, with no other text.
"""

        try:
            # Process the evaluation prompt with the larger model
            result = self.ollama_service.process_prompt(prompt, self.evaluation_model)
            
            # Try to parse the result as JSON
            try:
                # Extract JSON if it's embedded in text
                result = result.strip()
                
                # Handle case where JSON might be embedded in a code block
                if "```json" in result:
                    json_content = result.split("```json")[1].split("```")[0].strip()
                    evaluation = json.loads(json_content)
                elif "```" in result:
                    json_content = result.split("```")[1].strip()
                    evaluation = json.loads(json_content)
                else:
                    evaluation = json.loads(result)
                
                # Calculate overall score if not provided
                if "overall_score" not in evaluation:
                    scores = [
                        evaluation.get("accuracy", 0),
                        evaluation.get("relevance", 0),
                        evaluation.get("completeness", 0),
                        evaluation.get("conciseness", 0)
                    ]
                    evaluation["overall_score"] = sum(scores) / len(scores)
                
                return evaluation
                
            except (json.JSONDecodeError, ValueError):
                logger.error("Failed to parse structured evaluation metrics")
                # Return a default error response
                return {
                    "accuracy": 0.0,
                    "relevance": 0.0, 
                    "completeness": 0.0,
                    "conciseness": 0.0,
                    "overall_score": 0.0,
                    "explanation": "Failed to parse evaluation metrics",
                    "raw_response": result
                }
                
        except Exception as e:
            logger.error(f"Error during metrics evaluation: {str(e)}")
            return {
                "accuracy": 0.0,
                "relevance": 0.0,
                "completeness": 0.0, 
                "conciseness": 0.0,
                "overall_score": 0.0,
                "explanation": f"Evaluation failed: {str(e)}",
                "error": str(e)
            }