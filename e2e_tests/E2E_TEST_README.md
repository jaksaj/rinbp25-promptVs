# End-to-End Test Suite for PromptVs API

This directory contains a comprehensive end-to-end test suite that validates the entire PromptVs API workflow.

## Files

- `e2e_test.py` - Main end-to-end test script
- `run_e2e_test.py` - Simple runner script that sets up environment and runs tests
- `test_prompts_config.json` - Configuration file containing test prompts and expected solutions
- `E2E_TEST_README.md` - This file

## Test Flow

The end-to-end test performs the following steps:

1. **Start Ollama and Models**
   - Checks if Ollama service is running
   - Pulls required models if not available
   - Starts models needed for testing

2. **Create Prompt Group**
   - Creates a new prompt group for organizing test prompts

3. **Create Test Prompts**
   - Creates 2 test prompts from the configuration file:
     - Math Problem Solver
     - Creative Writing

4. **Create Prompt Versions**
   - For each prompt, creates 2 versions using different prompting techniques:
     - `cot_simple` (Chain of Thought - Simple)
     - `cot_reasoning` (Chain of Thought - Reasoning)

5. **Run Test Runs**
   - Executes each prompt version 2 times with different models
   - Uses models: `deepseek-r1:1.5b`, `gemma3:1b`
   - Total test runs: 2 prompts × 2 versions × 2 models × 2 runs = 16 test runs

6. **Evaluate Results**
   - Evaluates each test run using the evaluation model (`gemma3:4b`)
   - Compares actual output against expected solutions

7. **Compare Versions**
   - Performs A/B testing between different prompt versions
   - Determines which prompting technique performs better

8. **Generate Report**
   - Creates comprehensive report with all results
   - Identifies best performing versions for each prompt

## Prerequisites

1. **API Server Running**
   ```powershell
   python run.py
   ```

2. **Ollama Installed**
   - Download and install Ollama from https://ollama.ai
   - Ensure it's accessible from command line

3. **Neo4j Database**
   - Running Neo4j instance (local or remote)
   - Properly configured in app/core/config.py

4. **Python Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

## Running the Tests

### Option 1: Use the Simple Runner (Recommended)

```powershell
python run_e2e_test.py
```

This script will:
- Check dependencies
- Start the API server if needed
- Run the complete test suite

### Option 2: Run the Test Script Directly

```powershell
python e2e_test.py
```

Optional arguments:
- `--api-url` - API base URL (default: http://localhost:8000)
- `--config` - Configuration file path (default: test_prompts_config.json)
- `--verbose` - Enable verbose logging

Example:
```powershell
python e2e_test.py --api-url http://localhost:8000 --config test_prompts_config.json --verbose
```

## Configuration

Edit `test_prompts_config.json` to customize the test prompts:

```json
{
  "prompt_group": {
    "name": "E2E Test Prompt Group",
    "description": "Prompt group for end-to-end testing",
    "tags": ["e2e", "testing", "automation"]
  },
  "prompts": [
    {
      "name": "Your Prompt Name",
      "description": "Description of what the prompt does",
      "content": "The actual prompt content",
      "expected_solution": "Expected output or solution",
      "tags": ["tag1", "tag2"]
    }
  ]
}
```

## Output

The test generates several output files:

1. **Log File**: `e2e_test_YYYYMMDD_HHMMSS.log`
   - Detailed execution log with timestamps
   - Error messages and debugging information

2. **Test Report**: `e2e_test_report_YYYYMMDD_HHMMSS.json`
   - Comprehensive JSON report with all results
   - Performance metrics and comparisons
   - Best performing versions identified

## Test Report Structure

```json
{
  "test_summary": {
    "timestamp": "2025-05-25T...",
    "total_prompts": 2,
    "total_versions": 4,
    "total_test_runs": 16,
    "total_evaluations": 16,
    "total_comparisons": 8,
    "models_tested": ["deepseek-r1:1.5b", "gemma3:1b"],
    "evaluation_model": "gemma3:4b"
  },
  "technique_performance": {
    "cot_simple": 0.85,
    "cot_reasoning": 0.92
  },
  "best_versions_per_prompt": {
    "prompt_id_1": {
      "best_version_id": "version_id",
      "average_score": 0.92,
      "all_versions": {...}
    }
  },
  "detailed_results": {
    "evaluations": [...],
    "comparisons": [...],
    "test_runs_by_version": {...},
    "versions_by_prompt": {...}
  }
}
```

## Troubleshooting

### Common Issues

1. **API Server Not Running**
   ```
   Error: API health check failed
   ```
   **Solution**: Start the API server with `python run.py`

2. **Ollama Not Available**
   ```
   Error: Failed to start Ollama service
   ```
   **Solution**: Install Ollama and ensure it's in your PATH

3. **Models Not Pulling**
   ```
   Error: Failed to pull model
   ```
   **Solution**: Check internet connection and Ollama service status

4. **Neo4j Connection Issues**
   ```
   Error: Neo4j connection failed
   ```
   **Solution**: Verify Neo4j is running and connection settings in config.py

### Debug Mode

Run with verbose logging to see detailed execution:
```powershell
python e2e_test.py --verbose
```

### Manual Testing

You can test individual API endpoints using the provided curl commands in `curl_commands.md`.

## Extending the Tests

To add more test scenarios:

1. **Add More Prompts**: Edit `test_prompts_config.json`
2. **Add More Techniques**: Modify the `techniques` list in `create_prompt_versions()`
3. **Add More Models**: Update the `test_models` list in the `E2ETestRunner` class
4. **Add Custom Evaluations**: Extend the evaluation logic in `evaluate_test_runs()`

## Performance Considerations

- Each test run can take 2-5 minutes depending on model performance
- Models need to be downloaded first time (several GB)
- Large test suites may require significant disk space for logs and reports

## License

This test suite is part of the PromptVs project and follows the same license terms.
